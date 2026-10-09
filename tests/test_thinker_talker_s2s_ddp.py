"""Actual two-process gloo integration, tiny Qwen + frozen mock SenseVoice."""
import argparse
import json
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_thinker_talker_s2s import tiny, Tokenizer, sample
from dataset.thinker_talker_s2s import build_s2s_batch
from trainer import train_thinker_talker_s2s as cli


def _worker(rank, rendezvous, root):
    from pathlib import Path
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        for tuning in ("audio_proj", "all"):
            for scenario in ("mixed", "truncated", "empty", "text", "alternating", "missing_codes", "bad_prepare"):
                model = tiny()
                model.set_tuning(tuning)
                initial = {k: v.clone() for k, v in model.frontend.state_dict().items()}
                base_initial = {k: v.clone() for k, v in model.base.state_dict().items()}
                engine_initial = {k: v.clone() for k, v in model.encoder._engine.state_dict().items()}
                output = Path(root) / f"{tuning}_{scenario}"
                args = argparse.Namespace(data="mock", dev_data="mock", init_checkpoint=Path(root) / "base",
                    output=output, encoder=None, tuning=tuning, device="cpu", epochs=2,
                    batch_size=1, lr=1e-3, max_seq_len=128, loss_chunk=32, grad_clip=1., seed=42)
                calls, saves, steps, reports, losses = [], [], [], [], []
                with pytest.MonkeyPatch.context() as patch:
                    def report(message, **kwargs):
                        if message.startswith("{"):
                            reports.append(json.loads(message))
                    patch.setattr(cli, "print", report, raising=False)
                    real_loss = cli.batch_loss
                    def measured_loss(model, *a, **kwargs):
                        result = real_loss(model, *a, **kwargs)
                        loss = result[0] if kwargs.get("return_components") else result
                        losses.append((model.training, float(loss.detach())))
                        return result
                    patch.setattr(cli, "batch_loss", measured_loss)
                    patch.setattr(cli, "validate_init_checkpoint", lambda *a: {"task": "s2s"})
                    patch.setattr(cli, "load_s2s_checkpoint", lambda *a, **k: (model, Tokenizer()))
                    # Two batches/rank exercise reducer reset and set_epoch.
                    patch.setattr(cli, "ThinkerTalkerS2SDataset", lambda *a, **k: [sample()] * 4)
                    def prepare(*a, **k):
                        index = len(calls)
                        calls.append(True)
                        if scenario == "bad_prepare" and rank == 1:
                            raise ValueError("injected preparation error")
                        modality = "text" if (scenario == "text" or
                            scenario == "mixed" and rank == 0 or
                            scenario == "alternating" and index % 2 == 1) else "audio"
                        length = 1 if scenario == "empty" or scenario == "truncated" and rank == 0 else 128
                        row = dict(sample(), modality=modality)
                        if scenario == "missing_codes":
                            row.pop("answer_codes", None)
                        return build_s2s_batch(Tokenizer(), [row], length)
                    patch.setattr(cli, "prepare_s2s_batch", prepare)
                    patch.setattr(cli, "scheduled_sampling", lambda batch, tokenizer: batch)
                    real_save = cli.save_s2s_checkpoint
                    def save(*a):
                        saves.append(rank)
                        real_save(*a)
                    patch.setattr(cli, "save_s2s_checkpoint", save)
                    real_step = torch.optim.AdamW.step
                    def step(optimizer, *a, **k):
                        before = [p.detach().clone() for p in model.frontend.parameters()]
                        result = real_step(optimizer, *a, **k)
                        if scenario == "text" or scenario == "alternating" and len(calls) % 2 == 0:
                            assert all(torch.equal(p, old) for p, old in zip(model.frontend.parameters(), before))
                        steps.append(True)
                        return result
                    patch.setattr(torch.optim.AdamW, "step", step)
                    fails = scenario in ("empty", "bad_prepare") or scenario == "text" and tuning == "audio_proj"
                    if fails:
                        with pytest.raises((ValueError, RuntimeError), match="supervision|zero optimizer|injected"):
                            cli.train(args)
                        assert not saves and not steps
                        assert not list(output.glob("model_epoch_*"))
                    else:
                        cli.train(args)
                        expected = 2 if tuning == "audio_proj" and scenario == "alternating" else 4
                        assert len(steps) == expected
                        assert saves == ([0, 0] if rank == 0 else [])
                        for training, split in ((True, "train"), (False, "dev")):
                            values = [value for flag, value in losses if flag == training]
                            total, count = cli.collective_sum([sum(values), len(values)], "cpu")
                            if rank == 0:
                                assert [r["epoch"] for r in reports] == [1, 2]
                                assert all(r["train_updates"] == expected // 2 for r in reports)
                                assert sum(r[split + "_batches"] for r in reports) == count
                                assert sum(r[split + "_loss"] * r[split + "_batches"] for r in reports) == pytest.approx(total)
                        for epoch in (1, 2):
                            checkpoint = output / f"model_epoch_{epoch:03d}"
                            assert (checkpoint / "speech_frontend.pt").is_file()
                            assert (checkpoint / "s2s_metadata.json").is_file()
                            assert (checkpoint / "talker.pt").is_file()
                            assert (checkpoint / "thinker" / "model.safetensors").is_file()
                            assert (checkpoint / "tokenizer.json").is_file()
                            assert json.loads((checkpoint / "multitask_metadata.json").read_text())["epoch"] == epoch
                            assert json.loads((checkpoint / "s2s_metadata.json").read_text())["tuning"] == tuning
                    state = torch.cat([p.detach().flatten() for p in model.parameters()])
                    states = [torch.empty_like(state) for _ in range(2)]
                    dist.all_gather(states, state)
                    assert torch.equal(states[0], states[1])
                    assert all(torch.equal(v, engine_initial[k]) for k, v in model.encoder._engine.state_dict().items())
                    if tuning == "audio_proj":
                        assert all(torch.equal(v, base_initial[k]) for k, v in model.base.state_dict().items())
                    if fails or scenario == "text":
                        assert all(torch.equal(v, initial[k]) for k, v in model.frontend.state_dict().items())
                    else:
                        assert any(not torch.equal(v, initial[k]) for k, v in model.frontend.state_dict().items())
        # Both ranks reject a pre-existing output consistently before model loading.
        with pytest.raises(RuntimeError, match="already exists"):
            cli.train(args)
    finally:
        dist.destroy_process_group()


def _batch_boundary_worker(rank, rendezvous, root):
    from pathlib import Path
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        for tuning in ("audio_proj", "all"):
            model = tiny()
            output = Path(root) / tuning
            args = argparse.Namespace(data="mock", dev_data=None, init_checkpoint=Path(root) / "base",
                output=output, encoder=None, tuning=tuning, device="cpu", epochs=2,
                batch_size=2, lr=1e-3, max_seq_len=128, loss_chunk=32, grad_clip=1., seed=42)
            calls, steps, saves, reports = [], [], [], []
            established = {}
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(cli, "validate_init_checkpoint", lambda *a: {"task": "s2s"})
                patch.setattr(cli, "load_s2s_checkpoint", lambda *a, **k: (model, Tokenizer()))
                patch.setattr(cli, "ThinkerTalkerS2SDataset", lambda *a, **k: [sample()] * 8)
                patch.setattr(cli, "scheduled_sampling", lambda batch, tokenizer: batch)
                def report(message, **kwargs):
                    if message.startswith("{"):
                        reports.append(json.loads(message))
                patch.setattr(cli, "print", report, raising=False)
                def prepare(model, tokenizer, raw, length, **kwargs):
                    assert len(raw) == 2
                    calls.append(True)
                    rows = [dict(sample(), modality="audio")] * 2 if len(calls) == 1 else [
                        dict(sample(frames=200), modality="audio"),
                        dict(sample(), modality="text", answer_codes=None)]
                    batch = build_s2s_batch(tokenizer, rows, length)
                    if len(calls) > 1:
                        assert cli.supervised_samples(batch).tolist() == [False, True]
                        assert batch["speech_mask"].any(dim=1).tolist() == [True, False]
                        assert cli.speech_supervised_count(batch) == 0
                    return batch
                patch.setattr(cli, "prepare_s2s_batch", prepare)
                real_step = torch.optim.AdamW.step
                def step(optimizer, *a, **k):
                    before = {key: value.clone() for key, value in model.frontend.state_dict().items()}
                    result = real_step(optimizer, *a, **k)
                    steps.append(True)
                    if len(steps) == 1:
                        assert any(not torch.equal(value, before[key])
                                   for key, value in model.frontend.state_dict().items())
                        assert any(optimizer.state[p]["exp_avg"].abs().any()
                                   for p in model.frontend.parameters() if p in optimizer.state)
                        established.update({key: value.clone() for key, value in model.frontend.state_dict().items()})
                    else:
                        assert all(torch.equal(value, before[key])
                                   for key, value in model.frontend.state_dict().items())
                    return result
                patch.setattr(torch.optim.AdamW, "step", step)
                real_save = cli.save_s2s_checkpoint
                def save(*a):
                    saves.append(a[-1])
                    real_save(*a)
                patch.setattr(cli, "save_s2s_checkpoint", save)
                if tuning == "audio_proj":
                    with pytest.raises(ValueError, match="zero optimizer updates"):
                        cli.train(args)
                    assert len(steps) == 1
                    assert saves == ([1] if rank == 0 else [])
                    assert not (output / "model_epoch_002").exists()
                else:
                    cli.train(args)
                    assert len(steps) == 4
                    assert saves == ([1, 2] if rank == 0 else [])
                assert len(calls) == 4
                assert all(torch.equal(value, established[key])
                           for key, value in model.frontend.state_dict().items())
                if rank == 0:
                    assert reports[0]["train_updates"] == (1 if tuning == "audio_proj" else 2)
                state = torch.cat([p.detach().flatten() for p in model.parameters()])
                states = [torch.empty_like(state) for _ in range(2)]
                dist.all_gather(states, state)
                assert torch.equal(states[0], states[1])
    finally:
        dist.destroy_process_group()


def test_two_process_mixed_rows_preserve_frontend_momentum(tmp_path):
    mp.spawn(_batch_boundary_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
             nprocs=2, join=True)


class MockWandb:
    """In-memory SDK stand-in: no account, disk history or network."""
    def __init__(self, fail=None):
        self.events = []
        self.fail = fail

    def Settings(self, **kwargs):
        return kwargs

    def init(self, **kwargs):
        json.dumps(kwargs)  # Path and device objects must not leak into config.
        self.events.append(("init", kwargs))
        if self.fail == "init":
            raise RuntimeError("injected wandb init")
        return self

    def define_metric(self, *args, **kwargs):
        self.events.append(("define", kwargs))

    def log(self, values):
        self.events.append(("log", dict(values)))
        if self.fail == "log":
            raise RuntimeError("injected wandb log")

    def finish(self):
        self.events.append(("finish", None))
        if self.fail == "finish":
            raise RuntimeError("injected wandb finish")


def _wandb_case(rank, root, scenario, interval=1):
    import sys
    from pathlib import Path
    world = dist.get_world_size() if dist.is_initialized() else 1
    sdk = MockWandb(fail=scenario if scenario in ("init", "log", "finish") else None)
    args = argparse.Namespace(data="private-data", dev_data="private-dev", init_checkpoint=Path(root) / "base",
        output=Path(root) / f"{scenario}_{interval}", encoder=None, tuning="all", device="cpu", epochs=2,
        batch_size=1, lr=1e-3, max_seq_len=128, loss_chunk=32, grad_clip=1., seed=42,
        wandb=True, wandb_project="test", wandb_name="mock", wandb_log_interval=interval,
        api_key="must-not-be-collected")
    model = tiny()
    observed = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "wandb", None if scenario == "missing" else sdk)
        patch.setattr(cli, "validate_init_checkpoint", lambda *a: {"task": "s2s"})
        patch.setattr(cli, "load_s2s_checkpoint", lambda *a, **k: (model, Tokenizer()))
        patch.setattr(cli, "ThinkerTalkerS2SDataset", lambda *a, **k: [sample()] * (2 * world))
        patch.setattr(cli, "save_s2s_checkpoint", lambda *a: None)
        patch.setattr(cli, "scheduled_sampling", lambda batch, tokenizer: batch)
        def prepare(*a, **k):
            # Rank zero has no supervision; only rank one belongs in the denominator.
            length = 1 if world > 1 and rank == 0 else 128
            return build_s2s_batch(Tokenizer(), [dict(sample(), modality="audio")], length)
        patch.setattr(cli, "prepare_s2s_batch", prepare)
        real_loss = cli.batch_loss
        def measure(*a, **kw):
            loss, components = real_loss(*a, **kw)
            assert not components["text_loss"].requires_grad
            assert not components["audio_loss"].requires_grad
            assert float(loss.detach()) == pytest.approx(float(sum(components.values())), rel=1e-6)
            observed.append(float(loss.detach()))
            return loss, components
        patch.setattr(cli, "batch_loss", measure)
        if scenario in ("init", "log", "finish", "missing"):
            with pytest.raises(RuntimeError, match="wandb"):
                cli.train(args)
        else:
            cli.train(args)
    if rank != 0:
        assert sdk.events == []
    else:
        if scenario not in ("init", "missing"):
            assert sdk.events[-1][0] == "finish"
        if scenario == "ok":
            assert sum(event == "init" for event, _ in sdk.events) == 1
            config = sdk.events[0][1]["config"]
            assert "data" not in config and "api_key" not in config and "init_checkpoint" not in config
            logs = [value for event, value in sdk.events if event == "log"]
            steps = [value for value in logs if "train/loss" in value]
            assert [value["global_step"] for value in steps] == list(range(interval, 5, interval))
            assert [value["epoch"] for value in steps] == [(step - 1) // 2 + 1 for step in range(interval, 5, interval)]
            assert all(value["updates"] == value["global_step"] and value["lr"] == args.lr for value in logs)
            summaries = [value for value in logs if "dev/loss" in value]
            assert len(summaries) == 2
            assert [value["global_step"] for value in summaries] == [2, 4]
            for value in logs:
                for prefix in ("train/", "train/epoch_", "dev/"):
                    if prefix + "loss" in value:
                        assert value[prefix + "loss"] == pytest.approx(
                            value[prefix + "text_loss"] + value[prefix + "audio_loss"], rel=1e-6)
    if scenario == "ok":
        assert len(observed) == (0 if world > 1 and rank == 0 else 8)
        # Ground truth from actual forward calls, including unsupervised-rank exclusion.
        total, count = cli.collective_sum([sum(observed), len(observed)], "cpu")
        if rank == 0:
            assert sum(value["train/epoch_loss"] + value["dev/loss"] for value in summaries) / 4 == pytest.approx(total / count)


def _wandb_worker(rank, rendezvous, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        for scenario in ("ok", "init", "log", "finish", "missing"):
            _wandb_case(rank, root, scenario)
    finally:
        dist.destroy_process_group()


def test_wandb_two_process_rank_zero_and_synchronized_failures(tmp_path):
    mp.spawn(_wandb_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=True)


@pytest.mark.parametrize("scenario,interval", [("ok", 1), ("ok", 3), ("init", 1), ("log", 1), ("finish", 1), ("missing", 1)])
def test_wandb_single_process(tmp_path, scenario, interval):
    _wandb_case(0, str(tmp_path), scenario, interval)


@pytest.mark.parametrize("modality,with_audio", [("audio", True), ("text", False)])
def test_wandb_components_preserve_loss_gradients_and_single_forward(monkeypatch, modality, with_audio):
    model = tiny()
    model.set_tuning("all")
    row = dict(sample(), modality=modality)
    if not with_audio:
        row["answer_codes"] = None
    batch = build_s2s_batch(Tokenizer(), [row], 128)
    calls = []
    original = model.forward_streams
    def counted(**kwargs):
        calls.append(True)
        return original(**kwargs)
    monkeypatch.setattr(model, "forward_streams", counted)
    plain = cli.batch_loss(model, batch, "cpu", 32)
    plain.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    loss, components = cli.batch_loss(model, batch, "cpu", 32, return_components=True)
    assert len(calls) == 2
    assert torch.equal(plain, loss)
    assert torch.allclose(loss.detach(), sum(components.values()))
    if not with_audio:
        assert components["audio_loss"] == 0
    loss.backward()
    assert all(torch.equal(p.grad, grads[name]) for name, p in model.named_parameters() if name in grads)


def test_wandb_disabled_does_not_import(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "wandb":
            raise AssertionError("disabled logging imported wandb")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    logger = cli.WandbLogger(argparse.Namespace(wandb=False))
    logger.start()
    logger.log({"loss": 1.0})
    logger.finish()


class RecordingProgress(cli.tqdm):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closed = False
        self.advances = 0
        self.instances.append(self)

    def update(self, amount=1):
        self.advances += amount
        return super().update(amount)

    def close(self):
        self.closed = True
        return super().close()


def _terminal_case(rank, root, fail=False):
    import io
    import re
    from contextlib import redirect_stdout
    from pathlib import Path

    class FlushedOutput(io.StringIO):
        flushes = 0

        def flush(self):
            self.flushes += 1
            return super().flush()

    world = dist.get_world_size() if dist.is_initialized() else 1
    args = argparse.Namespace(data="mock", dev_data="mock", init_checkpoint=Path(root) / "base",
        output=Path(root) / ("failed" if fail else "terminal"), encoder=None,
        tuning="audio_proj", device="cpu", epochs=1, batch_size=1, lr=1e-3,
        max_seq_len=128, loss_chunk=32, grad_clip=1., seed=42, wandb=False)
    model = tiny()
    calls, components = [], []
    RecordingProgress.instances = []
    stream = FlushedOutput()
    with pytest.MonkeyPatch.context() as patch, redirect_stdout(stream):
        patch.setattr(cli, "tqdm", RecordingProgress)
        patch.setattr(cli, "validate_init_checkpoint", lambda *a: {"task": "s2s"})
        patch.setattr(cli, "load_s2s_checkpoint", lambda *a, **k: (model, Tokenizer()))
        patch.setattr(cli, "ThinkerTalkerS2SDataset", lambda *a, **k: [sample()] * (3 * world))
        patch.setattr(cli, "save_s2s_checkpoint", lambda *a: None)
        patch.setattr(cli, "scheduled_sampling", lambda batch, tokenizer: batch)
        real_loss = cli.batch_loss

        def measured(*a, **kwargs):
            assert kwargs["return_components"]
            loss, values = real_loss(*a, **kwargs)
            assert all(not value.requires_grad for value in values.values())
            components.append(values)
            return loss, values

        def prepare(*a, **kwargs):
            index = len(calls) % 3
            calls.append(index)
            if fail and index == 1 and rank == world - 1:
                raise ValueError("terminal preparation failure")
            # First batch is empty on every rank. Second has only one valid rank.
            length = 1 if index == 0 or (index == 1 and world > 1 and rank == 0) else 128
            modality = "text" if index == 2 else "audio"
            return build_s2s_batch(Tokenizer(), [dict(sample(), modality=modality)], length)

        patch.setattr(cli, "batch_loss", measured)
        patch.setattr(cli, "prepare_s2s_batch", prepare)
        if fail:
            with pytest.raises((ValueError, RuntimeError), match="terminal preparation failure"):
                cli.train(args)
        else:
            cli.train(args)
    output = stream.getvalue()
    if rank != 0:
        assert output == "" and RecordingProgress.instances == []
        return
    assert "loading model..." in output and "loaded model" in output
    assert "loading train data..." in output and "loaded train:" in output
    assert all(progress.closed for progress in RecordingProgress.instances)
    lines = re.findall(r"(?:train|dev) epoch=1/1 batch=[^\r\n]+", output)
    assert len(lines) == (1 if fail else 6)
    assert sum(progress.advances for progress in RecordingProgress.instances) == len(lines)
    assert stream.flushes >= len(lines)
    assert "batch=1/3 global_step=0 loss=n/a text_loss=n/a audio_loss=n/a lr=0.001 skipped=no_supervision" in lines[0]
    if not fail:
        assert "batch=2/3 global_step=1" in lines[1] and lines[1].endswith("updated accumulation=1/1")
        assert lines[2].endswith("skipped=no_active_gradients accumulation=1/1")
        assert lines[-1].startswith("dev ") and lines[-1].endswith("evaluated")
        for line in (lines[1], lines[2], lines[4], lines[5]):
            values = re.search(r"loss=([\d.]+) text_loss=([\d.]+) audio_loss=([\d.]+)", line)
            assert values
            loss, text, audio = map(float, values.groups())
            assert loss == pytest.approx(text + audio, rel=1e-6, abs=2e-6)
        assert len(components) == (2 if world > 1 else 4)


def _terminal_worker(rank, rendezvous, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        _terminal_case(rank, root)
        _terminal_case(rank, root, fail=True)
    finally:
        dist.destroy_process_group()


def test_terminal_two_process_rank_zero_only_and_failure_cleanup(tmp_path):
    mp.spawn(_terminal_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=True)


@pytest.mark.parametrize("fail", [False, True])
def test_terminal_single_process_without_wandb(tmp_path, fail):
    _terminal_case(0, str(tmp_path), fail)


def test_two_process_gloo_training(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=True)


@pytest.mark.parametrize("device,expected,backend", [("cuda:7", "cuda:1", "nccl"), ("cpu", "cpu", "gloo")])
def test_torchrun_device_selection_and_teardown(monkeypatch, tmp_path, device, expected, backend):
    # Routing contract only: this deliberately does not execute real CUDA/NCCL.
    import sys
    events = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: events.append(("device", rank)))
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: events.append(("backend", kw["backend"])))
    monkeypatch.setattr(dist, "destroy_process_group", lambda: events.append(("destroy", True)))
    def train(args):
        assert args.device == expected
        raise ValueError("injected train failure")
    monkeypatch.setattr(cli, "train", train)
    monkeypatch.setattr(sys, "argv", ["train", "--data", "mock", "--init-checkpoint", str(tmp_path),
                                     "--output", str(tmp_path / "run"), "--device", device])
    with pytest.raises(ValueError, match="injected train failure"):
        cli.main()
    assert ("backend", backend) in events
    assert (("device", 1) in events) == device.startswith("cuda")
    assert events[-1] == ("destroy", True)


def _accumulation_case(rank, root, tuning, scenario="mixed"):
    """Exact gradients and log means across unequal, empty and tail windows."""
    import sys
    from pathlib import Path
    world = dist.get_world_size() if dist.is_initialized() else 1
    torch.manual_seed(42)
    model = tiny()
    sdk = MockWandb()
    args = argparse.Namespace(data="mock", dev_data=None, init_checkpoint=Path(root) / "base",
        output=Path(root) / f"{tuning}_{scenario}", encoder=None, tuning=tuning, device="cpu", epochs=1,
        batch_size=1, lr=1e-3, max_seq_len=128, loss_chunk=32, grad_clip=1e9, seed=42,
        gradient_accumulation_steps=4, wandb=True, wandb_log_interval=1)
    modes = ["audio", "text", "empty", "empty", "text", "text", "text", "text", "audio", "empty"]
    if rank == 1:
        modes[0], modes[2], modes[9] = "empty", "audio", "audio"
    if scenario == "recovery":
        modes = ["empty"] * 4 + (["audio", "empty", "audio", "empty"] if rank == 0 else
                                  ["empty", "audio", "empty", "empty"])
    elif scenario == "empty_epoch":
        modes = ["empty"] * 8
    elif scenario == "dev":
        args.dev_data = "mock-dev"
    calls, observed, gradients, boundaries, clips, zeros = [], [], [], [], [], []
    saves, reports, backwards, dev_observed, dev_state = [], [], [], [], []
    expected_gradients, expected_means = [], []
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "wandb", sdk)
        patch.setattr(cli, "validate_init_checkpoint", lambda *a: {"task": "s2s"})
        patch.setattr(cli, "load_s2s_checkpoint", lambda *a, **k: (model, Tokenizer()))
        patch.setattr(cli, "ThinkerTalkerS2SDataset", lambda *a, **k: [sample()] * (len(modes) * world))
        patch.setattr(cli, "save_s2s_checkpoint", lambda *a: saves.append(a[-1]))
        patch.setattr(cli, "scheduled_sampling", lambda batch, tokenizer: batch)
        patch.setattr(cli, "print", lambda message, **k: reports.append(json.loads(message))
                      if message.startswith("{") else None, raising=False)

        def prepare(*a, **k):
            index = len(calls) % len(modes)
            if not model.training:
                state = [(p.detach().clone(), None if p.grad is None else p.grad.detach().clone())
                         for p in model.parameters()]
                dev_state.append(state)
            calls.append(index)
            mode = modes[index]
            return build_s2s_batch(Tokenizer(), [dict(sample(), modality="text" if mode == "text" else "audio")],
                                   1 if mode == "empty" else 128)

        def loss(model, batch, device, chunk, *, return_components=False):
            index = calls[-1]
            coefficient = float((rank + 1) * (index + 1))
            parameter = next(model.frontend.parameters()) if modes[index] == "audio" else next(model.base.parameters())
            value = parameter.sum() * coefficient + 100
            assert torch.is_grad_enabled() == model.training
            (observed if model.training else dev_observed).append((index, float(value.detach())))
            components = {"text_loss": value.detach() * .25, "audio_loss": value.detach() * .75}
            return (value, components) if return_components else value

        patch.setattr(cli, "prepare_s2s_batch", prepare)
        patch.setattr(cli, "batch_loss", loss)
        original_step = torch.optim.AdamW.step
        original_zero = torch.optim.AdamW.zero_grad
        original_clip = torch.nn.utils.clip_grad_norm_
        original_backward = torch.Tensor.backward

        def backward(tensor, *a, **k):
            assert model.training
            backwards.append(len(calls))
            return original_backward(tensor, *a, **k)

        patch.setattr(torch.Tensor, "backward", backward)

        def zero(optimizer, *a, **k):
            zeros.append(len(calls))
            return original_zero(optimizer, *a, **k)

        def clip(*a, **k):
            clips.append(len(calls))
            return original_clip(*a, **k)

        def step(optimizer, *a, **k):
            end = len(calls)
            start = (end - 1) // 4 * 4
            valid = sum(mode != "empty" for mode in modes[start:end])
            numerator = sum((rank + 1) * (i + 1) for i in range(start, end) if modes[i] == "audio")
            metric = sum(value for i, value in observed if start <= i < end)
            numerator, metric, valid = cli.collective_sum([numerator, metric, valid], "cpu")
            expected_gradients.append(numerator / valid if numerator else None)
            expected_means.append(metric / valid)
            parameter = next(model.frontend.parameters())
            gradients.append(None if parameter.grad is None else parameter.grad.detach().clone())
            boundaries.append(end)
            return original_step(optimizer, *a, **k)

        patch.setattr(torch.optim.AdamW, "zero_grad", zero)
        patch.setattr(torch.optim.AdamW, "step", step)
        patch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
        if scenario == "empty_epoch":
            initial = {key: value.clone() for key, value in model.state_dict().items()}
            with pytest.raises(ValueError, match="train has no next-token supervision"):
                cli.train(args)
            assert not saves and not reports and not observed
            assert not list(args.output.glob("model_epoch_*"))
            assert all(torch.equal(value, initial[key]) for key, value in model.state_dict().items())
        else:
            cli.train(args)
            assert saves == ([1] if rank == 0 else [])
    assert len(calls) == len(modes) * (2 if scenario == "dev" else 1)
    assert backwards == list(range(1, len(modes) + 1))
    assert zeros == list(range(0, len(modes), 4))
    expected_boundaries = ([8] if scenario == "recovery" else [] if scenario == "empty_epoch" else
                           [4, 10] if tuning == "audio_proj" else [4, 8, 10])
    assert boundaries == clips == expected_boundaries
    if scenario == "dev":
        assert len(dev_state) == len(modes)
        final_state = [(p.detach(), p.grad) for p in model.parameters()]
        for state in [*dev_state[1:], final_state]:
            for (value, grad), (initial_value, initial_grad) in zip(state, dev_state[0]):
                assert torch.equal(value, initial_value)
                assert (grad is None) == (initial_grad is None)
                if grad is not None:
                    assert torch.equal(grad, initial_grad)
        total, count = cli.collective_sum([sum(v for _, v in dev_observed), len(dev_observed)], "cpu")
        if rank == 0:
            assert len(reports) == 1
            assert reports[0]["dev_batches"] == count
            assert reports[0]["dev_loss"] == pytest.approx(total / count)
            epoch_logs = [value for event, value in sdk.events if event == "log" and "dev/loss" in value]
            assert len(epoch_logs) == 1
            assert epoch_logs[0]["global_step"] == len(expected_boundaries)
            for key, weight in (("loss", 1), ("text_loss", .25), ("audio_loss", .75)):
                assert epoch_logs[0][f"dev/{key}"] == pytest.approx(total / count * weight)
    for gradient, expected in zip(gradients, expected_gradients):
        if expected is None:
            assert gradient is None
        else:
            torch.testing.assert_close(gradient, torch.full_like(gradient, expected))
    if rank == 0:
        logs = [value for event, value in sdk.events if event == "log" and "train/loss" in value]
        assert [value["global_step"] for value in logs] == list(range(1, len(boundaries) + 1))
        assert [value["train/loss"] for value in logs] == pytest.approx(expected_means)
        config = sdk.events[0][1]["config"]
        assert config["gradient_accumulation_steps"] == 4
        assert config["world_size"] == world and config["effective_batch_size"] == 4 * world
    else:
        assert sdk.events == []


def _accumulation_worker(rank, rendezvous, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        for tuning in ("audio_proj", "all"):
            for scenario in ("mixed", "recovery", "empty_epoch", "dev"):
                _accumulation_case(rank, root, tuning, scenario)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tuning", ["audio_proj", "all"])
@pytest.mark.parametrize("scenario", ["mixed", "recovery", "empty_epoch", "dev"])
def test_accumulation_windows_single_process(tmp_path, tuning, scenario):
    _accumulation_case(0, str(tmp_path), tuning, scenario)


def test_accumulation_windows_two_process(tmp_path):
    mp.spawn(_accumulation_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=True)

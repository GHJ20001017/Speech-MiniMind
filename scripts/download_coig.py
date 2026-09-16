#!/usr/bin/env python3
"""Download the complete BAAI/COIG repository without filtering.

This script mirrors every file exposed by the ``BAAI/COIG`` dataset repository
(default endpoint: ``https://hf-mirror.com``).  It intentionally performs no
QA/exam/value filtering; downstream scripts can read the downloaded JSONL/JSON
files and decide which rows to keep.

Features:
  * discovers files from the Hugging Face dataset tree API;
  * resumes partial downloads with ``curl --continue-at -``;
  * verifies remote file sizes;
  * verifies SHA-256 for LFS files when the API provides an LFS object id;
  * writes a machine-readable ``manifest.json``.

Example:
    python scripts/download_coig.py --output data/coig
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_REPO = "BAAI/COIG"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_OUTPUT = Path("data/coig")
DEFAULT_REVISION = "main"


def log(message: str) -> None:
    print(message, flush=True)


def dataset_tree_url(endpoint: str, repo: str, revision: str) -> str:
    quoted_repo = urllib.parse.quote(repo, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    return (
        f"{endpoint.rstrip('/')}/api/datasets/{quoted_repo}/tree/"
        f"{quoted_revision}?recursive=true&expand=false"
    )


def resolve_url(endpoint: str, repo: str, revision: str, path: str) -> str:
    quoted_repo = urllib.parse.quote(repo, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_path = urllib.parse.quote(path, safe="/")
    return (
        f"{endpoint.rstrip('/')}/datasets/{quoted_repo}/resolve/"
        f"{quoted_revision}/{quoted_path}"
    )


def fetch_tree(endpoint: str, repo: str, revision: str) -> list[dict]:
    url = dataset_tree_url(endpoint, repo, revision)
    request = urllib.request.Request(url, headers={"User-Agent": "Speech-MiniMind/1.0"})
    log(f"fetching file tree: {url}")
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise RuntimeError(f"unexpected tree response for {repo}: {payload!r}")
    files = [item for item in payload if item.get("type") == "file"]
    files.sort(key=lambda item: item["path"])
    if not files:
        raise RuntimeError(f"no files found in {repo}")
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_lfs_sha(item: dict) -> str | None:
    lfs = item.get("lfs")
    if isinstance(lfs, dict):
        value = lfs.get("oid") or lfs.get("sha256")
        return str(value).lower() if value else None
    return None


def verify_file(path: Path, expected_size: int | None, lfs_sha: str | None, verify_sha: bool) -> None:
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise RuntimeError(f"size mismatch for {path}: {actual_size} != {expected_size}")
    if lfs_sha and verify_sha:
        actual_sha = sha256_file(path)
        if actual_sha.lower() != lfs_sha:
            raise RuntimeError(f"sha256 mismatch for {path}: {actual_sha} != {lfs_sha}")


def curl_download(url: str, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not partial.exists():
        destination.replace(partial)
    elif destination.exists() and partial.exists():
        # Keep the larger file as the resume source rather than discarding data.
        if destination.stat().st_size > partial.stat().st_size:
            destination.replace(partial)

    command = [
        "curl",
        "--location",
        "--fail",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--retry-connrefused",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    log(f"downloading {url}")
    subprocess.run(command, check=True)
    if not partial.exists() or partial.stat().st_size == 0:
        raise RuntimeError(f"download produced no data: {url}")


def download_one(
    item: dict,
    endpoint: str,
    repo: str,
    revision: str,
    output: Path,
    overwrite: bool,
    verify_sha: bool,
    dry_run: bool,
) -> dict:
    relative_path = str(item["path"])
    destination = output / relative_path
    partial = destination.with_name(destination.name + ".part")
    expected_size = item.get("size")
    lfs_sha = expected_lfs_sha(item)
    url = resolve_url(endpoint, repo, revision, relative_path)

    result = {
        "path": relative_path,
        "url": url,
        "expected_size": expected_size,
        "lfs_sha256": lfs_sha,
        "status": "pending",
    }

    if dry_run:
        result["status"] = "dry-run"
        log(f"would download {relative_path} ({expected_size or 'unknown'} bytes)")
        return result

    if destination.exists() and not overwrite:
        try:
            verify_file(destination, expected_size, lfs_sha, verify_sha)
        except RuntimeError:
            pass
        else:
            if partial.exists():
                partial.unlink()
            result["actual_size"] = destination.stat().st_size
            result["status"] = "skipped"
            log(f"skip complete {relative_path} ({destination.stat().st_size} bytes)")
            return result

    if overwrite:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    started = time.time()
    curl_download(url, destination)
    verify_file(partial, expected_size, lfs_sha, verify_sha)
    partial.replace(destination)
    result["actual_size"] = destination.stat().st_size
    result["elapsed_seconds"] = round(time.time() - started, 2)
    result["status"] = "downloaded"
    log(f"downloaded {relative_path} ({destination.stat().st_size} bytes, {result['elapsed_seconds']}s)")
    return result


def write_manifest(output: Path, repo: str, endpoint: str, revision: str, results: list[dict]) -> Path:
    manifest = {
        "repo": repo,
        "endpoint": endpoint,
        "revision": revision,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "file_count": len(results),
        "downloaded_count": sum(item["status"] == "downloaded" for item in results),
        "skipped_count": sum(item["status"] == "skipped" for item in results),
        "dry_run_count": sum(item["status"] == "dry-run" for item in results),
        "files": results,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repo id")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Hugging Face-compatible endpoint")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="branch, tag, or commit")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination directory")
    parser.add_argument("--overwrite", action="store_true", help="re-download files even if they already exist")
    parser.add_argument("--no-verify-sha", action="store_true", help="skip LFS SHA-256 verification")
    parser.add_argument("--dry-run", action="store_true", help="list files and planned downloads only")
    parser.add_argument("--list-only", action="store_true", help="print discovered file paths and exit")
    args = parser.parse_args()

    if shutil.which("curl") is None:
        raise SystemExit("curl is required for reliable resumed downloads")

    files = fetch_tree(args.endpoint, args.repo, args.revision)
    if args.list_only:
        for item in files:
            print(item["path"])
        return

    args.output.mkdir(parents=True, exist_ok=True)
    log(f"repo={args.repo} revision={args.revision} files={len(files)} output={args.output}")

    results: list[dict] = []
    failures: list[tuple[str, str]] = []
    for item in files:
        try:
            results.append(
                download_one(
                    item=item,
                    endpoint=args.endpoint,
                    repo=args.repo,
                    revision=args.revision,
                    output=args.output,
                    overwrite=args.overwrite,
                    verify_sha=not args.no_verify_sha,
                    dry_run=args.dry_run,
                )
            )
        except Exception as error:  # noqa: BLE001 - keep mirroring remaining files
            result = {
                "path": str(item.get("path", "")),
                "expected_size": item.get("size"),
                "lfs_sha256": expected_lfs_sha(item),
                "status": "failed",
                "error": repr(error),
            }
            results.append(result)
            failures.append((result["path"], repr(error)))
            log(f"FAILED {result['path']}: {error}")

    manifest_path = write_manifest(args.output, args.repo, args.endpoint, args.revision, results)
    log(f"manifest: {manifest_path}")

    if failures:
        log(f"completed with {len(failures)} failure(s)")
        sys.exit(1)
    if args.dry_run:
        log("dry-run complete; no files were downloaded")
    else:
        log("all files are present and verified")


if __name__ == "__main__":
    main()

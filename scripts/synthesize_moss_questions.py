"""Multi-GPU, resumable Qwen3-TTS synthesis for stage-2 manifests."""
from __future__ import annotations
import argparse,json,os,multiprocessing as mp
from pathlib import Path
import soundfile as sf

def worker(gpu, items, out, model_dir, speaker):
 os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 from qwen_tts import Qwen3TTSModel
 model=Qwen3TTSModel.from_pretrained(model_dir,device_map='cuda:0')
 for idx,row in items:
  path=out/row['audio']; path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists() and path.stat().st_size>1000: continue
  try:
   wav,sr=model.generate_custom_voice(row['instruction'],speaker=speaker,language='chinese')
   sf.write(path,wav[0],sr)
  except Exception as e:
   with open(out/'synthesis_errors.jsonl','a',encoding='utf8') as f:
    f.write(json.dumps({'index':idx,'error':repr(e),'instruction':row['instruction']},ensure_ascii=False)+'\n')

def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--model',required=True); p.add_argument('--gpus',default='0'); p.add_argument('--speaker',default='vivian')
 a=p.parse_args(); rows=[json.loads(x) for x in a.manifest.read_text(encoding='utf8').splitlines() if x.strip()]; a.output.mkdir(parents=True,exist_ok=True)
 ids=[int(x) for x in a.gpus.split(',')]; chunks=[[] for _ in ids]
 for i,r in enumerate(rows): chunks[i%len(ids)].append((i,r))
 mp.set_start_method('spawn',force=True); ps=[]
 for g,c in zip(ids,chunks):
  q=mp.Process(target=worker,args=(g,c,a.output,a.model,a.speaker)); q.start(); ps.append(q)
 for q in ps:q.join()
 print(f'finished: {len(rows)} items on GPUs {ids}')
if __name__=='__main__': main()

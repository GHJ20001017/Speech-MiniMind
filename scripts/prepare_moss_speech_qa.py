"""Extract a small Chinese multi-turn subset from moss-003 SFT data."""
from __future__ import annotations
import argparse, json, re, zipfile
from pathlib import Path

def chinese(s): return len(re.findall(r'[\u4e00-\u9fff]', s))
def english(s): return len(re.findall(r'[A-Za-z]', s))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/moss_speech_qa'))
    p.add_argument('--limit',type=int,default=1000); p.add_argument('--dev-ratio',type=float,default=.05)
    p.add_argument('--max-question-chars',type=int,default=300); p.add_argument('--max-answer-chars',type=int,default=1200)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    rows=[]; stats={'total':0,'kept':0,'rejected':{}}
    with zipfile.ZipFile(a.input) as z:
      name=next(n for n in z.namelist() if n.endswith('.jsonl'))
      with z.open(name) as f:
       for raw in f:
        stats['total']+=1
        try: obj=json.loads(raw)
        except Exception: continue
        turns=[]
        chat=obj.get('chat',{})
        values=chat.values() if isinstance(chat,dict) else chat if isinstance(chat,list) else []
        for t in values:
          if isinstance(t,dict):
            h=(t.get('Human') or '').strip(); m=(t.get('MOSS') or '').strip()
            h=re.sub(r'^<\|Human\|>:\s*','',h).replace('<eoh>','').strip()
            m=re.sub(r'^<\|MOSS\|>:\s*','',m).replace('<eom>','').strip()
            if h and m and h.lower()!='none' and m.lower()!='none': turns.append((h,m))
        if len(turns)<2: continue
        # use the final turn, retaining preceding text as context
        q,ans=turns[-1]; alltext=''.join(x+y for x,y in turns)
        if chinese(alltext)<30 or chinese(alltext)<english(alltext): continue
        if len(q)>a.max_question_chars or len(ans)>a.max_answer_chars: continue
        history=[]
        for h,m in turns[:-1]: history += [{'role':'user','content':h},{'role':'assistant','content':m}]
        rows.append({'audio':f'audio/{len(rows):06d}.wav','history':history,
          'instruction':q,'answer':ans,'task':'speech_qa','source':'moss-003-sft-data'})
        if len(rows)>=a.limit: break
    ndev=max(1,round(len(rows)*a.dev_ratio)); dev=rows[:ndev]; train=rows[ndev:]
    for fn,data in [('train.jsonl',train),('dev.jsonl',dev)]:
      (a.output/fn).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf8')
    stats['kept']=len(rows); stats['train']=len(train); stats['dev']=len(dev); stats['input']=str(a.input)
    (a.output/'metadata.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

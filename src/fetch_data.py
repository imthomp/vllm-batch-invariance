import json,sys
from datasets import load_dataset
out=sys.argv[1]
g=load_dataset("openai/gsm8k","main",split="test")
m=load_dataset("HuggingFaceH4/MATH-500",split="test")
recs=[{"id":f"gsm{i}","src":"gsm8k","q":r["question"],"a":r["answer"].split("####")[-1].strip()} for i,r in enumerate(g)]
recs+=[{"id":f"math{i}","src":"math500","q":r["problem"],"a":r["answer"],"level":r["level"]} for i,r in enumerate(m)]
json.dump(recs,open(out,"w")); print(len(recs))

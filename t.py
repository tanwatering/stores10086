import sys, time
from openai import OpenAI
base = "http://localhost:8000/v1"
model = sys.argv[1] if len(sys.argv) > 1 else "m"
c = OpenAI(base_url=base, api_key="EMPTY")
t0 = time.time(); first = None
for ch in c.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "用三句话介绍一下你自己"}],
        stream=True):
    p = ch.choices[0].delta.content or ""
    if p and first is None:
        first = time.time() - t0
    print(p, end="", flush=True)
print(f"\n\nTTFT={first:.2f}s  total={time.time()-t0:.2f}s")

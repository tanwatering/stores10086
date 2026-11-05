#!/usr/bin/env python3
"""并发压测：python bench.py <模型> <并发> <请求数>"""
import asyncio, sys, time, statistics
from openai import AsyncOpenAI

BASE = "http://localhost:8000/v1"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "m"
CONC  = int(sys.argv[2]) if len(sys.argv) > 2 else 4
N     = int(sys.argv[3]) if len(sys.argv) > 3 else 40
PROMPT = "请用三句话介绍一下人工智能的发展历史。"

async def one(client, results):
    t0 = time.time(); ttft = None; chars = 0
    try:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=128, stream=True, timeout=180)
        async for ch in stream:
            p = ch.choices[0].delta.content or ""
            if p and ttft is None:
                ttft = time.time() - t0
            chars += len(p)
        results.append((ttft, time.time() - t0, chars, True))
    except Exception as e:
        results.append((None, time.time() - t0, 0, False))

async def main():
    client = AsyncOpenAI(base_url=BASE, api_key="EMPTY")
    sem = asyncio.Semaphore(CONC)
    results = []
    async def wrapped():
        async with sem:
            await one(client, results)
    t0 = time.time()
    await asyncio.gather(*[wrapped() for _ in range(N)])
    wall = time.time() - t0

    ok = [r for r in results if r[3]]
    print(f"并发={CONC}  请求数={N}  成功={len(ok)}  墙钟={wall:.2f}s")
    if ok:
        ttfts = [r[0] for r in ok]
        chars = sum(r[2] for r in ok)
        print(f"TTFT   p50={statistics.median(ttfts):.3f}s   max={max(ttfts):.3f}s")
        print(f"字符吞吐  {chars / wall:.1f} 字/秒   (约 {chars / wall / 1.5:.0f} tok/s)")
        print(f"请求吞吐  {len(ok) / wall:.2f} req/s")
asyncio.run(main())

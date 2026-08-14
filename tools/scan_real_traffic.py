"""Scan real agent transcripts for orderprobe-eligible decision points.

Eligible means: several tools dispatched AT ONCE, so their results are ordered
by completion time rather than by causality. Sequential loops do not qualify --
permuting a causal sequence manufactures an incoherent context.

Structure only. Reads block types, tool names and payload sizes. Never reads,
records or transmits message content. Result on the authors' corpus:
3 eligible points in 1,855 sessions. See RESULTS-REALTRAFFIC.md.

Do assistant turns dispatch several tools AT ONCE? Those results are ordered
by completion time, which is exactly the arbitrary ordering orderprobe targets.
Structure only: counts tool names and sizes, not message content."""
import json, glob, collections

files = sorted(glob.glob('/home/craigm26/.claude/projects/**/*.jsonl', recursive=True))
par = collections.Counter()          # n parallel tool_use in one assistant msg
combos = collections.Counter()       # which tools get dispatched together
eligible_groups = 0

for f in files:
    try:
        for line in open(f, errors='ignore'):
            try: r = json.loads(line)
            except Exception: continue
            if r.get('type') != 'assistant': continue
            cont = (r.get('message') or {}).get('content')
            if not isinstance(cont, list): continue
            tus = [b for b in cont
                   if isinstance(b, dict) and b.get('type') == 'tool_use']
            if not tus: continue
            par[len(tus)] += 1
            if len(tus) >= 3:
                eligible_groups += 1
                combos[tuple(sorted(t.get('name','?') for t in tus))] += 1
    except Exception:
        continue

print("parallel tool_use blocks in one assistant turn -> how many turns")
for k in sorted(par):
    if k <= 12: print(f"  {k:2} at once : {par[k]:6}")
big = sum(v for k, v in par.items() if k > 12)
if big: print(f"  >12       : {big}")
print(f"\nturns dispatching >=2 in parallel: {sum(v for k,v in par.items() if k>=2)}")
print(f"ELIGIBLE (>=3 dispatched at once): {eligible_groups}")
print("\nmost common parallel combinations:")
for combo, n in combos.most_common(8):
    print(f"  {n:4}x  {', '.join(combo)}")

"""Paired bootstrap comparisons for run 2 (CF-order arm) that the notebook doesn't print.

    python analysis/paired_tests.py <dir> <out_dir>

Same <dir> layout as diagnostics.py. Reports Spec RAG vs the single drafter, Gemini on the users it answered, the
verifier against the drafter's own score, and the hit rate of films the LLM swaps in vs pushes out of CF's top 10.
"""
import sys, io, json, runpy, contextlib, numpy as np, pandas as pd
from pathlib import Path
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(Path(__file__).with_name("diagnostics.py")), run_name="x")
U, CTX, T, ndcg, bf, parse, RES = (g[k] for k in "U CTX T ndcg bf parse RES".split())
SPEC_CF = pd.read_pickle(RES / "spec_cf.pkl")
load = lambda f: {r["key"]: r for r in map(json.loads, open(RES / f)) if "error" not in r and r.get("text", "") is not None}
single_cf, gem_cf, gem = load("single_cf.jsonl"), load("gemini_cf.jsonl"), load("gemini.jsonl")
rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x); b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return f"{x.mean():+.4f} [{np.quantile(b, .025):+.4f}, {np.quantile(b, .975):+.4f}]"
cf = np.array([ndcg(u, CTX[u]["cands"][:10]) for u in U])
nd = lambda L: np.array([ndcg(u, bf(L[u], CTX[u]["cands"])) for u in U])
S = {"spec_argmax": {u: SPEC_CF[u]["argmax"] for u in U}, "spec_rrf": {u: SPEC_CF[u]["rrf"] for u in U},
     "single": {u: parse(single_cf[f"{u}:all"]["text"])[1] for u in U}}
N = {k: nd(v) for k, v in S.items()}
print("CF arm, paired:")
print("  Spec argmax - single drafter:", ci(N["spec_argmax"] - N["single"]))
print("  Spec argmax - CF           :", ci(N["spec_argmax"] - cf))
print("  Spec RRF    - CF           :", ci(N["spec_rrf"] - cf))
for name, gg in [("shuffled", gem), ("CF order", gem_cf)]:
    ok = [i for i, u in enumerate(U) if str(u) in gg]
    gn = np.array([ndcg(U[i], bf(parse(gg[str(U[i])]["text"])[1], CTX[U[i]]["cands"])) for i in ok])
    print(f"  Gemini [{name}] on its {len(ok)} answered users - CF on same users:", ci(gn - cf[ok]))
both_blocked = [u for u in U if str(u) not in gem and str(u) not in gem_cf]
print("  users blocked in both arms:", len(both_blocked))
for f in ["draft_level_CF_order.csv", "draft_level_shuffled.csv"]:
    d = pd.read_csv(RES / f)
    print(f"{f}: verifier pick - mean draft {ci(d.chosen - d.mean_draft)} | verifier pick - draft-score pick {ci(d.chosen - d.by_draft_only)}")
# what the LLM swaps in vs. what it pushes out of CF's top 10
for k in ["single", "spec_argmax"]:
    pin, pout, nsw = [], [], []
    for u in U:
        rel = set(T[u].indices); top = CTX[u]["cands"][:10]; mine = bf(S[k][u], CTX[u]["cands"])
        ins = [i for i in mine if i not in top]; outs = [i for i in top if i not in mine]
        pin += [i in rel for i in ins]; pout += [i in rel for i in outs]; nsw.append(len(ins))
    print(f"{k}: swaps per user {np.mean(nsw):.2f} | hit rate of films brought in {np.mean(pin):.4f} vs films pushed out {np.mean(pout):.4f}"
          f" | CF ranks 11-50 base rate {np.mean([i in set(T[u].indices) for u in U for i in CTX[u]['cands'][10:]]):.4f}")

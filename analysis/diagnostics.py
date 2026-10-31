"""Draft-level diagnostics for run 1 (shuffled candidate order).

    python analysis/diagnostics.py <dir> <out_dir>

<dir> must contain data/ (the notebook's §1 parquet files) and results/ (the notebook's outputs). The script rebuilds
the test split exactly as §2 does, then reports: which score picks the best draft, position bias, where picks come
from, and score distributions.
"""
import sys, json, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from scipy.stats import spearmanr
R = Path(sys.argv[1]); RES = R / "results"; DATA = R / "data"
ratings = pd.read_parquet(DATA / "ratings.parquet"); movies = pd.read_parquet(DATA / "movies.parquet")
ratings = ratings[ratings.movieId.isin(movies.movieId)]
ratings = ratings.sort_values(["userId", "timestamp", "movieId"], kind="stable").reset_index(drop=True)
ratings = ratings[ratings.groupby("userId").movieId.transform("size") >= 20]
fe = ratings.groupby("userId").cumcount(ascending=False).values
ratings["split"] = np.where(fe < 5, "test", np.where(fe < 10, "val", "train"))
tc = ratings.loc[ratings.split == "train", "movieId"].value_counts()
catalog = np.sort(tc[tc >= 50].index.values)
ratings = ratings[ratings.movieId.isin(catalog)].reset_index(drop=True)
users = np.unique(ratings.userId.values)
ratings["uid"] = np.searchsorted(users, ratings.userId.values); ratings["iid"] = np.searchsorted(catalog, ratings.movieId.values)
movies = movies.set_index("movieId").loc[catalog].reset_index()
title = movies.title_clean.tolist()
d = pd.Series(title).duplicated(keep=False).values
title = [f"{t} #{m}" if x else t for t, x, m in zip(title, d, movies.movieId)]
t2i = {t: i for i, t in enumerate(title)}
tg = ratings[(ratings.split == "test") & (ratings.rating >= 4)]
T = sp.csr_matrix((np.ones(len(tg)), (tg.uid, tg.iid)), shape=(len(users), len(catalog)))
fit = ratings[ratings.split != "test"]; pop = np.bincount(fit[fit.rating >= 4].iid, minlength=len(catalog))
disc = 1 / np.log2(np.arange(2, 12))
def ndcg(u, rec):
    rel = set(T[u].indices); h = np.array([i in rel for i in rec[:10]], float)
    return (h * disc[:len(h)]).sum() / disc[:min(len(rel), 10)].sum()
def parse(text):
    try: o = json.loads(text)
    except Exception: return None, []
    out = []
    for t in o.get("ranking", []):
        i = t2i.get(t)
        if i is not None and i not in out: out.append(i)
    return o.get("rationale", ""), out
def bf(r, c): return (list(r) + [x for x in c if x not in r])[:10]
CTX = pd.read_pickle(RES / "contexts_test.pkl"); SPEC = pd.read_pickle(RES / "spec_base.pkl")
U = sorted(CTX)
print("users", len(U), "| users with any test positive in 50 cands:", np.mean([len(set(T[u].indices) & set(CTX[u]["cands"])) > 0 for u in U]).round(3))

# --- draft-level
rows = []
for u in U:
    sc = SPEC[u]["scored"]
    if len(sc) < 2: continue
    nd = np.array([ndcg(u, bf(s["ranking"], CTX[u]["cands"])) for s in sc]); P = np.array([s["parts"] for s in sc])
    rows.append(dict(u=u, n=len(sc), mean=nd.mean(), oracle=nd.max(), worst=nd.min(),
        total=nd[P.sum(1).argmax()], draft=nd[P[:, 0].argmax()], sc=nd[P[:, 1].argmax()], sr=nd[P[:, 2].argmax()],
        sc_sr=nd[P[:, 1:].sum(1).argmax()], spread=nd.std(), ident=len({tuple(s["ranking"]) for s in sc})))
DL = pd.DataFrame(rows)
print("\n== draft selection: mean NDCG@10 of the draft picked by each score ==")
print(DL[["mean", "total", "draft", "sc", "sr", "sc_sr", "oracle", "worst"]].mean().round(4).to_string())
print("users where drafts differ in NDCG:", (DL.spread > 0).mean().round(3), "| distinct rankings per user:", DL.ident.mean().round(2))
# within-user correlation between each score and draft NDCG
cors = {k: [] for k in ["draft", "sc", "sr", "total"]}
for u in U:
    sc = SPEC[u]["scored"]
    nd = np.array([ndcg(u, bf(s["ranking"], CTX[u]["cands"])) for s in sc])
    if len(sc) < 3 or nd.std() == 0: continue
    P = np.array([s["parts"] for s in sc])
    for k, col in zip(cors, [P[:, 0], P[:, 1], P[:, 2], P.sum(1)]):
        if np.std(col) > 0: cors[k].append(spearmanr(col, nd).correlation)
print("within-user Spearman(score, NDCG):", {k: (round(np.nanmean(v), 3), len(v)) for k, v in cors.items()})
P = np.array([s["parts"] for u in U for s in SPEC[u]["scored"]])
print("\n== score distributions (log) ==")
print(pd.DataFrame(P, columns=["draft", "sc", "sr"]).describe().round(3).to_string())
print("share of SR at floor (p_yes<1e-7):", (P[:, 2] < -16).mean().round(3), "| between-user var share of SR:",
      round(pd.DataFrame([(u, s["parts"][2]) for u in U for s in SPEC[u]["scored"]]).groupby(0)[1].mean().var() / P[:, 2].var(), 3))

# --- what drives LLM picks: candidate-list position vs CF rank
def pick_stats(name, lists):
    pos, cfr, hits_by_rank = [], [], []
    for u in U:
        c = CTX[u]; order = {i: r for r, i in enumerate(c["order_full"])}; cr = {i: r for r, i in enumerate(c["cands"])}
        for i in lists[u][:10]:
            pos.append(order[i]); cfr.append(cr[i])
    pos, cfr = np.array(pos), np.array(cfr)
    print(f"{name:28s} mean shown-position {pos.mean():5.1f} (random 24.5) | share from first 10 shown {np.mean(pos<10):.2f} (random .20)"
          f" | mean CF rank {cfr.mean():5.1f} | share in CF top-10 {np.mean(cfr<10):.2f}")
single = {json.loads(l)["key"]: json.loads(l) for l in open(RES / "single_base.jsonl")}
gem = {r["key"]: r for r in map(json.loads, open(RES / "gemini.jsonl")) if r.get("text")}
L = {"drafter single": {u: parse(single[f"{u}:all"]["text"])[1][:10] for u in U},
     "Spec RAG argmax": {u: SPEC[u]["argmax"] for u in U},
     "Gemini": {u: parse(gem[str(u)]["text"])[1][:10] if str(u) in gem else [] for u in U}}
print("\n== where do the picks come from ==")
for k, v in L.items(): pick_stats(k, v)
print("gemini null responses:", sum(r["text"] is None for r in gem.values()), "/", len(gem),
      "| empty-parse:", sum(len(v) == 0 for v in L["Gemini"].values()))
# hit rate by CF rank and by shown position
hr_cf = np.zeros(50); hr_pos = np.zeros(50)
for u in U:
    rel = set(T[u].indices); c = CTX[u]
    for r, i in enumerate(c["cands"]): hr_cf[r] += i in rel
    for r, i in enumerate(c["order_full"]): hr_pos[r] += i in rel
print("test-positive rate by CF rank bucket (0-9,10-19,...):", (hr_cf.reshape(5, 10).sum(1) / (10 * len(U))).round(4))
print("test-positive rate by shown position bucket:          ", (hr_pos.reshape(5, 10).sum(1) / (10 * len(U))).round(4))
# precision of picks
for k, v in L.items():
    prec = np.mean([len(set(v[u][:10]) & set(T[u].indices)) / 10 for u in U])
    print(f"precision@10 {k:18s} {prec:.4f}")
print(f"precision@10 {'CF top-10':18s} {np.mean([len(set(CTX[u]['cands'][:10]) & set(T[u].indices)) / 10 for u in U]):.4f}")
# popularity of picks
for k, v in L.items(): print(f"median train-pop of picks {k:18s} {np.median([pop[i] for u in U for i in v[u]]):.0f}")
print(f"median train-pop of picks {'CF top-10':18s} {np.median([pop[i] for u in U for i in CTX[u]['cands'][:10]]):.0f}")
print(f"median train-pop of test positives   {np.median([pop[i] for u in U for i in T[u].indices]):.0f}")
# release year of picks vs. test positives
yr = movies.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float).values
for k, v in L.items(): print(f"median year {k:18s} {np.nanmedian([yr[i] for u in U for i in v[u]]):.0f}")
print(f"median year {'CF top-10':18s} {np.nanmedian([yr[i] for u in U for i in CTX[u]['cands'][:10]]):.0f}  | test positives {np.nanmedian([yr[i] for u in U for i in T[u].indices]):.0f}")
# finish reasons and parse validity
dr = [json.loads(l) for l in open(RES / "drafts_base.jsonl")]
print("\ndraft finish reasons:", pd.Series([x["finish"] for x in dr]).value_counts().to_dict(),
      "| single:", pd.Series([x["finish"] for x in single.values()]).value_counts().to_dict())
print("fallback users:", sum(SPEC[u]["fallback"] for u in U), "| scored drafts per user:", pd.Series([len(SPEC[u]["scored"]) for u in U]).value_counts().to_dict())
DL.to_csv(Path(sys.argv[2]) / "draft_level.csv", index=False)

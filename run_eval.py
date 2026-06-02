"""
run_eval.py — evaluate all three pipelines from the paper against the
generated synthetic applications, with strict data-leakage prevention.

Inputs:
    A batch directory produced by run_batch.py (default: the latest under
    BOP_Prompt_Workspace/outputs/).

Outputs (under <batch>/eval/):
    eval_results.csv     — one row per app, all three pipeline decisions
    eval_summary.json    — accuracy, confusion matrices, cosine-similarity
    eval_log.jsonl       — per-app append-only log (also used for --resume)

Usage:
    python run_eval.py
    python run_eval.py --limit 25
    python run_eval.py --batch outputs/apps_<ts>
    python run_eval.py --workers 5 --model gpt-4o-mini
    python run_eval.py --pipelines single_llm naive_rag agentic_rag
    python run_eval.py --resume
"""
from __future__ import annotations
import argparse, json, os, sys, time, csv, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_pipelines as ep


DATA_DIR  = HERE / "data"
GUIDE_PDF = DATA_DIR / "BOP_Guidebook.pdf"

PIPELINE_NAMES = ("single_llm", "naive_rag", "agentic_rag")


def _latest_batch() -> Path:
    batches = sorted([p for p in (HERE / "outputs").glob("apps_*") if p.is_dir()])
    assert batches, "No batches under outputs/. Run run_batch.py first."
    return batches[-1]


def _embedding_cache_get(text: str, llm_emb) -> list:
    """Compute an embedding for cosine similarity of rationales (cached locally)."""
    return llm_emb.embed_query(text)


def _cosine(a: list, b: list) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(aa); nb = np.linalg.norm(bb)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(aa, bb) / (na * nb))


def _eval_one(fp: Path, pipelines: ep.PipelineBundle,
              chosen_pipelines: list[str]) -> dict:
    raw = json.load(open(fp))
    # CRITICAL: sanitize before letting any pipeline see the record
    sanitized = ep.sanitize_app_for_eval(raw)
    flat_app, flat_tp = ep.flat_application_view(sanitized)

    gt = raw.get("ground_truth") or {}
    row: dict = {
        "file":                  str(fp.name),
        "scenario":              raw.get("scenario", ""),
        "business_type":         raw.get("business_type", ""),
        "ground_truth_decision": gt.get("decision", ""),
        "ground_truth_reason":   gt.get("reason", ""),
    }
    name_to_pipeline = {
        "single_llm":   pipelines.single,
        "naive_rag":    pipelines.naive,
        "agentic_rag":  pipelines.agentic,
    }
    for pname in chosen_pipelines:
        pl = name_to_pipeline[pname]
        t0 = time.time()
        try:
            out = pl.run(flat_app, flat_tp)
            dt = time.time() - t0
            row[f"{pname}_decision"]    = out.get("decision", "")
            row[f"{pname}_reason"]      = (out.get("reason") or "")[:1000]
            row[f"{pname}_cap"]         = out.get("logistic_cap")
            row[f"{pname}_latency_sec"] = round(dt, 2)
            row[f"{pname}_error"]       = ""
        except Exception as e:
            dt = time.time() - t0
            row[f"{pname}_decision"]    = "ERROR"
            row[f"{pname}_reason"]      = f"{type(e).__name__}: {e}"[:1000]
            row[f"{pname}_cap"]         = None
            row[f"{pname}_latency_sec"] = round(dt, 2)
            row[f"{pname}_error"]       = f"{type(e).__name__}: {e}"
    return row


def _summarize(rows: list[dict], chosen: list[str],
               llm_emb=None, sim_sample: int = 100) -> dict:
    labels = ["ACCEPT", "REJECT", "REFER_TO_HUMAN_REVIEW"]
    summary: dict = {"n_apps": len(rows), "pipelines": {}}

    # Per-scenario breakdown for ground truth
    by_scenario_gt = defaultdict(Counter)
    for r in rows:
        by_scenario_gt[r["scenario"]][r["ground_truth_decision"]] += 1
    summary["ground_truth_by_scenario"] = {
        k: dict(v) for k, v in by_scenario_gt.items()
    }

    for pname in chosen:
        decs = [r[f"{pname}_decision"] for r in rows]
        gts  = [r["ground_truth_decision"] for r in rows]
        valid_pairs = [(g, d) for g, d in zip(gts, decs)
                       if g in labels and d in labels]
        n_valid = len(valid_pairs)
        correct = sum(1 for g, d in valid_pairs if g == d)
        accuracy = correct / n_valid if n_valid else 0.0

        # confusion matrix: rows=gt, cols=predicted
        cm = {g: {d: 0 for d in labels} for g in labels}
        for g, d in valid_pairs:
            cm[g][d] += 1

        # per-scenario accuracy
        per_sc: dict = {}
        for sc in {r["scenario"] for r in rows}:
            srows = [r for r in rows if r["scenario"] == sc]
            n_sc = sum(1 for r in srows
                       if r["ground_truth_decision"] in labels
                       and r[f"{pname}_decision"] in labels)
            ok_sc = sum(1 for r in srows
                        if r["ground_truth_decision"] in labels
                        and r["ground_truth_decision"] == r[f"{pname}_decision"])
            per_sc[sc] = {
                "n": n_sc,
                "accuracy": (ok_sc / n_sc) if n_sc else 0.0,
                "predictions": dict(Counter(r[f"{pname}_decision"] for r in srows)),
            }

        # error rate
        n_err = sum(1 for r in rows if r[f"{pname}_decision"] == "ERROR")

        # latency
        lats = [r[f"{pname}_latency_sec"] for r in rows if r[f"{pname}_latency_sec"] is not None]
        latency = {"mean": float(np.mean(lats)) if lats else 0.0,
                   "median": float(np.median(lats)) if lats else 0.0,
                   "p95": float(np.percentile(lats, 95)) if lats else 0.0}

        pl_summary = {
            "n_apps": len(rows),
            "n_valid": n_valid,
            "n_errors": n_err,
            "accuracy_overall": round(accuracy, 4),
            "confusion_matrix_rows_gt": cm,
            "per_scenario": per_sc,
            "latency_sec": {k: round(v, 2) for k, v in latency.items()},
        }
        summary["pipelines"][pname] = pl_summary

    # Cosine similarity of reasoning vs ground-truth reason (subsample)
    if llm_emb is not None and sim_sample > 0:
        sub = rows[:sim_sample]
        gt_embs = {i: _embedding_cache_get(r["ground_truth_reason"], llm_emb)
                   for i, r in enumerate(sub) if r["ground_truth_reason"]}
        for pname in chosen:
            sims = []
            for i, r in enumerate(sub):
                if i not in gt_embs:
                    continue
                txt = r[f"{pname}_reason"]
                if not txt:
                    continue
                try:
                    sims.append(_cosine(gt_embs[i], _embedding_cache_get(txt, llm_emb)))
                except Exception:
                    pass
            summary["pipelines"][pname]["reason_cosine_similarity"] = {
                "n": len(sims),
                "mean": round(float(np.mean(sims)), 4) if sims else None,
                "median": round(float(np.median(sims)), 4) if sims else None,
            }

    return summary


def _resume_seen(log_path: Path) -> set[tuple[str, str]]:
    """Return the set of (scenario, filename) pairs already evaluated.

    The key must include scenario: filenames collide across scenario dirs
    (e.g. Accounting_and_Financial_Services.json exists in all 5), so keying
    on filename alone would wrongly skip every app once one scenario was done.
    """
    seen = set()
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen.add((r.get("scenario", ""), r["file"]))
                except Exception:
                    pass
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None,
                    help="Path to outputs/apps_<ts>/ (default: latest)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--model", default=os.environ.get("BOP_MODEL", "gpt-4o-mini"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Only evaluate the first N apps (stratified across scenarios).")
    ap.add_argument("--pipelines", nargs="*", default=list(PIPELINE_NAMES))
    ap.add_argument("--resume", action="store_true",
                    help="Skip apps already in eval_log.jsonl.")
    ap.add_argument("--sim-sample", type=int, default=100,
                    help="N apps to compute reason-cosine-sim on (0 to skip).")
    args = ap.parse_args()

    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
    for p in args.pipelines:
        if p not in PIPELINE_NAMES:
            ap.error(f"Unknown pipeline {p!r}; choices: {PIPELINE_NAMES}")

    batch = Path(args.batch).resolve() if args.batch else _latest_batch()
    print(f"[init] model={args.model}  workers={args.workers}  batch={batch.name}")
    print(f"[init] pipelines={args.pipelines}")

    # Discover apps (stratified if --limit set)
    by_sc: dict[str, list[Path]] = defaultdict(list)
    for sc_dir in sorted(batch.iterdir()):
        if not sc_dir.is_dir() or sc_dir.name in ("eval",):
            continue
        for fp in sorted(sc_dir.glob("*.json")):
            by_sc[sc_dir.name].append(fp)
    print(f"[init] discovered: {sum(len(v) for v in by_sc.values())} apps across {len(by_sc)} scenarios")

    files: list[Path] = []
    if args.limit:
        per_sc = max(1, args.limit // max(1, len(by_sc)))
        for sc, fs in by_sc.items():
            files.extend(fs[:per_sc])
    else:
        for fs in by_sc.values():
            files.extend(fs)
    print(f"[init] evaluating {len(files)} apps")

    out_dir = batch / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path     = out_dir / "eval_log.jsonl"
    csv_path     = out_dir / "eval_results.csv"
    summary_path = out_dir / "eval_summary.json"

    if args.resume:
        seen = _resume_seen(log_path)
        # Each file's scenario is its parent directory name.
        files = [f for f in files if (f.parent.name, f.name) not in seen]
        print(f"[resume] skipping {len(seen)} done; {len(files)} remaining")

    # Build shared resources
    print(f"[init] loading guidelines + FAISS …")
    guidelines = ep.GuidelineContext.from_files(str(DATA_DIR))
    pages = PyPDFLoader(str(GUIDE_PDF)).load_and_split()
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(pages)
    emb = OpenAIEmbeddings(request_timeout=120, max_retries=3)
    vs = FAISS.from_documents(docs, emb)
    retriever = vs.as_retriever(search_kwargs={"k": 6})
    # request_timeout caps a single API call; max_retries lets transient
    # stalls/5xx retry instead of hanging a worker thread forever (which
    # previously froze the whole ThreadPoolExecutor at ~86% complete).
    llm = ChatOpenAI(model=args.model, temperature=0,
                     request_timeout=120, max_retries=3)
    pipelines = ep.build_pipelines(llm=llm, retriever=retriever,
                                    guidelines=guidelines, threshold=1.75)
    print(f"[init]   {len(docs)} chunks in FAISS; pipelines ready\n")

    # Run evaluation
    t_start = time.time()
    log_f = open(log_path, "a", encoding="utf-8")
    rows_done: list[dict] = []
    counts_by_pipe = defaultdict(Counter)
    n_done = n_err = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_eval_one, fp, pipelines, args.pipelines): fp for fp in files}
        for fut in as_completed(futs):
            fp = futs[fut]
            try:
                row = fut.result()
                log_f.write(json.dumps(row) + "\n"); log_f.flush()
                rows_done.append(row)
                for p in args.pipelines:
                    counts_by_pipe[p][row[f"{p}_decision"]] += 1
                n_done += 1
                if n_done % 25 == 0 or n_done == len(files):
                    dt = time.time() - t_start
                    rate = n_done / dt if dt else 0
                    eta = (len(files) - n_done) / max(rate, 0.01)
                    pacc = {}
                    for p in args.pipelines:
                        valid = [r for r in rows_done if r[f"{p}_decision"] != "ERROR"]
                        if valid:
                            ok = sum(1 for r in valid if r[f"{p}_decision"] == r["ground_truth_decision"])
                            pacc[p] = f"{ok}/{len(valid)}={100*ok/len(valid):.0f}%"
                    print(f"  [{n_done}/{len(files)}]  acc: {pacc}  "
                          f"eta={int(eta//60)}m{int(eta%60):02d}s")
            except Exception as e:
                n_err += 1
                print(f"  [ERROR] {fp.name}: {type(e).__name__}: {e}")
                traceback.print_exc()
    log_f.close()

    # Rebuild rows from log to include any prior resume rows
    all_rows = []
    with open(log_path) as f:
        for line in f:
            try: all_rows.append(json.loads(line))
            except Exception: pass
    print(f"\n[done] this run: {n_done} apps, {n_err} task errors, "
          f"elapsed {int((time.time()-t_start)//60)}m{int((time.time()-t_start)%60):02d}s")
    print(f"[done] total in log: {len(all_rows)} apps")

    # Write CSV
    if all_rows:
        # Stable column order
        cols = ["file", "scenario", "business_type",
                "ground_truth_decision", "ground_truth_reason"]
        for p in args.pipelines:
            cols += [f"{p}_decision", f"{p}_reason", f"{p}_cap",
                     f"{p}_latency_sec", f"{p}_error"]
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in all_rows: w.writerow(r)
        print(f"[done] wrote {csv_path}")

    # Summary
    print(f"[summary] computing accuracy / confusion / cosine …")
    summary = _summarize(all_rows, args.pipelines,
                         llm_emb=emb if args.sim_sample > 0 else None,
                         sim_sample=args.sim_sample)
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    print(f"[done] wrote {summary_path}")

    # Pretty print to console
    print("\n=== Pipeline accuracy ===")
    for p in args.pipelines:
        s = summary["pipelines"][p]
        print(f"  {p:<12}  acc={s['accuracy_overall']*100:5.1f}%  "
              f"(n_valid={s['n_valid']}, errors={s['n_errors']})  "
              f"latency mean={s['latency_sec']['mean']}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())

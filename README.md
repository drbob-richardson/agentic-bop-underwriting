# Agentic AI for Straight-Through Underwriting — Data & Code Release

Companion artifacts for the paper *Agentic AI and Retrieval-Augmented Models
in Straight-Through Underwriting*. This repository contains the 635-application
synthetic Business Owner Policy (BOP) dataset, the implementation of the three
underwriting pipelines compared in the paper (single-LLM, naïve RAG, and the
multi-agent Agentic RAG), precomputed evaluation results, and a viewer
notebook for reproducing the tables and figures.

## What's here

```
.
├── README.md                          # this file
├── requirements.txt                   # Python dependencies
├── BOP_Synthetic_Dataset.csv          # 635 apps as a flat CSV (paper-friendly format)
├── data_dictionary.csv                # column-by-column description for the CSV
├── eval_pipelines.py                  # the three pipelines + sanitization
├── run_eval.py                        # re-evaluate the pipelines on the published apps
├── Agentic_RAG_Underwriting.ipynb     # viewer notebook (loads precomputed results)
├── data/
│   ├── BOP_Guidebook.pdf              # 143-page synthetic underwriting guidebook
│   ├── global_general_guidelines.json # the 16 global underwriting questions
│   ├── stage2_expanded_guidelines.json# expanded per-business-type guidelines
│   └── guidebook.json                 # full structured guidebook
└── outputs/
    └── apps_published_v1/
        ├── accept/                    # 127 JSON apps (one per business type)
        ├── reject_guideline/          # 127 JSON apps
        ├── reject_logit/              # 127 JSON apps
        ├── incomplete_recoverable/    # 127 JSON apps
        ├── incomplete_irrecoverable/  # 127 JSON apps
        ├── all_apps.jsonl             # generation log (one row per app)
        ├── ground_truth.jsonl         # verified labels + visibility check
        └── eval/
            ├── eval_results.csv       # per-app, per-pipeline decisions + rationales
            └── eval_summary.json      # accuracy, confusion matrices, latency
```

## Quickstart

Three things you might want to do:

### 1. Inspect the data
The flat dataset is in `BOP_Synthetic_Dataset.csv`. Open it in any tool;
column descriptions are in `data_dictionary.csv`. The first nine columns are
identifiers + ground-truth labels; the next 74 are application form fields;
the last 11 are simulated third-party data (claims history + Google reviews).

If you'd rather work with the original per-application JSON records (one file
per application, with the field/third-party/ground-truth blocks separated),
they're under `outputs/apps_published_v1/<scenario>/`.

### 2. Reproduce the paper's pipeline-comparison results
The precomputed per-app, per-pipeline results are in
`outputs/apps_published_v1/eval/eval_results.csv`. Open
`Agentic_RAG_Underwriting.ipynb` and run all cells. The notebook loads the
precomputed CSV and produces the accuracy tables, confusion matrices, and
disagreement analyses reported in the paper. No API calls are required for
this path.

### 3. Re-evaluate the pipelines from scratch (requires OpenAI API key)
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python run_eval.py --batch outputs/apps_published_v1 --workers 5 --model gpt-4o-mini
# or --model gpt-5.2 for the stronger-model results
```
Output lands in `outputs/apps_published_v1/eval/`. Approximate cost is
~US$5–10 for `gpt-4o-mini` and ~US$5–7 for `gpt-5.2` on the full 635 × 3
evaluations.

## Dataset summary

| | |
|---|---|
| Applications | 635 |
| Business types | 127 |
| Scenarios | 5 (Compliant Accept, Single-Issue Violation, Logistic Failure, Missing-Info Recoverable, Missing-Info Irrecoverable) |
| Decision labels | ACCEPT (127), REJECT (345), REFER\_TO\_HUMAN\_REVIEW (163) |
| Application form fields | 74 |
| Third-party data fields | 11 |
| Generation model | OpenAI `gpt-4o-mini` (synthetic, temperature 1) |
| Label-validation model | OpenAI `gpt-4o-mini` (LLM-as-judge over application + third-party text) |

See the paper, Section 4, for the generation procedure and label scheme,
and `data_dictionary.csv` for column-level descriptions.

## Pipelines

`eval_pipelines.py` implements three pipelines that all return a decision in
`{ACCEPT, REJECT, REFER_TO_HUMAN_REVIEW}`:

1. **`SingleLLMPipeline`** — one chat completion per application with the
   global and business-specific guidelines inlined.
2. **`NaiveRAGPipeline`** — one chat completion per application with FAISS
   retrieval (top-`k=6`) over the underwriting guidebook PDF.
3. **`AgenticRAGPipeline`** — a LangGraph state machine with an appetite
   check (Agent 1), an optional reflection step (Agent 3, up to two retries),
   a logistic risk-screen gate, a completeness check (Agent 2), a third-party
   evaluation step, and a mandatory decision-reflection critic. The critic
   may only confirm an ACCEPT or escalate it to REFER, and only if it can
   name a specific decisive missing exclusion.

A label-leakage helper (`sanitize_app_for_eval`) strips the ground-truth,
chosen-reason, scenario, decision, and check fields from each record before
any pipeline observes it.

## Requirements

Python 3.10+ and the packages listed in `requirements.txt` (pandas, numpy,
scikit-learn, langchain + langchain-openai + langchain-community + langgraph,
faiss-cpu, pypdf, jupyter). All language-model calls go to the OpenAI API;
no local model inference is required.

## Citation

If you use this dataset or code in academic work, please cite the paper:

```bibtex
@article{<key>,
  title   = {Agentic AI and Retrieval-Augmented Models in Straight-Through Underwriting},
  author  = {<authors>},
  journal = {<venue>},
  year    = {<year>}
}
```

## License

Code and data are released under [CHOOSE: MIT for code / CC BY 4.0 for data].
The synthetic underwriting guidebook in `data/BOP_Guidebook.pdf` is a
machine-generated artifact and is also released under the same terms.

## Contact

Questions, bug reports, and pull requests are welcome. Please open an issue
on this repository.

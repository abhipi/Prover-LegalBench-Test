# LegalBench + DeepSeek-Prover-V2 (OpenRouter) starter harness

This is a minimal evaluation harness for a recommended subset of LegalBench tasks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="..."
# Optional attribution headers:
export OPENROUTER_HTTP_REFERER="https://yourapp.com"
export OPENROUTER_X_TITLE="my-legalbench-prover-run"
```

## Run

```bash
python run_eval.py --n_shots 3 --max_test 100
```

Results are written to `runs/<task>.jsonl` and `runs/summary.json`.

## Add more tasks

1. Add the task name to `TASKS` in `task_configs.py`.
2. Specify the canonical labels and normalization rules (regexes).
3. Run: `python run_eval.py --tasks your_task_name`.

Let me finish auditing the files I haven't fully checked yet, so this TODO is accurate and not guesswork.Good — that surfaced two more compounding bugs that matter a lot: `MetricsMonitor()` crashes on instantiation, and it's imported at module level in `main.py`. That means **the FastAPI app currently can't even start**, independent of every other bug already found. I'll flag where I've fully verified vs. where you still need to audit using the same pattern, so nothing here is guesswork dressed up as fact.

Work top to bottom — each phase only unblocks once the one above it is done, so you'll always know exactly what to touch next.

---
COMMAND BEFORE RUNNING ANY FILE TO LAUNCH VIRTUAL ENVIRONMENT 
PS C:\Users\nikhi\Downloads\Large-Scale-Taxi-Demand-Forecasting-System> (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& c:\Users\nikhi\Downloads\Large-Scale-Taxi-Demand-Forecasting-System\venv\Scripts\Activate.ps1)
## Phase 0 — Setup (do once, ~10 min)
- [x] `git checkout -b fix/core-pipeline` — keep this off `main` until it actually runs
- [ ] `pip install -r requirements.txt` locally, confirm no install errors
- [x] check if it is possible to add zone_id to the data using longituse and latitude,check other columns,compare with program and make neccessary changes to the data or program which ever is most efficient or faster
- [x] Get a small sample of NYC taxi parquet data locally (a few days, not the full multi-GB set) so you can iterate fast without waiting on Spark jobs
    - [x] find how to properly split the data and choose so that it is not biased or gives wrong results
- [x] figure out if dataset needs to be cleaned
- [x] add target column demand into the dataset

---

## Phase 1 — `config/settings.py` (THE blocker — nothing else works until this is fixed)
- [x] Delete the four self-referential `from config.settings import (...)` blocks at the top (lines 1–31) — this file should **define** these names, not import them from itself
- [x] Fix the missing comma bug that's a hard syntax error: `PROCESSED_PATH, FEATURES_PATH,MONITORED_FEATURES` directly followed by `LAG_HOURS` with no comma (around line 8)
- [X] Replace the bare dangling names (lines 32–37: `METRICLOG`, `PERFORMANCELOG`, `MODEL_PATH,QUANTILE_LOW_MODEL_PATH...`, `HOTSPOTS,RECENT_HISTORY`, `SERVING_HALTED`) with **actual variable assignments**, e.g. `MODEL_PATH = MODEL_DIR / "lgbm_demand.pkl"`
- [x] Cross-reference every name imported anywhere else in the repo (`COLUMN_MAP`, `REQUIRED_COLUMNS`, `VALID_ZONE_MIN/MAX`, `MIN/MAX_FARE`, `MIN/MAX_DISTANCE`, `MIN/MAX_PASSENGER`, `DATA_START/END_DATE`, `TIME_GRANULARITY`, `FILL_MISSING_ZEROS`, `SPARK_*`, `LAG_HOURS`, `ROLLING_WINDOWS`, `PUBLIC_HOLIDAYS_2022`, `TRAIN/VAL_END_DATE`, `TEST_START_DATE`, `TARGET_COLUMN`, `FEATURE_COLUMNS`, `MONITORED_FEATURES`, `RECENT_HISTORY`, `HOTSPOTS`, `LOG`, `METRICLOG`, `PERFORMANCELOG`, `SERVING_HALTED`) and make sure each is actually defined with a real value
- [x] **Verify:** `python -c "import config.settings"` runs with zero errors before moving on

---

## Phase 2 — `etl/spark_pipeline.py`
- [ ] Download TLC taxi zone shapefile → data/raw/taxi_zones/
- [x]  Fix COLUMN_MAP to map actual raw column names
- [x]  Fix all the bugs from the original Phase 2 list (alias F vs f, far_amount typo, fare_amount <= MAX_DISTANCE wrong column, method name mismatch)
- [x]  Test on your 21-row sample — confirm output parquet has columns zone_id, hour_timestamp, demand with sensible values before running on the real data
- [x] do def clean column names verified and logic checked
- [x] Line 9: `stringType` → `StringType` (capitalization)
- [x] Line 7 imports `functions as f` (lowercase) but lines 133, 158, 163–167, 171–172 use `F` (uppercase, never imported) — pick one alias and use it everywhere
- [x] Line 105: `df.withColumn("far_amount", ...)` → should overwrite `"fare_amount"`, not create a new column `"far_amount"`
- [x] Line 123: filter compares `fare_amount <= MAX_DISTANCE` — should be `trip_distance <= MAX_DISTANCE`
- [x] Line 258: `self.fill_missing_zeros(df)` called, but method is defined as `filling_missing_zeros` (line 179) — make the call match the definition (rename one or the other)
- []**For the full multi-GB dataset**, doing `toPandas()` defeats the point of PySpark. Once you have the basic flow working on a sample, the production-grade fix is Apache Sedona (formerly GeoSpark) — a Spark-native spatial library. Add a comment in the code saying this explicitly:

```python
# NOTE: toPandas() here is acceptable for samples / dev runs.
# For production scale (multi-GB), replace with Apache Sedona:
# https://sedona.apache.org/latest-snapshot/api/python/reference/
- [ ] **Verify:** run the ETL on your small local sample, confirm a parquet file actually gets written and row counts printed make sense

---

## Phase 3 — `features/engineer.py`
- [x] Line 115: self.zone_stats-(...)` → should be `self.zone_stats=(...)` (assignment, not subtraction)
- [x] Line 119 creates column `zone_std_demands` (with trailing "s"), but line 124 reads `zone_std_demand` (no "s") — make the name match in both places
- [x] Line 129: `pd.read_csv()` called with **no path argument** — add the actual zone-lookup file path, e.g. `pd.read_csv("data/taxi_zone_lookup.csv")`
- [x] Same line: this is wrapped in `except FileNotFoundError`, but a missing-argument call raises `TypeError`, not `FileNotFoundError` — once the path is added this becomes moot, but double check the except clause still makes sense
- [x] Lines 99–104: the `for window in [6,24]:` loop's body (computing `rolled_std`) is indented inside the loop, but the line that assigns it to `df[f"roll_std_{window}h"]` (line 104) is indented **outside** the loop — meaning `roll_std_6h` is silently never created, only `roll_std_24h` (using the loop's last value). Fix the indentation so the assignment is inside the loop
- [x] Line 204: calls `self.finalize(df)` but the method is defined as `finalise` (line 160, British spelling) — make these match
- [x] **Verify:** run feature engineering on the ETL output from Phase 2, confirm the printed `missing_features` warning list is empty

---

## Phase 4 — `models/train.py` + `models/evaluate.py`
**train.py:**
- [x] Remove duplicate `import numpy as np` (lines 3 and 13)
- [x] Line 22: `from evaluate import Evaluate` → `from models.evaluate import Evaluate` (or run things as a proper package — pick one convention and apply it everywhere in the repo)
- [x] Lines 193 and 227: inside `train_quantile_low_model`/`train_quantile_high_model`, `self.model.fit(...)` is called — should be `self.quantile_low_model.fit(...)` and `self.quantile_high_model.fit(...)` respectively
- [x] Lines 202, 236, 285: all reference `self.version`, but it's only ever set inside `save_best_model(self, version)` — which runs *after* these methods in the pipeline. Either pass a version string through the constructor, or reorder the pipeline so `save_best_model` runs first
- [x] Line 358: `self.save_best_model()` called with **no arguments**, but the method requires `version` — pass an actual version string, e.g. `self.save_best_model(version=datetime.now().strftime("%Y%m%d_%H%M"))`
- [x] Line 361: `self.save_artifacts()` is called but **this method doesn't exist anywhere in the class** — either write it, or remove the call if it's redundant with `save_best_model`
- [x] Line 313 area: the baseline-metric calls run on `self.train`/`self.val`/`self.test` (raw Spark DataFrames), but `Evaluate.baseline_naive_seasonal` indexes them like pandas (`df['lag_168h']`) — switch these calls to use `self.train_pd`/`self.val_pd`/`self.test_pd` instead
- [x] Optional cleanup: `split_data_pandas` (line 59) is never called anywhere — either delete it or actually use it; dead code is confusing in an interview walkthrough

**evaluate.py:**
- [x] Line 32: the parameter `print: bool=False` shadows Python's built-in `print` function — rename it to something like `verbose: bool=False`
- [x] Line 83: `axs[i].ylabel(...)` → `axs[i].set_ylabel(...)` (matplotlib Axes objects don't have `.ylabel()`)
- [x] Line 134: `analyze_residuals(self, df, split_name, model)` requires `model`, but it's called from `train.py` line 351 as `evaluation.analyze_residuals(self.test, 'Test')` — missing the `model` argument; add it
- [x] **Verify:** run training end-to-end on your sample, confirm a model file actually lands in `models/artifacts/`

---

## Phase 5 — Monitoring layer (must work before the API will even start)
**`monitoring/prediction_logger.py`:**
- [ ] Line 15: `pd.read_parquet(path)` isn't wrapped in a try/except — wrap it so a first-ever run (no log file yet) doesn't crash
- [ ] Line 18: `if df_existing:` — truthiness check on a DataFrame raises an error; change to `if not df_existing.empty:`
- [ ] Lines 35, 39: both write to `self.log_path`, but `PredictionLogger.__init__` never sets this attribute — use the `path` parameter that's already passed into the method instead
- [ ] `log_predictions`/`log_predictions_actual` build a plain dict and pass it to `save_logs_parquet`, which does `pd.DataFrame(logs)` — a dict of scalars needs to be wrapped in a list first: `pd.DataFrame([logs])`

**`monitoring/metrics_monitor.py`:**
- [ ] Line 3: `from monitoring.prediction_logger import prediction_logger` — the actual class is `PredictionLogger` (capitalized) — fix the import name
- [ ] Line 19 (`__init__`): `self.metrics` with no assignment — this crashes the moment `MetricsMonitor()` is instantiated. Change to `self.metrics = {}`
- [ ] `save_metrics_parquet`: `self.metric_path[filename]` treats a `Path` object like a dictionary — fix to construct an actual path, e.g. `self.metric_path / filename`
- [ ] Same method also writes to `self.log_path`, which isn't set anywhere in this class — use `self.metric_path` instead
- [ ] **Verify:** `python -c "from monitoring.metrics_monitor import MetricsMonitor; MetricsMonitor()"` runs without crashing — this single line failing is currently why your whole API can't boot

---

## Phase 6 — `api/main.py` + `api/predictor.py`
**main.py:**
- [ ] Line 1: `httpexception` → `HTTPException` (and actually use it for error responses instead of returning plain strings)
- [ ] Line 27: `with open() as f:` — **no file path passed at all**; this is your main `/forecast` endpoint and it currently crashes on every call. Pass the actual serving-halted flag file path, e.g. `open(SERVING_HALTED) as f:`
- [ ] Line 26: `zone_id` has no type hint, so FastAPI passes it as a string; it's then compared against `valid_zone`, a numpy int array. Add `zone_id: int` to the function signature
- [ ] Line 29: `if flag==True:` compares a whole loaded dict to `True` — change to check the actual key, e.g. `if flag.get("serving_halted"):`
- [ ] Line 2: `from predictor import TaxiForecaster` → `from api.predictor import TaxiForecaster` (same package-path issue as `train.py`)

**predictor.py:**
- [ ] `curr_df=pd.DataFrame(curr)` — `curr` is a dict of scalars; needs `pd.DataFrame([curr])`
- [ ] `build_feature_row` returns a tuple `(row_df, row)`, but `build_feature_rows_all_zones` calls it as `row_df = self.build_feature_row(...)` (single variable) — unpack it properly: `row_df, _ = self.build_feature_row(...)`
- [ ] `self.recent_history_df=ps.read_csv(recent_history_path)` reads CSV, but predictions are saved as **parquet** elsewhere (`save_logs_parquet`) — pick one format and make read/write consistent
- [ ] Important, not just a bug: `build_feature_row` only computes `roll_mean_6h` — your trained model expects more rolling features (`roll_mean_3h`, `roll_mean_24h`, `roll_std_6h`, etc. per `feature_engineer.py`). Right now those get silently filled with `0` at inference time via `reindex(..., fill_value=0)`. That's a real train/serve skew, not just a crash — predictions would be systematically wrong even once the code runs. Compute all the rolling features that training actually used.
- [ ] Both `np.clip(pred, 0)` calls — double-check this works with your installed numpy version; some versions require both `a_min` and `a_max`
- [ ] **Verify:** start the API locally (`uvicorn api.main:app --reload`), hit `/health`, then `/forecast/{a_valid_zone_id}/24` and confirm you get real numbers back, not a 500

---

## Phase 7 — Audit the files I haven't fully line-by-line verified yet
I read `mlops_pipeline.py` (confirmed: it's corrupted with pasted AI-chat prose and won't parse — needs a full rewrite, not a patch) and `dashboard/app.py` (confirmed empty). I have **not** yet done a full line-by-line pass on these — audit them yourself using the same pattern above (check every import resolves, every method called actually exists with matching name/signature, every `self.X` is set before it's read):
- [ ] `monitoring/alert_manager.py`
- [ ] `monitoring/drift_detector.py`
- [ ] `models/drift_detector.md` (this is markdown sitting in `models/` — probably notes, but check it's not meant to be a `.py` file that got mis-saved)
- [ ] `retraining/model_registry_manager.py`
- [ ] `retraining/retrain_pipeline.py`
- [ ] Fully rewrite `monitoring/mlops_pipeline.py` — the current content isn't recoverable, it's prose mixed with broken syntax
- [ ] compare the project with the resume points,add todo for incomplete points and finish and tidy the project

---

## Phase 8 — Fill the real gaps (after everything above runs)
- [ ] Pin `requirements.txt` to actual versions (`pip freeze > requirements.txt` once everything works)
- [ ] Decide honestly: keep the "Cloud Storage" resume claim only if you actually wire up S3/GCS read-write — otherwise drop it from the resume bullet rather than risk an interview catching the gap
- [ ] Add a `tests/` directory — even 5–10 tests on the ETL cleaning logic and feature engineering is enough to legitimately claim "tested"
- [ ] Add `.github/workflows/` for at least lint + test on push
- [ ] Add a `Dockerfile` for the FastAPI service
- [ ] Write `dashboard/app.py` for real, or remove the empty folder if it's not going to happen — an empty file is worse than no file

---

## Phase 9 — Documentation (only once Phases 1–6 are verified working)
This is where the README, full code walkthrough, and STAR-method interview doc come in — I'll generate all three once you confirm the core pipeline runs end to end. Writing them before that would mean documenting bugs as if they were features.

---

**Suggested pacing:** Phases 1–2 today (these unblock everything and are mechanical fixes), Phase 3–4 next session, Phase 5–6 after that — that's the point where you have a genuinely working system end to end and the rest is polish. Want to start Phase 1 right now together?
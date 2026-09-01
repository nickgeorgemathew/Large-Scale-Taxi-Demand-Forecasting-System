Noticed you skipped the three questions — but a build checklist is a legitimate ask on its own (that's Project Architect mode, not "write my code for me"), so I'll give you the plan. Just flag: when you get to step 2, actually go find those answers yourself before wiring it — don't let me hand you the wiring too.

## Checklist — MLOps Orchestration Layer

### Step 1: Resolve the missing dependency
- [ ] Check if a `model_registry` class exists anywhere in the repo (grep for `class.*Registry` or `.rollback()`)
- [ ] If it doesn't exist: you need a minimal one with at minimum `.rollback()` and probably `.get_current_version()`, `.promote(version)`. This can be dead simple — even just tracking "current" vs "previous" model path in a JSON file and swapping a symlink/copy on rollback. Don't over-engineer this into a full registry service.

### Step 2: Trace your data contracts (do this yourself, on paper or a comment block)
- [ ] Write down: what exact columns/shape does `compute_feature_drift` need for `training_df` and `current_df`?
- [ ] Write down: where does `current_df` come from at runtime — `PredictionLogger.load_prediction_logs()`?
- [ ] Write down: what does `MetricsMonitor.compute_rolling_metrics()` return, and does it match what `detect_performance_drift` expects as input?
- [ ] Confirm: does `detect_performance_drift` need a filename string ("rolling_24h.json") — where does that get generated, and by what?

### Step 3: Build the orchestrator script (`pipeline/monitor_loop.py` or similar)
- [ ] Instantiate all four/five classes once at the top (`DriftDetector`, `MetricsMonitor`, `AlertManager`, `Pipeline`, `ModelRegistry`)
- [ ] Write `run_monitoring_cycle()` per the skeleton from my last message — call in dependency order, pass outputs forward
- [ ] Wrap in either a loop with `time.sleep(interval)`, or leave callable and invoke externally
- [ ] Add basic logging (not print statements — use Python's `logging` module, you already import it in `pipeline.py`) so a cycle's outcome is traceable

### Step 4: Fix the pre-existing bugs before testing (from last message)
- [ ] `AlertManager.log_state` path-join bug
- [ ] Missing `logs/` and `flags/` directory creation before writes
- [ ] `"taxi_demand_xgboost_v2"` → correct model name, grep whole repo for other `xgboost` leftovers

### Step 5: Prove it works — this is the part most people skip and it's the part that actually matters for hirability
- [ ] Manufacture a synthetic drift event: take your real `training_df`, create a `current_df` with deliberately shifted distributions (e.g. `current_df['some_feature'] *= 3`)
- [ ] Run one monitoring cycle against it, confirm severity escalates correctly and `trigger_action` fires the right branch
- [ ] Do the same for a synthetic performance-degradation case (fake bad predictions logged)
- [ ] Screenshot/log output of an actual halt + rollback happening
- [ ] Write 4-5 sentences in your README describing this test, with the screenshot — this is the artifact that turns "I wrote monitoring code" into "I demonstrated a working monitoring system" on your resume/portfolio

### Step 6: Only after Step 5 passes — deployment layer
- [ ] Dockerfile for FastAPI serving app
- [ ] GitHub Actions: lint + basic test on push
- [ ] Deploy to Render/Railway/Fly free tier
- [ ] Point orchestrator at deployed instance, or run as a scheduled job (cron/GitHub Actions scheduled workflow — free, no infra needed) rather than a long-running `while True` loop, which is unnecessary complexity for a portfolio project

Step 5 is the one to not skip. Working code nobody's run end-to-end isn't a demonstrated capability yet — it's a hypothesis.
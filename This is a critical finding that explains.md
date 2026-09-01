
**For the full multi-GB dataset**, doing `toPandas()` defeats the point of PySpark. Once you have the basic flow working on a sample, the production-grade fix is Apache Sedona (formerly GeoSpark) — a Spark-native spatial library. Add a comment in the code saying this explicitly:

```python
# NOTE: toPandas() here is acceptable for samples / dev runs.
# For production scale (multi-GB), replace with Apache Sedona:
# https://sedona.apache.org/latest-snapshot/api/python/reference/
```

This is a legitimate architectural decision and a good answer to "how would you scale this?" in an interview.

### Step 4 — Fix the pipeline `run()` order

The current `run()` calls things in the wrong order now that spatial join exists. Fix to:

```python
def run(self):
    df = self.load(RAW_PATH, fmt="csv")
    df = self.rename_columns(df)          # tpep_pickup_datetime → pickup_datetime
    df = self.add_zone_ids(df)            # NEW: coordinates → zone_id
    self.validate_schema(df)              # NOW zone_id exists, validation will pass
    df = self.clean(df)                   # filter bad rows
    df = self.aggregate_demand(df)        # trip rows → zone-hour demand counts
    df = self.filling_missing_zeros(df)   # fill gaps with zero demand
    self.save(df)
```

### Step 5 — Add geopandas to requirements.txt

It's already in `requirements.txt` but unversioned. Pin it and add shapely too:

```
geopandas==0.14.4
shapely==2.0.4
pyarrow==16.1.0
```

---

## What Doesn't Change

Everything **downstream of the ETL** is fine as-is (modulo the bugs already in the Phase 3–6 TODO):

- `features/engineer.py` — already expects `zone_id`, `hour_timestamp`, `demand` ✓
- `models/train.py` — operates on the feature matrix, not raw data ✓
- `api/predictor.py` — takes `zone_id` as input from the user ✓

The ETL is the only layer that touches raw trip data. Fix it to output clean zone-hour parquet, and the rest of the pipeline works on the correct shape.

---

## Revised Phase 2 order in your TODO

Replace the old Phase 2 ETL task list with this sequence:

- [ ] Download TLC taxi zone shapefile → `data/raw/taxi_zones/`
- [ ] Add `DATETIME_FORMAT = "dd-MM-yyyy HH.mm"` to `config/settings.py`
- [ ] Fix `COLUMN_MAP` to map actual raw column names
- [ ] Fix `REQUIRED_COLUMNS` — remove `zone_id` (doesn't exist pre-join)
- [ ] Fix all the bugs from the original Phase 2 list (alias `F` vs `f`, `far_amount` typo, `fare_amount <= MAX_DISTANCE` wrong column, method name mismatch)
- [ ] Add `add_zone_ids()` method to `SparkETLPipeline`
- [ ] Fix `rename_columns` → `add_zone_ids` → `validate_schema` → `clean` order in `run()`
- [ ] Add `geopandas`, `shapely` to `requirements.txt` with pinned versions
- [ ] Test on your 21-row sample — confirm output parquet has columns `zone_id`, `hour_timestamp`, `demand` with sensible values before running on the real data

Everything else in the original Phase 1–9 TODO stays the same — just insert this before the existing ETL bullet points in Phase 2.



Two separate questions here — where to get the data, and how to split it correctly. The second one is more important and has a wrong answer that will silently ruin your model.

---

## Part 1 — Getting the data, and a decision you need to make first

Your sample shows **2015 data with longitude/latitude**. The NYC TLC changed their format in 2017 — from 2017 onward, every trip record has `PULocationID` (which is `zone_id` directly, integer 1–263). No spatial join needed.

This means you have a real choice:

| Dataset | Format | Zone ID | Spatial join needed | Recommended? |
|---|---|---|---|---|
| 2015 (what you have) | lat/long | ❌ must derive | ✅ yes, complex | Only if you want the challenge |
| 2022 (matches your code's `PUBLIC_HOLIDAYS_2022`) | `PULocationID` | ✅ already there | ❌ no | **Yes — use this** |

**Use 2022 data.** Your code already references `PUBLIC_HOLIDAYS_2022` and the settings file is built around 2022 dates — so this is clearly what the code was written for. The 2015 sample you showed was probably just a quick Google for "NYC taxi sample data" without checking the year.

Using 2022 data also means the `COLUMN_MAP` becomes straightforward:
```python
COLUMN_MAP = {
    "tpep_pickup_datetime": "pickup_datetime",
    "PULocationID": "zone_id",
    "tpep_dropoff_datetime": "dropoff_datetime",
}
```
And you can remove the entire `add_zone_ids()` spatial join step — `zone_id` already exists in the raw data.

**Download a single month** to start — March 2022 is a good pick (explained below):
```
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-03.parquet
```
One month of 2022 yellow taxi data is roughly 35–45MB as parquet and about 3 million rows — large enough to be realistic, small enough that local Spark iterations take seconds, not minutes. Download directly into `data/raw/`.

---

## Part 2 — How to split correctly (this is where people get it badly wrong)

**The rule for any forecasting/time-series project: never random split.** This is not a style preference — random splitting on time-series data is a data leakage bug. Here's why it matters concretely for your project:

Your model uses lag features — `lag_1h`, `lag_24h`, `lag_168h` (demand 1 hour ago, 1 day ago, 1 week ago). If you random split, your test set contains rows from January and your training set contains rows from February. The model "predicts" January demand while having seen February lag values during training. That's future data leaking into the past. Your metrics look great, your model is useless.

**The correct approach: chronological split with a gap.**

For a full year of data the standard cut is roughly:

```
Jan 2022 ──────────────── Aug 2022 | Sep ─ Oct | Nov ─ Dec
         TRAIN (70%)               | VAL (15%) | TEST (15%)
```

For your local single-month dev sample (March 2022), scale it down proportionally:

```
Mar 1 ──────────────── Mar 21 | Mar 22 ─ Mar 26 | Mar 27 ─ Mar 31
      TRAIN (21 days)         | VAL (5 days)     | TEST (5 days)
```

In your `config/settings.py` for the dev run:
```python
DATA_START_DATE  = "2022-03-01"
DATA_END_DATE    = "2022-03-31"
TRAIN_END_DATE   = "2022-03-21"
VAL_END_DATE     = "2022-03-26"
TEST_START_DATE  = "2022-03-27"
```

For the full production run (when you scale to the full year):
```python
DATA_START_DATE  = "2022-01-01"
DATA_END_DATE    = "2022-12-31"
TRAIN_END_DATE   = "2022-08-31"
VAL_END_DATE     = "2022-10-31"
TEST_START_DATE  = "2022-11-01"
```

---

## Why March specifically (avoiding sampling bias)

If you pick the wrong month, your dev results will mislead you about whether the model actually works. Avoid:

- **January** — post-New Year slump, atypically low demand, not representative
- **February** — shortest month, Super Bowl weekend anomaly
- **July/August** — summer slump, many New Yorkers leave, atypically low
- **November/December** — Thanksgiving, Christmas, New Year all distort demand heavily

**March** is the cleanest single month: no major holidays, no weather extremes, demand patterns are typical weekday/weekend cycles. You'll get a representative picture of what the model actually needs to learn.

If you want to stress-test robustness later, download a second month — **October** is the best complement to March (similar demand levels, no anomalies, captures autumn patterns that March misses).

---

## One more thing: don't sample rows, sample time periods

When you want to go smaller than one month during debugging (e.g., just to test that the ETL runs), do this:

```python
# Good — keeps temporal structure intact
df = df.filter(f.col("pickup_datetime").between("2022-03-01", "2022-03-07"))

# Bad — breaks lag features, destroys temporal structure
df = df.sample(fraction=0.1)
```

A random 10% sample means your `lag_1h` feature will almost always be null (the row from an hour ago is probably in the 90% you dropped). The model will silently learn from zeroed-out lag features and you'll have no idea. Always filter by time range, never by random fraction, when you need a smaller dev subset.





This is great news — this data is much cleaner than the 2015 sample. Here's exactly what it tells you and what changes in your code:

## What this confirms

**`PULocationID` exists and is already an integer zone ID.** Zone 90, 148, 132, 211, etc. — exactly what your whole pipeline expects. This kills the spatial join entirely. No geopandas, no shapefile, no `add_zone_ids()` method needed. Delete that from the Phase 2 plan.

**Datetime is standard ISO format** — `"2022-03-01 00:13:08"`. Not the dot-separated mess from the 2015 sample. PySpark handles this natively with a plain `cast(TimestampType())`. So the `DATETIME_FORMAT = "dd-MM-yyyy HH.mm"` I told you to add earlier — scratch that, it was based on the wrong dataset.

**Two new columns exist that weren't in 2015:** `congestion_surcharge` and `airport_fee`. Your ETL's `REQUIRED_COLUMNS` check will fail if it expects the 2015 schema. These are fine to keep — they're legitimate features.

## What this fixes in `config/settings.py`

Your `COLUMN_MAP` should be exactly this — nothing more complex:

```python
COLUMN_MAP = {
    "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
    "PULocationID": "zone_id",
    "DOLocationID": "dropoff_zone_id",
    "RatecodeID": "rate_code",
}

REQUIRED_COLUMNS = [
    "tpep_pickup_datetime", "PULocationID",
    "fare_amount", "trip_distance", "passenger_count"
]
```

Note `REQUIRED_COLUMNS` checks the **raw** column names before renaming, since `validate_schema` runs before `rename_columns` in your current pipeline order.

## One real data problem you need to handle

Row 8 in your sample is a **reversal/cancellation record** — look at it:

```
fare_amount = -2.5, extra = -0.50, total_amount = -3.80
trip_distance = 0.10, pickup_zone = dropoff_zone = 145
```

These negative-value rows exist throughout the real dataset and will corrupt your demand aggregation and feature engineering if not filtered. Add this to your clean step in `etl/spark_pipeline.py`:

```python
# Filter cancellation/reversal records
df = df.filter(f.col("fare_amount") > 0)
df = df.filter(f.col("trip_distance") > 0)
```

Also notice rows 8 and 9 have **identical pickup timestamps, identical zones, and are clearly a correction pair** (one negative, one positive for the same trip). The positive record (row 9) is fine to keep after the `fare_amount > 0` filter — the negative one gets dropped automatically.

## Updated COLUMN_MAP impact on the rest of the pipeline

The aggregation step in your ETL needs to group by `zone_id` and `hour` — make sure it references the **renamed** column names, not the raw ones:

```python
def aggregate_demand(self, df):
    df = df.withColumn(
        "hour_timestamp",
        f.date_trunc("hour", f.col("pickup_datetime"))
    )
    return df.groupBy("zone_id", "hour_timestamp").agg(
        f.count("*").alias("demand")
    )
```

## Nothing else changes from the original TODO

Every other Phase 1–9 task stands exactly as written. The dataset being 2022 format just removes the spatial join detour — everything else (the config bugs, ETL alias bugs, training bugs, API bugs, monitoring bugs) are all still there and still need fixing in the same order.

Start Phase 1 (`config/settings.py`) — with this dataset confirmed you now have all the information you need to define every constant correctly.
`demand` doesn't exist in the raw data — it's something your ETL **creates** by aggregating the trip-level rows. This is actually the most important transformation in the whole pipeline, and it's worth understanding it clearly because it'll come up in every interview.

## What's actually happening

Your raw data has one row per **trip**. Your model needs one row per **zone per hour** with a count of how many trips happened. That count *is* demand.

Visually, from your sample:

```
RAW (trip-level) — 10 rows, all from 2022-03-01 00:xx:xx
─────────────────────────────────────────────────
zone_id  pickup_datetime       fare_amount  ...
90       2022-03-01 00:13:08   10.0
148      2022-03-01 00:47:52   10.5
132      2022-03-01 00:02:46   52.0
211      2022-03-01 00:52:43   11.0
138      2022-03-01 00:15:35   25.0
132      2022-03-01 00:11:57   43.5        ← second trip from zone 132
...

AFTER AGGREGATION (zone-hour level) — what the model trains on
─────────────────────────────────────────────────
zone_id  hour_timestamp        demand
90       2022-03-01 00:00:00   1      ← 1 trip from zone 90 in this hour
148      2022-03-01 00:00:00   1
132      2022-03-01 00:00:00   2      ← 2 trips from zone 132 in this hour
211      2022-03-01 00:00:00   1
138      2022-03-01 00:00:00   1
...
```

One group per `(zone_id, hour)`. Count of trips in that group = `demand`. That's it.

---

## The code that creates it

This goes in `etl/spark_pipeline.py` as the `aggregate_demand` method:

```python
def aggregate_demand(self, df):
    """
    Transforms trip-level rows into zone-hour demand counts.
    
    This is where the target column 'demand' is created.
    demand = number of trips that started in a given zone 
             during a given hour.
             
    Input:  one row per trip
    Output: one row per (zone_id, hour_timestamp) pair
    """
    print("\n[4/5] Aggregating trips → zone-hour demand...")
    
    # Truncate pickup_datetime to the hour
    # e.g. 2022-03-01 00:13:08 → 2022-03-01 00:00:00
    df = df.withColumn(
        "hour_timestamp",
        f.date_trunc("hour", f.col("pickup_datetime"))
    )
    
    # Count trips per zone per hour → this count IS demand
    df = df.groupBy("zone_id", "hour_timestamp").agg(
        f.count("*").alias("demand")
    )
    
    row_count = df.count()
    print(f"  → {row_count:,} zone-hour combinations produced")
    return df
```

---

## Why this also creates the zero-demand problem

After aggregation you only have rows where **at least one trip happened**. Zone 50 at 3am might have zero trips — that row simply won't exist in the output. But your model needs to know about zero-demand hours too, otherwise it never learns "this zone is quiet at night."

That's exactly what `filling_missing_zeros` handles — it creates the missing `(zone_id, hour_timestamp)` combinations and fills `demand = 0` for them. This is why that method exists and must run right after `aggregate_demand` in your pipeline.

The full `run()` order makes complete sense now:

```python
def run(self):
    df = self.load(RAW_PATH)              # 3M rows (one per trip)
    df = self.rename_columns(df)          # tpep_pickup_datetime → pickup_datetime, PULocationID → zone_id
    self.validate_schema(df)              # check required raw columns exist
    df = self.clean(df)                   # drop negatives, nulls, bad values
    df = self.aggregate_demand(df)        # 3M trip rows → ~60,000 zone-hour rows, demand column created here
    df = self.filling_missing_zeros(df)   # add zero-demand rows for quiet zone-hours
    self.save(df)                         # write parquet to PROCESSED_PATH
```

From 3 million rows down to roughly 60,000 (263 zones × 744 hours in March). That's the shape everything downstream expects.
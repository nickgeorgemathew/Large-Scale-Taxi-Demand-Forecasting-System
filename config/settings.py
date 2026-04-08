from config.settings import (
    PROCESSED_PATH, FEATURES_PATH,
    LAG_HOURS, ROLLING_WINDOWS,
    TRAIN_END_DATE, VAL_END_DATE,TEST_START_DATE,
    TARGET_COLUMN, FEATURE_COLUMNS
)
from config.settings import (
    PROCESSED_PATH, FEATURES_PATH,
    LAG_HOURS, ROLLING_WINDOWS,
    PUBLIC_HOLIDAYS_2022,
    TRAIN_END_DATE, VAL_END_DATE,
    TARGET_COLUMN, FEATURE_COLUMNS
)
from config.settings import (
    PROCESSED_PATH, FEATURES_PATH,
    LAG_HOURS, ROLLING_WINDOWS,
    TRAIN_END_DATE, VAL_END_DATE,TEST_START_DATE,
    TARGET_COLUMN, FEATURE_COLUMNS
)
from config.settings import(
    RAW_DATA_PATH, RAW_DATA_FORMAT,
    PROCESSED_PATH,
    COLUMN_MAP, REQUIRED_COLUMNS,
    VALID_ZONE_MIN, VALID_ZONE_MAX,
    MIN_FARE, MAX_FARE,
    MIN_DISTANCE, MAX_DISTANCE,
    MIN_PASSENGER, MAX_PASSENGER,
    DATA_START_DATE, DATA_END_DATE,
    TIME_GRANULARITY, FILL_MISSING_ZEROS,
    SPARK_APP_NAME, SPARK_SHUFFLE_PARTITIONS, SPARK_DRIVER_MEMORY
)
from config.settings import LOG #file location for storing parquet logs
# feature_cols = [
        #     'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
        #     'is_rush_am', 'is_rush_pm', 'zone_id', 'borough_encoded',
        #     'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_24h', 'lag_48h', 'lag_168h',
        #     'roll_mean_3h', 'roll_mean_6h', 'roll_mean_24h',
        #     'roll_std_6h', 'mean', 'std'   
        # ]
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from pyspark.sql import SparkSession
from pyspark.sql import Functions as f
from pyspark.sql.types import(
    StructType,StructField,IntegerType,FloatType,TimestampType,stringType
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




def create_spark_session()->SparkSession:
    spark=(SparkSession.builder.appName(SPARK_APP_NAME).master("local[*]").config("spark.driver.memory",SPARK_DRIVER_MEMORY).config("spark.sql.shuffle.partition",SPARK_SHUFFLE_PARTITIONS).config("spark.sql.adaptive.enabled","true").config("spark.driver.extraJavaOptions", "-Dlog4j.logLevel=WARN")
        .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    print(f"SparkSession created | version: {spark.version}")
    return spark






class TaxiDemandEtl:
    def __init__(self,spark:SparkSession):
        self.spark=spark


    
    def load_raw(self,path:str,fmt:str):

        print(f"\n[1/5] Loading raw data from: {path}")

        if fmt=="parquet":
            df=self.spark.read.parquet(path)
        elif fmt=="csv":
            df=(self.spark.read
                .option("header",True)
                .option("inferSchema",True)
                .csv(path))
        else:
            raise ValueError(f"unsupported format:{fmt}. Use 'parquet' or 'csv' .")
        
        df.cache()

        row_count=df.count()
        print(f"loaded {row_count:,} raw rows")
        print(f" columns:{df.columns}")
        return df
    
    def rename_columns(self,df):
        for raw_name,standard_name in COLUMN_MAP.items():
            if raw_name in df.columns:
                df=df.withColumnRenamed(raw_name,standard_name)
        return df
    





    def validate_schema(self,df)->None:
        missing=[col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                f"Schema validation failed. Missing columns: {missing}\n"
                f"Available columns: {df.columns}\n"
                f"Check COLUMN_MAP in config/settings.py"
            )
        print(f"  → Schema validation passed. All required columns present.")







    def clean(self,df):
        print(f"\n[2/5] cleaning data....")
        initial_count=df.count()

        df=df.dropna(subset=["pickup_datetime","zone_id","fare_amount","trip_distance"])


        df=df.withColumn("zone_id",F.col("zone_id").cast(IntegerType()))
        df=df.withColumn("far_amount",F.col("fare_amount").cast(FloatType()))
        df=df.withColumn("trip_distance",F.col("trip_distance").cast(FloatType()))
        df=df.withColumn("pickup_datetime",F.col("pickup_datetime").cast(TimestampType()))




        df=df.filter(
            F.col("zone_id").between(VALID_ZONE_MIN,VALID_ZONE_MAX)
        )

        df=df.filter(
            (F.col("fare_amount")>=MIN_FARE)&
            (F.col("fare_amount")<=MAX_FARE)
        )

        df=df.filter(
            (F.col("trip_distance")>=MIN_DISTANCE)&
            (F.col("fare_amount")<=MAX_DISTANCE)
        )


        if "passenger_count" in df.columns:
            df=df.filter(
                F.col("passenger_count").between(MIN_PASSENGER,MAX_PASSENGER)
            )

        df=df.filter(
            (F.col("pickup_datetime")>=F.lit(DATA_START_DATE))&
            (F.col("pickup_datetime")<=F.lit(DATA_END_DATE))
        )



        df=df.select(REQUIRED_COLUMNS)

        clean_count=df.count()
        removed=initial_count-clean_count
        pct = (removed / initial_count) * 100 if initial_count > 0 else 0
        print(f"  → Removed {removed:,} bad rows ({pct:.1f}%)")
        print(f"  → Clean rows remaining: {clean_count:,}")

        return df







    def aggregate_demand(self,df):
        print(f"\n[3/5] Aggregating demand by zone and hour...")
        df=df.withColumn("hour_timestamp",
                        F.date_trunc(TIME_GRANULARITY,F.col("pickup_datetime")))

        demand_df=(
            df.groupby("zone_id","hour_timestamp")
            .agg(
                F.count("*").alias("demand"),
                F.avg("fare_amount").alias("avg_fare"),
                F.avg("trip_distance").alias(
                    "avg_distance"
                )
            )
        )
        print(f"  → Aggregated to {demand_df.count():,} (zone, hour) records")
        print(f"  → Date range: {demand_df.agg(F.min('hour_timestamp')).collect()[0][0]} "
                f"to {demand_df.agg(F.max('hour_timestamp')).collect()[0][0]}")
        return demand_df





    def filling_missing_zeros(self,df):
        if not FILL_MISSING_ZEROS:
            return df

        print(f"\n[4/5] Filling missing zero-demand hours...")

        zones_df=df.select("zone_id").distinct()
        date_range_df=self.spark.sql(f"""
            SELECT explode(
                    sequence(
                            timestamp('{DATA_START_DATE}),
                            timestamp('{DATA_END_DATE}),
                            interval 1 hour
                                             )
                        )AS hour_timestamp            
             """
        )
        full_grid=zones_df.crossJoin(date_range_df)

        complete_df=(
            full_grid
            .join(df,on=["zone_id", "hour_timestamp"], how="left")
            .fillna({
                "demand":       0,
                "avg_fare":     None,
                "avg_distance": None
            })
        )
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


    
    def load_raw(self,path:str,fmt:str)->"DataFrame":


        if fmt=="parquet":
            df=self.spark.read.parquet(path)
        elif fmt=="csv":
            df=(self.spark.read
                .option("header",True)
                .option("inferSchema",True)
                .csv(path))
        else:
            raise ValueError(f"unsupported format:{fmt}. Use 'parquet' or 'csv' .")
import logging

from pyspark.sql import SparkSession


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main():
    logger.info("Creating Spark Session ...")
    spark = SparkSession\
        .builder\
        .appName("pyspark_training")\
        .getOrCreate()
    
    logger.info(spark)

    df = spark.read.csv("s3s://mlops204-dataproc-bucket/data/BankChurners.csv", header=True, inferSchema=True)

    logger.info(df.limit(10).toPandas())


if __name__ == "__main__":
    main()
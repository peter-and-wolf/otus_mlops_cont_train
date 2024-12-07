import logging

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def data_prep_pipeline(spark):
  
  gender_index = StringIndexer(inputCol='Sex', outputCol='SexIndex')
  gender_encoder = OneHotEncoder(inputCol='SexIndex', outputCol='SexVector')

  embark_indexer = StringIndexer(inputCol='Embarked', outputCol='EmbarkedIndex')
  embark_encoder = OneHotEncoder(inputCol='EmbarkedIndex', outputCol='EmbarkVector')

  assembler = VectorAssembler(
    inputCols=[
      'Pclass',
      'SexVector',
      'Age',
      'SibSp',
      'Parch',
      'Fare',
      'EmbarkVector'
    ],
    outputCol='features'
  )

  return Pipeline(stages=[
    gender_index,
    embark_indexer,
    gender_encoder,
    embark_encoder,
    assembler
  ])


# def train():

def main():
  logger.info("Creating Spark Session ...")
  spark = SparkSession\
      .builder\
      .appName("pyspark_training")\
      .getOrCreate()
    
  logger.info(spark)

  df = spark.read.csv("s3s://mlops204-dataproc-bucket/data/titanic/train.csv", header=True, inferSchema=True)

  logger.info(df.limit(10).toPandas())


if __name__ == "__main__":
  main()
import os
import logging
from functools import partial

import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, SparkTrials

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def data_prep_pipeline():
  
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
    outputCol='Features'
  )

  return Pipeline(stages=[
    gender_index,
    embark_indexer,
    gender_encoder,
    embark_encoder,
    assembler
  ])


# Определяем пространство поиска для hyperopt
search_space = {
  'regParam': hp.lognormal('regParam', 0, 1.0),
  'fitIntercept': hp.choice('fitIntercept', [False, True])
}


def objective(params, train_data, test_data):

  lr = LogisticRegression()\
    .setMaxIter(1000)\
    .setRegParam(params['regParam'])\
    .setFeaturesCol('Features')\
    .setLabelCol('Survived')

  evaluator = BinaryClassificationEvaluator()\
    .setLabelCol('Survived')

  lg_model = lr.fit(train_data)

  auc = evaluator.evaluate(lg_model.transform(test_data))

  with mlflow.start_run():
    mlflow.set_tag('experimentalist', 'robot')
    mlflow.log_params(params)
    mlflow.log_metric('auc', auc)
  
  return {'loss': -auc, 'status': STATUS_OK}


def main():

  logger.info("Creating Spark Session ...")
  
  spark = SparkSession\
    .builder\
    .appName('Spark ML Research')\
    .config('spark.sql.repl.eagerEval.enabled', True) \
    .getOrCreate()
    
  logger.info(spark)

  df = spark.read.csv(
    's3a://mlops204-dataproc-bucket/data/titanic/train.csv', 
    header=True, 
    inferSchema=True
  )

  df = df.select([
    'Survived',
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Embarked'
  ]).na.drop()

  logger.info(df.limit(10).toPandas())

  dataproc = data_prep_pipeline()

  ready_data = dataproc.fit(df).transform(df)

  train_data, test_data = ready_data.randomSplit([.7, .3])

  trials = Trials()

  mlflow.set_experiment('classification')

  best = fmin(
    fn=partial(
        objective, 
        train_data=train_data,
        test_data=test_data
    ),
    space=search_space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials
  )


if __name__ == "__main__":

  os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://storage.yandexcloud.net'
  os.environ['MLFLOW_TRACKING_URI']='http://10.0.0.35:8000'

  main()

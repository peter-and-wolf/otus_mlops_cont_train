# otus_mlops_cont_train

1. Запуск spark-задания с внедрением python-окружения, архив с которым предварительно загружен в s3-бакет по пути `s3a://pyspark-venvs/mlflow-dataproc-2.1.18.tar.gz`.

```bash
spark-submit --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./venv/bin/python --conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./venv/bin/python --conf spark.yarn.dist.archives=s3a://pyspark-venvs/mlflow-dataproc-2.1.18.tar.gz#venv --deploy-mode=cluster test.py
```
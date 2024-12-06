## Запуск mlflow

На виртуальной машине в YC, которая находитcя в той же сети и подсети, что и ваш dataproc-cluster:

1. Создайте файлик .env, в который добавьте примерно следующее:

```bash 
PG_USER=mlflow
PG_PASSWORD=mlflow
PG_DATABASE=mlflow
```

1. Запустите Postgres в docker-compose:

```bash
docker-compose up -d
```
1. Запустите MLFlow Tracking Server, который будет сохранять метрики в Postgres в докере, а модели и прочие артефакты – в S3-бакете в YC.
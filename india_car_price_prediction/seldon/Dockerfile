FROM seldonio/mlserver:1.3.5-slim

WORKDIR /app

# Установка MLflow runtime и зависимостей, совместимых с Python 3.8
RUN pip install \
    mlserver-mlflow==1.3.5 \
    mlflow==2.11.3 \
    boto3 \
    catboost==1.2.5 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    scikit-learn==1.3.2

# Копирование конфигурации
COPY model-settings.json /mnt/models/

EXPOSE 8080 8081

CMD ["mlserver", "start", "/mnt/models"]
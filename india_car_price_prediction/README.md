<h2> MLOps-стек для предсказания цен на автомобили в Индии: </h2>

- **MLflow** (трекинг экспериментов, хранение моделей)
- **PostgreSQL** (база метаданных MLflow)
- **MinIO** (S3-совместимое хранилище артефактов)
- **FastAPI** (REST-сервис для инференса)
- **Seldon MLServer** (оптимизированное обслуживание моделей, *опционально*)

<h2> Установка зависимостей </h2>

pip install -r requirements.txt

<h2> Запуск инфраструктуры </h2>

docker-compose up -d

<h2> Обучение модели </h2>

python -m machine_learning.train_car_price_model

<h2> Проверка API </h2>

Health check:
curl http://localhost:8000/health

Тестовое предсказание:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Make": "Honda",
    "Model": "City",
    "Year": 2018,
    "Kilometer": 45000,
    "Fuel Type": "Petrol",
    "Transmission": "Manual",
    "Location": "Mumbai",
    "Color": "White",
    "Owner": "First",
    "Seller Type": "Individual",
    "Engine": 1498.0,
    "Max Power": 98.6,
    "Max Torque": 200.0,
    "Drivetrain": "FWD",
    "Length": 3995.0,
    "Width": 1695.0,
    "Height": 1505.0,
    "Seating Capacity": "5",
    "Fuel Tank Capacity": 40.0
  }'

  <h2> Seldon MLServer </h2>

Запуск Seldon:
docker-compose --profile seldon up -d

Проверка:
curl http://localhost:9002/v2/models/car-price-model/ready

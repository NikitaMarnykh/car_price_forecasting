"""
FastAPI сервис для предсказания цен на автомобили.

Основные эндпоинты:
- GET /health - проверка здоровья сервиса
- POST /predict - предсказание цены автомобиля
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager


from app.schemas import CarFeatures, PredictionResponse, HealthResponse, ErrorResponse
from app.exceptions import (
    ModelLoadException, PredictionException, PreprocessingException,
    ValidationException
)
from app.utils import (
    setup_mlflow_config, load_model_from_mlflow, load_preprocessing_artifacts,
    check_mlflow_connection, get_model_info, InputPreprocessor, PricePredictor
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Глобальные переменные для модели и препроцессора
model = None
predictor = None
selected_features = None
service_start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения.
    Загружает модель при старте и освобождает ресурсы при завершении.
    """
    global model, predictor, selected_features, service_start_time

    # Запуск сервиса
    service_start_time = time.time()
    logger.info("Starting Car Price Prediction API...")

    try:
        # Настройка MLflow
        setup_mlflow_config()

        # Загрузка модели
        logger.info("Loading model from MLflow...")
        model, selected_features = load_model_from_mlflow()

        if model is None:
            raise ModelLoadException("car_price_model_catboost", "Model is None after loading")

        # Загрузка артефактов препроцессинга
        logger.info("Loading preprocessing artifacts...")
        artifacts = load_preprocessing_artifacts()

        # Создание препроцессора
        preprocessor = InputPreprocessor(
            expected_features=selected_features,
            frequency_mappings=artifacts.get('frequency_mappings'),
            ohe_encoder=artifacts.get('ohe_encoder'),
            ohe_feature_names=artifacts.get('ohe_feature_names'),
            rare_categories_map=artifacts.get('rare_categories_map')
        )

        # Создание предиктора
        predictor = PricePredictor(model=model, preprocessor=preprocessor)

        logger.info("Service started successfully!")

    except ModelLoadException as e:
        logger.error("Failed to load model: %s", e)
        # Сервис может работать без модели для проверки здоровья
        model = None
        predictor = None

    except Exception as e:
        logger.error("Unexpected error during startup: %s", e)
        model = None
        predictor = None

    yield

    # Завершение работы
    logger.info("Shutting down Car Price Prediction API...")
    model = None
    predictor = None


# Создание приложения FastAPI
app = FastAPI(
    title="Car Price Prediction API",
    description="API для предсказания цен на автомобили с использованием CatBoost и MLflow",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ModelLoadException)
async def model_load_exception_handler(request, exc):
    """Обработчик ошибок загрузки модели."""
    logger.error("Model load error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="ModelLoadError",
            message=str(exc),
            detail="The ML model could not be loaded. Please check MLflow connection."
        ).dict()
    )


@app.exception_handler(PredictionException)
async def prediction_exception_handler(request, exc):
    """Обработчик ошибок предсказания."""
    logger.error("Prediction error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="PredictionError",
            message=str(exc),
            detail="An error occurred during prediction."
        ).dict()
    )


@app.exception_handler(PreprocessingException)
async def preprocessing_exception_handler(request, exc):
    """Обработчик ошибок предобработки."""
    logger.error("Preprocessing error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="PreprocessingError",
            message=str(exc),
            detail="Input data preprocessing failed."
        ).dict()
    )


@app.exception_handler(ValidationException)
async def validation_exception_handler(request, exc):
    """Обработчик ошибок валидации."""
    logger.error("Validation error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="ValidationError",
            message=str(exc),
            detail="Input validation failed."
        ).dict()
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Проверка здоровья сервиса.

    Возвращает статус сервиса, информацию о модели и подключении к MLflow.
    """
    global model, selected_features, service_start_time

    mlflow_status = check_mlflow_connection()

    model_info = get_model_info(model, selected_features if selected_features else [])

    uptime_seconds = time.time() - service_start_time if service_start_time else 0

    health_data = {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "mlflow_connected": mlflow_status,
        "model_type": model_info.get("model_type"),
        "features_count": model_info.get("features_count"),
        "uptime_seconds": round(uptime_seconds, 2)
    }

    logger.info("Health check: %s", health_data["status"])

    return health_data


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(features: CarFeatures):
    """
    Предсказание цены автомобиля.

    Принимает характеристики автомобиля и возвращает предсказанную цену в INR.

    :param features: Характеристики автомобиля
    :return: Предсказанная цена
    """
    global predictor

    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service is unavailable."
        )

    try:
        start_time = time.time()

        # Выполняем предсказание
        predicted_price = predictor.predict(features)

        prediction_time = time.time() - start_time

        logger.info(
            "Prediction completed in %.3f seconds. Price: %.2f INR",
            prediction_time, predicted_price
        )

        response = PredictionResponse(
            predicted_price=predicted_price,
            currency="INR",
            model="car_price_model_catboost",
            processing_time=round(prediction_time, 4)
        )
        return response

    except PreprocessingException as e:
        logger.error("Preprocessing failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except PredictionException as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction."
        )


@app.get("/", tags=["Info"])
async def root():
    """Корневой эндпоинт с информацией о сервисе."""
    return {
        "service": "Car Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
"""
Утилиты для Car Price Prediction API.

Содержит:
- Кастомные исключения для API
- Препроцессор входных данных (InputPreprocessor)
- Предиктор (PricePredictor)
- Функции для работы с MLflow и загрузки модели
"""

import mlflow

import mlflow.catboost

import logging

import os

import joblib

from typing import Any

import pandas as pd

from app.exceptions import PreprocessingException, PredictionException, ModelLoadException

logger = logging.getLogger(__name__)

# КОНСТАНТЫ
# Признаки с высокой кардинальностью (требуют частотного кодирования)
HIGH_CARDINALITY_FEATURES = ['Make', 'Model', 'Location', 'Color']

# Категориальные признаки (как в пайплайне обучения)
CATEGORICAL_FEATURES = [
    'Make', 'Model', 'Fuel Type', 'Transmission', 'Location',
    'Color', 'Owner', 'Seller Type', 'Drivetrain', 'Seating Capacity'
]

# Название модели в MLflow Registry
MODEL_NAME = "car_price_model_catboost"


# ПРЕПРОЦЕССОР ВХОДНЫХ ДАННЫХ
def _pydantic_to_dataframe(features: Any) -> pd.DataFrame:
    """Преобразует Pydantic модель в DataFrame."""
    try:
        input_dict = features.model_dump(by_alias=True)
        df = pd.DataFrame([input_dict])
        logger.debug("Input converted to DataFrame. Shape: %s", df.shape)
        return df
    except Exception as e:
        raise PreprocessingException(
            "pydantic_to_dataframe",
            "Failed to convert input to DataFrame"
        ) from e


def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит типы колонок в соответствии с ожидаемыми моделью."""
    try:
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype('category')

        # Приводим числовые колонки к правильным типам
        numeric_columns = [
            'Year', 'Kilometer', 'Engine', 'Max Power', 'Max Torque',
            'Length', 'Width', 'Height', 'Fuel Tank Capacity'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.debug("Types converted successfully")
        return df
    except Exception as e:
        raise PreprocessingException(
            "convert_types",
            "Failed to convert column types"
        ) from e


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Гарантирует, что все колонки имеют числовой тип.
    """
    try:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df
    except Exception as e:
        raise PreprocessingException(
            "ensure_numeric",
            "Failed to convert columns to numeric types"
        ) from e


class InputPreprocessor:
    """
    Препроцессор входных данных для модели CatBoost.

    Использует сохранённый пайплайн предобработки (частоты, OHE-энкодер)
    для корректного преобразования входных данных.
    """

    def __init__(
            self,
            expected_features: list,
            frequency_mappings: dict | None = None,
            ohe_encoder: Any = None,
            ohe_feature_names: list | None = None,
            rare_categories_map: dict | None = None
    ) -> None:
        """
        :param expected_features: Список признаков, ожидаемых моделью (порядок важен!).
        :param frequency_mappings: Словарь {col: {value: frequency}} с частотами для частотного кодирования.
        :param ohe_encoder: Обученный OneHotEncoder.
        :param ohe_feature_names: Имена признаков после OHE (из encoder.get_feature_names_out()).
        :param rare_categories_map: Словарь {col: [rare_values]} для замены редких категорий на 'Other'.
        """
        self.expected_features = expected_features
        self.frequency_mappings = frequency_mappings or {}
        self.ohe_encoder = ohe_encoder
        self.ohe_feature_names = ohe_feature_names or []
        self.rare_categories_map = rare_categories_map or {}

        logger.info(
            "Preprocessor initialized. Expected features: %d, "
            "Frequency mappings: %d, OHE features: %d, Rare categories maps: %d",
            len(expected_features),
            len(self.frequency_mappings),
            len(self.ohe_feature_names),
            len(self.rare_categories_map)
        )

    def transform(self, features: Any) -> pd.DataFrame:
        """
        Выполняет полный цикл предобработки входных данных.

        :param features: Pydantic модель с характеристиками автомобиля.
        :return: DataFrame, готовый для передачи в модель.
        :raise PreprocessingException: Если на любом этапе произошла ошибка.
        """
        try:
            df = _pydantic_to_dataframe(features)
            df = _convert_types(df)
            df = self._replace_rare_categories(df)
            df = self._apply_frequency_encoding(df)
            df = self._apply_one_hot_encoding(df)
            df = self._align_features(df)
            df = _ensure_numeric(df)

            logger.debug("Preprocessing completed. Final shape: %s", df.shape)
            return df

        except PreprocessingException:
            raise
        except Exception as e:
            raise PreprocessingException(
                "transform",
                "Unexpected error during preprocessing pipeline"
            ) from e

    def _replace_rare_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Заменяет редкие категории на 'Other' в соответствии с картой,
        построенной при обучении.
        """
        if not self.rare_categories_map:
            logger.debug("No rare categories map provided. Skipping.")
            return df

        try:
            for col, rare_values in self.rare_categories_map.items():
                if col in df.columns:
                    # Конвертируем колонку в строковый тип для безопасной замены
                    df[col] = df[col].astype(str)
                    df.loc[df[col].isin(rare_values), col] = 'Other'
                    df[col] = df[col].astype('category')
                    logger.debug(
                        "Replaced %d rare categories in '%s' with 'Other'",
                        len(rare_values), col
                    )
            return df
        except Exception as e:
            raise PreprocessingException(
                "replace_rare_categories",
                "Failed to replace rare categories"
            ) from e

    def _apply_frequency_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет частотное кодирование, используя частоты из обучающей выборки.
        Для неизвестных категорий (не встречавшихся при обучении) используется 0.0.
        """
        if not self.frequency_mappings:
            logger.debug("No frequency mappings provided. Using placeholders (0.0).")
            # Fallback: заглушки для всех high-cardinality признаков
            for col in HIGH_CARDINALITY_FEATURES:
                if col in df.columns:
                    df[f'{col}_encoded'] = 0.0
                    df.drop(col, axis=1, inplace=True)
            return df

        try:
            for col, freq_dict in self.frequency_mappings.items():
                if col in df.columns:
                    # Конвертируем в строку для сопоставления с ключами словаря
                    df[col] = df[col].astype(str)
                    # Применяем сохранённые частоты, для новых категорий — 0.0
                    df[f'{col}_encoded'] = (
                        df[col]
                        .map(freq_dict)
                        .fillna(0.0)
                        .astype(float)
                    )
                    df.drop(col, axis=1, inplace=True)
                    logger.debug(
                        "Frequency encoding applied to '%s'. "
                        "Known categories: %d",
                        col, len(freq_dict)
                    )
            return df
        except Exception as e:
            raise PreprocessingException(
                "apply_frequency_encoding",
                "Failed to apply frequency encoding"
            ) from e

    def _apply_one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет One-Hot Encoding, используя сохранённый энкодер.
        """
        if self.ohe_encoder is None:
            logger.debug("No OHE encoder provided. Using placeholders for remaining categoricals.")
            # Fallback: заглушки для оставшихся категориальных колонок
            cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
            for col in cat_cols:
                if col in df.columns:
                    # Добавляем заглушки для всех возможных OHE колонок
                    for ohe_col in self.ohe_feature_names:
                        if ohe_col.startswith(f"{col}_"):
                            df[ohe_col] = 0.0
                    df.drop(col, axis=1, inplace=True)
            return df

        try:
            cat_cols = self.ohe_encoder.feature_names_in_
            # Оставляем только те колонки, которые есть в данных
            cat_cols_present = [c for c in cat_cols if c in df.columns]

            if cat_cols_present:
                # Конвертируем в строки для OHE
                for col in cat_cols_present:
                    df[col] = df[col].astype(str)

                encoded_array = self.ohe_encoder.transform(df[cat_cols_present])
                encoded_df = pd.DataFrame(
                    encoded_array,
                    columns=self.ohe_feature_names,
                    index=df.index
                )
                df = df.drop(cat_cols_present, axis=1)
                df = pd.concat([df, encoded_df], axis=1)
                logger.debug("OHE applied. Added %d features.", len(self.ohe_feature_names))

            return df
        except Exception as e:
            raise PreprocessingException(
                "apply_one_hot_encoding",
                "Failed to apply One-Hot Encoding"
            ) from e

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выравнивает признаки с ожидаемыми моделью.
        Добавляет отсутствующие с 0.0 и упорядочивает колонки.
        """
        try:
            # Добавляем отсутствующие признаки
            for feat in self.expected_features:
                if feat not in df.columns:
                    df[feat] = 0.0

            # Оставляем только нужные и в правильном порядке
            df = df[self.expected_features]

            logger.debug("Features aligned. Final shape: %s", df.shape)
            return df
        except KeyError as e:
            raise PreprocessingException(
                "align_features",
                f"Feature not found: {e}"
            ) from e
        except Exception as e:
            raise PreprocessingException(
                "align_features",
                "Failed to align features"
            ) from e


# ПРЕДИКТОР
class PricePredictor:
    """
    Выполняет предсказание цены автомобиля.
    """

    def __init__(self, model: Any, preprocessor: InputPreprocessor) -> None:
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, features: Any) -> float:
        """
        Выполняет предсказание цены.

        :param features: Pydantic модель с характеристиками автомобиля.
        :return: Предсказанная цена.
        :raise PredictionException: Если предсказание не удалось.
        """
        try:
            df = self.preprocessor.transform(features)

            # Убеждаемся, что данные в правильном формате
            if hasattr(self.model, 'feature_names_'):
                df = df[self.model.feature_names_]

            prediction_array = self.model.predict(df)
            predicted_price = float(prediction_array[0])

            if predicted_price < 0:
                logger.warning(
                    "Negative price predicted: %.2f. Clamping to 0.", predicted_price
                )
                predicted_price = 0.0

            logger.info("Prediction successful: %.2f", predicted_price)
            return round(predicted_price, 2)

        except PreprocessingException:
            raise
        except Exception as e:
            raise PredictionException(
                str(e),
                "Model prediction call failed"
            ) from e


# ФУНКЦИИ ДЛЯ РАБОТЫ С MLFLOW
def setup_mlflow_config() -> str:
    """
    Настраивает подключение к MLflow из переменных окружения.

    :return: URI MLflow Tracking Server.
    """
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = mlflow_s3_endpoint
    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info("MLflow configured. Tracking URI: %s", mlflow_tracking_uri)

    return mlflow_tracking_uri


def load_model_from_mlflow() -> tuple[Any, list | None]:
    """
    Загружает модель и метаданные из MLflow Registry.

    Порядок загрузки:
    1. Пробуем загрузить из MLflow Registry
    2. Fallback: загружаем локальные файлы

    :return: Кортеж (модель, список_признаков).
    :raise ModelLoadException: Если модель не удалось загрузить ниоткуда.
    """
    selected_features = None

    # Попытка 1: MLflow Registry
    try:
        model_uri = f"models:/{MODEL_NAME}/latest"
        logger.info("Attempting to load model from MLflow: %s", model_uri)

        model = mlflow.catboost.load_model(model_uri)

        # Пробуем загрузить selected_features из артефактов MLflow
        try:
            import tempfile

            # Создаем клиент MLflow
            client = mlflow.tracking.MlflowClient()

            # Ищем последнюю версию модели
            versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging", "Production"])
            if versions:
                run_id = versions[0].run_id
                # Скачиваем артефакт
                with tempfile.TemporaryDirectory() as tmpdir:
                    local_path = client.download_artifacts(run_id, "selected_features.joblib", tmpdir)
                    selected_features = joblib.load(local_path)
                    logger.info("Selected features loaded from MLflow artifacts. Count: %d", len(selected_features))
        except Exception as e:
            logger.warning("Could not load selected_features from MLflow artifacts: %s", e)

        logger.info("Model loaded successfully from MLflow Registry")
        return model, selected_features

    except Exception as e:
        logger.warning("Failed to load model from MLflow: %s. Trying local fallback...", e)

    # Попытка 2: Локальные файлы
    try:
        model_path = './ml_model.joblib'
        features_path = './selected_features.joblib'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local model file not found: {model_path}")

        model = joblib.load(model_path)
        logger.info("Model loaded from local file: %s", model_path)

        if os.path.exists(features_path):
            selected_features = joblib.load(features_path)
            logger.info("Selected features loaded from local file. Count: %d", len(selected_features))
        else:
            logger.warning("Local selected_features.joblib not found. Using None.")

        return model, selected_features

    except Exception as e:
        logger.error("Failed to load model locally: %s", e)
        raise ModelLoadException(
            MODEL_NAME,
            "Failed to load model from both MLflow and local storage"
        ) from e


def load_preprocessing_artifacts() -> dict:
    """
    Загружает артефакты препроцессинга (частоты, OHE-энкодер, карту редких категорий).

    :return: Словарь с ключами:
        - 'frequency_mappings': dict {col: {value: frequency}}
        - 'ohe_encoder': OneHotEncoder или None
        - 'ohe_feature_names': list или []
        - 'rare_categories_map': dict {col: [values]}
    """
    artifacts = {
        'frequency_mappings': {},
        'ohe_encoder': None,
        'ohe_feature_names': [],
        'rare_categories_map': {},
    }

    # Сначала пробуем загрузить локальные файлы (быстрее и надёжнее)
    local_artifacts = {
        'frequency_mappings': ('./frequency_mappings.joblib', True),
        'ohe_encoder': ('./ohe_encoder.joblib', True),
        'ohe_feature_names': ('./ohe_feature_names.joblib', True),
    }

    for key, (path, required) in local_artifacts.items():
        if os.path.exists(path):
            try:
                artifacts[key] = joblib.load(path)
                logger.info("Loaded %s from local file: %s", key, path)

                if key == 'ohe_feature_names' and isinstance(artifacts[key], list):
                    logger.info("OHE feature names count: %d", len(artifacts[key]))
                elif key == 'frequency_mappings' and isinstance(artifacts[key], dict):
                    logger.info("Frequency mappings columns: %s", list(artifacts[key].keys()))

            except Exception as e:
                logger.warning("Failed to load %s from local file: %s", key, e)
        elif required:
            logger.debug("Local artifact not found: %s", path)

    # Пробуем загрузить из MLflow, если локальные не найдены
    if not any(artifacts.values()):
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging", "Production"])

            if versions:
                run_id = versions[0].run_id
                import tempfile

                with tempfile.TemporaryDirectory() as tmpdir:
                    # Загружаем OHE encoder
                    try:
                        local_path = client.download_artifacts(run_id, "ohe_encoder.joblib", tmpdir)
                        artifacts['ohe_encoder'] = joblib.load(local_path)
                        logger.info("OHE encoder loaded from MLflow")
                    except Exception:
                        logger.debug("No OHE encoder found in MLflow artifacts")

                    # Загружаем имена OHE-признаков
                    try:
                        local_path = client.download_artifacts(run_id, "ohe_feature_names.joblib", tmpdir)
                        artifacts['ohe_feature_names'] = joblib.load(local_path)
                        logger.info("OHE feature names loaded from MLflow. Count: %d",
                                    len(artifacts['ohe_feature_names']))
                    except Exception:
                        pass

                    # Загружаем частоты для frequency encoding
                    try:
                        local_path = client.download_artifacts(run_id, "frequency_mappings.joblib", tmpdir)
                        artifacts['frequency_mappings'] = joblib.load(local_path)
                        logger.info("Frequency mappings loaded from MLflow. Columns: %s",
                                    list(artifacts['frequency_mappings'].keys()))
                    except Exception:
                        logger.warning("No frequency mappings found in MLflow")

        except Exception as e:
            logger.warning("Could not load preprocessing artifacts from MLflow: %s", e)

    return artifacts


def check_mlflow_connection() -> bool:
    """
    Проверяет доступность MLflow Tracking Server.

    :return: True если MLflow доступен.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        client.search_experiments(max_results=1)
        logger.info("MLflow connection successful")
        return True
    except Exception as e:
        logger.warning("MLflow connection failed: %s", e)
        return False


def get_model_info(model: Any, selected_features: list) -> dict:
    """
    Возвращает информацию о загруженной модели.

    :param model: Загруженная модель.
    :param selected_features: Список признаков.
    :return: Словарь с информацией.
    """
    info = {
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model is not None else "None",
        "features_count": len(selected_features) if selected_features else 0,
        "features_list": selected_features[:10] if selected_features else [],
    }

    if model is not None:
        try:
            if hasattr(model, 'get_params'):
                params = model.get_params()
                info["model_params_count"] = len(params)
        except Exception:
            pass

    logger.info("Model info: %s", info)
    return info

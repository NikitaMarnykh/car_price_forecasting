"""
Исключения для ML-пайплайна Car Price Prediction.
"""


class CarPriceBaseException(Exception):
    """Базовое исключение для проекта."""
    pass


class DataLoadException(CarPriceBaseException):
    """Ошибка загрузки данных."""
    def __init__(self, data_source: str, message: str = "Failed to load data") -> None:
        self.data_source = data_source
        super().__init__(f"{message} from: '{data_source}'")


class DataPreprocessingException(CarPriceBaseException):
    """Ошибка предобработки данных."""
    def __init__(self, step: str, message: str = "Data preprocessing failed") -> None:
        self.step = step
        super().__init__(f"{message} at step: '{step}'")


class ModelTrainingException(CarPriceBaseException):
    """Ошибка обучения модели."""
    def __init__(self, model_name: str, message: str = "Model training failed") -> None:
        self.model_name = model_name
        super().__init__(f"{message} for model: '{model_name}'")


class ModelEvaluationException(CarPriceBaseException):
    """Ошибка оценки модели."""
    def __init__(self, model_name: str, message: str = "Model evaluation failed") -> None:
        self.model_name = model_name
        super().__init__(f"{message} for model: '{model_name}'")


class MlflowLoggingException(CarPriceBaseException):
    """Ошибка логирования в MLflow."""
    def __init__(self, artifact: str, message: str = "Failed to log to MLflow") -> None:
        self.artifact = artifact
        super().__init__(f"{message} artifact: '{artifact}'")
"""
Исключения для FastAPI сервиса предсказания цен на автомобили.
"""


class PredictionServiceException(Exception):
    """
    Базовое исключение для сервиса предсказаний.
    """
    pass


class ModelLoadException(PredictionServiceException):
    """
    Ошибка при загрузке модели из MLflow.
    """

    def __init__(self, model_name: str, message: str = "Failed to load model") -> None:
        """
        :param model_name: Название модели, которую не удалось загрузить
        :param message: Описание ошибки
        """
        self.model_name = model_name
        super().__init__(f"{message}: '{model_name}'")


class PredictionException(PredictionServiceException):
    """
    Ошибка при выполнении предсказания.
    """

    def __init__(self, step: str, message: str = "Prediction failed") -> None:
        """
        :param step: Шаг, на котором произошла ошибка
        :param message: Описание ошибки
        """
        self.step = step
        super().__init__(f"{message} at step: '{step}'")


class PreprocessingException(PredictionServiceException):
    """
    Ошибка при предобработке данных для предсказания.
    """

    def __init__(self, step: str, message: str = "Preprocessing failed") -> None:
        """
        :param step: Шаг предобработки, на котором произошла ошибка
        :param message: Описание ошибки
        """
        self.step = step
        super().__init__(f"{message} at preprocessing step: '{step}'")


class ValidationException(PredictionServiceException):
    """
    Ошибка при валидации входных данных.
    """

    def __init__(self, field: str, message: str = "Validation failed") -> None:
        """
        :param field: Поле, которое не прошло валидацию
        :param message: Описание ошибки
        """
        self.field = field
        super().__init__(f"{message} for field: '{field}'")


class ConfigurationException(PredictionServiceException):
    """
    Ошибка конфигурации сервиса.
    """

    def __init__(self, config_key: str, message: str = "Configuration error") -> None:
        """
        :param config_key: Ключ конфигурации, вызвавший ошибку
        :param message: Описание ошибки
        """
        self.config_key = config_key
        super().__init__(f"{message} for configuration: '{config_key}'")


class ServiceHealthException(PredictionServiceException):
    """
    Ошибка при проверке работоспособности сервиса.
    """

    def __init__(self, component: str, message: str = "Service health check failed") -> None:
        """
        :param component: Компонент, который не прошёл проверку
        :param message: Описание ошибки
        """
        self.component = component
        super().__init__(f"{message} for component: '{component}'")
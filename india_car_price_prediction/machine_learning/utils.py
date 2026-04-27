import logging

import pandas as pd

from machine_learning.exceptions import DataLoadException, DataPreprocessingException, ModelEvaluationException

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Вспомогательные функции
def load_data(file_path: str, dtype_dict: dict) -> pd.DataFrame:
    """
    Загружает набор данных из CSV-файла с заданными типами колонок.

    :param file_path: Путь к CSV-файлу с данными.
    :param dtype_dict: Словарь с типами данных для каждой колонки.
    :return: Загруженный DataFrame.
    :raise DataLoadException: Если файл не найден, пуст или не может быть прочитан.
    """
    try:
        dataset = pd.read_csv(file_path, dtype=dtype_dict)
        if dataset.empty:
            raise ValueError("Dataset is empty")
        logger.info("Dataset loaded successfully from '%s'. Shape: %s", file_path, dataset.shape)
        return dataset
    except FileNotFoundError as e:
        raise DataLoadException(file_path, "Dataset file not found") from e
    except Exception as e:
        raise DataLoadException(file_path, "An unexpected error occurred while reading the dataset") from e


def group_rare_categories(data: pd.DataFrame, column: str, min_frequency: int = 5) -> pd.DataFrame:
    """
    Объединяет редко встречающиеся категории признака в одну категорию 'Other'.

    :param data: Исходный DataFrame.
    :param column: Название столбца для обработки.
    :param min_frequency: Пороговое значение частоты. Категории с частотой ниже этого порога будут заменены.
    :return: Новый DataFrame с объединенными редкими категориями.
    :raise DataPreprocessingException: Если столбец не найден или произошла ошибка при замене категорий.
    """
    try:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        freq = data[column].value_counts()
        rare_categories = freq[freq < min_frequency].index.tolist()

        if rare_categories:
            # Добавляем категорию 'Other', если её ещё нет
            if 'Other' not in data[column].cat.categories:
                data[column] = data[column].cat.add_categories(['Other'])

            # Создаём маску для редких категорий
            mask = data[column].isin(rare_categories)

            # Заменяем редкие категории на 'Other' (без использования replace)
            data.loc[mask, column] = 'Other'

            # Удаляем неиспользуемые категории для оптимизации памяти
            data[column] = data[column].cat.remove_unused_categories()

            logger.debug(
                "Grouped %d rare categories in column '%s' into 'Other'.",
                len(rare_categories), column
            )

        return data

    except Exception as e:
        raise DataPreprocessingException(
            f"group_rare_categories('{column}')",
            "Failed to group rare categories"
        ) from e


def calculate_adjusted_coefficient_of_determination(
    r2: float,
    number_of_observations: int,
    number_of_features: int
) -> float:
    """
    Вычисляет скорректированный коэффициент детерминации (Adjusted R-squared).

    :param r2: Коэффициент детерминации.
    :param number_of_observations: Количество наблюдений в выборке.
    :param number_of_features: Количество признаков в модели.
    :return: Скорректированный коэффициент детерминации.
    :raise ModelEvaluationException: Если количество признаков превышает количество наблюдений.
    """
    try:
        if number_of_observations <= number_of_features + 1:
            raise ValueError(
                f"Number of observations ({number_of_observations}) must be greater "
                f"than number of features + 1 ({number_of_features + 1})"
            )
        return 1 - (1 - r2) * (number_of_observations - 1) / (number_of_observations - number_of_features - 1)
    except Exception as e:
        raise ModelEvaluationException(
            "AdjustedR2",
            "Failed to calculate adjusted R-squared"
        ) from e

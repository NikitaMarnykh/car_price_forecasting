"""
ML-пайплайн для предсказания стоимости автомобилей с использованием CatBoost.
"""

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Integer

from machine_learning.exceptions import (
    DataLoadException, ModelTrainingException, ModelEvaluationException,
    DataPreprocessingException, MlflowLoggingException, CarPriceBaseException
)

from machine_learning.utils import (
    group_rare_categories, calculate_adjusted_coefficient_of_determination, load_data
)

import os
import logging
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import joblib
from typing import Any

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Настройка MLflow из переменных окружения
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_S3_ENDPOINT_URL'] = MLFLOW_S3_ENDPOINT_URL
os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY


def get_fixed_model_params() -> dict:
    """
    Возвращает фиксированные параметры CatBoost, которые не оптимизируются.

    :return: Словарь с фиксированными параметрами.
    """
    return {
        'random_state': 123,
        'thread_count': -1,
        'verbose': False,
        'allow_writing_files': False,
    }


def create_catboost_param_space() -> dict:
    """
    Создает пространство поиска гиперпараметров для генетического алгоритма.

    :return: Словарь с пространством поиска.
    """
    return {
        'iterations': Integer(100, 1000),
        'learning_rate': Continuous(0.01, 0.3),
        'depth': Integer(1, 10),
        'l2_leaf_reg': Continuous(1, 20),
        'border_count': Integer(1, 255),
    }


def remove_outliers_iqr(data: pd.DataFrame, target_column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Удаляет выбросы из набора данных на основе метода межквартильного размаха (IQR).

    :param data: Исходный DataFrame.
    :param target_column: Название целевой переменной для фильтрации выбросов.
    :param multiplier: Множитель для определения границ (по умолчанию 1.5).
    :return: DataFrame без выбросов.
    :raise DataPreprocessingException: Если целевая колонка не найдена или произошла ошибка при фильтрации.
    """
    try:
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        percentile_25_value = data[target_column].quantile(0.25)
        percentile_75_value = data[target_column].quantile(0.75)
        iqr = percentile_75_value - percentile_25_value

        normal_mask = (
            (data[target_column] >= percentile_25_value - multiplier * iqr) &
            (data[target_column] <= percentile_75_value + multiplier * iqr)
        )

        filtered_data = data[normal_mask].reset_index(drop=True)
        removed_count = len(data) - len(filtered_data)
        logger.info(
            "Outlier removal completed. Removed %d rows (%.1f%% of original data).",
            removed_count, 100 * removed_count / len(data)
        )

        return filtered_data
    except Exception as e:
        raise DataPreprocessingException(
            "remove_outliers_iqr",
            "Failed to remove outliers using IQR method"
        ) from e


def apply_frequency_encoding(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Применяет частотное кодирование к заданным категориальным колонкам.
    Частоты вычисляются исключительно на обучающей выборке.

    :param train_df: Обучающая выборка.
    :param validation_df: Валидационная выборка.
    :param test_df: Тестовая выборка.
    :param columns: Список колонок для частотного кодирования.
    :return: Кортеж (train_df, validation_df, test_df, frequency_mappings).
    :raise DataPreprocessingException: Если колонка не найдена или произошла ошибка при кодировании.
    """
    try:
        frequency_mappings = {}

        for column in columns:
            if column not in train_df.columns:
                raise ValueError(f"Column '{column}' not found in training DataFrame")

            freq_encoding = train_df[column].value_counts(normalize=True)
            frequency_mappings[column] = freq_encoding.to_dict()

            train_df[f'{column}_encoded'] = train_df[column].map(freq_encoding).astype(float)
            validation_df[f'{column}_encoded'] = validation_df[column].map(freq_encoding).astype(float)
            test_df[f'{column}_encoded'] = test_df[column].map(freq_encoding).astype(float)

            logger.debug("Applied frequency encoding to column '%s'.", column)

        train_df.drop(columns, axis=1, inplace=True)
        validation_df.drop(columns, axis=1, inplace=True)
        test_df.drop(columns, axis=1, inplace=True)

        # Сохраняем frequency mappings
        joblib.dump(frequency_mappings, './frequency_mappings.joblib')
        logger.info("Frequency mappings saved to './frequency_mappings.joblib'")

        return train_df, validation_df, test_df, frequency_mappings
    except Exception as e:
        raise DataPreprocessingException(
            "apply_frequency_encoding",
            "Failed to apply frequency encoding"
        ) from e


def apply_one_hot_encoding(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list, OneHotEncoder | None]:
    """
    Применяет One-Hot Encoding к оставшимся категориальным признакам.

    :param train_df: Обучающая выборка.
    :param validation_df: Валидационная выборка.
    :param test_df: Тестовая выборка.
    :return: Кортеж (train_df, validation_df, test_df, category_columns, ohe_feature_names, ohe_encoder).
    :raise DataPreprocessingException: Если произошла ошибка при OHE-кодировании.
    """
    try:
        category_columns = train_df.select_dtypes(include=['category']).columns.tolist()

        if not category_columns:
            logger.info("No categorical columns left for One-Hot Encoding.")
            return train_df, validation_df, test_df, [], [], None

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')

        encoded_train = encoder.fit_transform(train_df[category_columns])
        encoded_validation = encoder.transform(validation_df[category_columns])
        encoded_test = encoder.transform(test_df[category_columns])

        feature_names = encoder.get_feature_names_out(category_columns)

        encoded_train_df = pd.DataFrame(encoded_train, columns=feature_names, index=train_df.index)
        encoded_validation_df = pd.DataFrame(encoded_validation, columns=feature_names, index=validation_df.index)
        encoded_test_df = pd.DataFrame(encoded_test, columns=feature_names, index=test_df.index)

        train_df = train_df.drop(category_columns, axis=1)
        validation_df = validation_df.drop(category_columns, axis=1)
        test_df = test_df.drop(category_columns, axis=1)

        train_df = pd.concat([train_df, encoded_train_df], axis=1)
        validation_df = pd.concat([validation_df, encoded_validation_df], axis=1)
        test_df = pd.concat([test_df, encoded_test_df], axis=1)

        # Сохраняем OHE encoder и имена признаков
        joblib.dump(encoder, './ohe_encoder.joblib')
        joblib.dump(feature_names.tolist(), './ohe_feature_names.joblib')
        logger.info("OHE encoder and feature names saved.")

        logger.info(
            "Applied One-Hot Encoding to %d columns, resulting in %d features.",
            len(category_columns), len(feature_names)
        )

        return train_df, validation_df, test_df, category_columns, feature_names.tolist(), encoder
    except Exception as e:
        raise DataPreprocessingException(
            "apply_one_hot_encoding",
            "Failed to apply One-Hot Encoding"
        ) from e


def optimize_hyperparameters(
    factors_train: pd.DataFrame,
    targets_train: pd.Series,
    ga_params: dict | None = None
) -> dict:
    """
    Оптимизирует гиперпараметры CatBoostRegressor с помощью генетического алгоритма.
    Возвращает ТОЛЬКО словарь с лучшими гиперпараметрами.

    :param factors_train: Матрица признаков для обучения (ВСЕ признаки).
    :param targets_train: Вектор целевой переменной.
    :param ga_params: Словарь с параметрами генетического алгоритма.
    :return: Словарь с оптимизированными гиперпараметрами.
    :raise ModelTrainingException: Если оптимизация завершилась с ошибкой.
    """
    try:
        if ga_params is None:
            ga_params = {
                'cv': min(5, len(targets_train) // 10),
                'scoring': 'neg_mean_squared_error',
                'population_size': 15,
                'n_generations': 10,
                'tournament_size': 5,
                'mutation_rate': 0.2,
                'crossover_rate': 0.8,
                'n_jobs': -1,
                'verbose': True,
            }

        cv = ga_params.get('cv', 5)
        scoring = ga_params.get('scoring', 'neg_mean_squared_error')
        population_size = ga_params.get('population_size', 20)
        n_generations = ga_params.get('n_generations', 15)
        tournament_size = ga_params.get('tournament_size', 3)
        mutation_rate = ga_params.get('mutation_rate', 0.1)
        crossover_rate = ga_params.get('crossover_rate', 0.9)
        n_jobs = ga_params.get('n_jobs', -1)
        verbose = ga_params.get('verbose', True)

        base_model = CatBoostRegressor(**get_fixed_model_params())
        param_grid = create_catboost_param_space()

        logger.info(
            "Starting hyperparameter optimization via GA. "
            "Population: %d, Generations: %d, CV folds: %d",
            population_size, n_generations, cv
        )

        genetic_search = GASearchCV(
            estimator=base_model,
            cv=cv,
            scoring=scoring,
            population_size=population_size,
            generations=n_generations,
            tournament_size=tournament_size,
            mutation_probability=mutation_rate,
            crossover_probability=crossover_rate,
            param_grid=param_grid,
            n_jobs=n_jobs,
            verbose=verbose,
            error_score='raise'
        )

        genetic_search.fit(factors_train, targets_train)

        best_params = genetic_search.best_params_
        best_score = genetic_search.best_score_
        cv_results_df = pd.DataFrame(genetic_search.cv_results_)

        logger.info("Hyperparameter optimization completed.")
        logger.info(
            "Best params: %s",
            {k: round(v, 6) if isinstance(v, float) else v for k, v in best_params.items()}
        )
        logger.info("Best CV score: %.6f", best_score)

        # Логируем эволюцию fitness
        if not cv_results_df.empty and 'mean_test_score' in cv_results_df.columns:
            initial_fitness = cv_results_df['mean_test_score'].iloc[0]
            final_fitness = cv_results_df['mean_test_score'].iloc[-1]
            logger.info(
                "Fitness evolution: Initial = %.6f → Final = %.6f (Δ = %+.6f)",
                initial_fitness, final_fitness, final_fitness - initial_fitness
            )

        # Сохраняем историю
        cv_results_df.to_csv('hyperparameter_optimization_history.csv', index=False)

        return best_params

    except Exception as e:
        raise ModelTrainingException(
            "CatBoostRegressor",
            "Hyperparameter optimization failed"
        ) from e


def select_features(
    factors_train: pd.DataFrame,
    targets_train: pd.Series,
    model_params: dict,
    rfecv_params: dict | None = None
) -> list:
    """
    Отбирает оптимальный набор признаков с помощью RFECV,
    используя переданные (оптимизированные) гиперпараметры.

    :param factors_train: Матрица признаков для обучения (ВСЕ признаки).
    :param targets_train: Вектор целевой переменной.
    :param model_params: ПОЛНЫЙ словарь параметров модели (фиксированные + оптимизированные).
    :param rfecv_params: Словарь с параметрами RFECV.
    :return: Список отобранных признаков.
    :raise ModelTrainingException: Если отбор признаков завершился с ошибкой.
    """
    try:
        n_features = factors_train.shape[1]

        if rfecv_params is None:
            step_size = max(1, n_features // 20)
            cv_folds = min(10, len(targets_train) // 10)
            rfecv_params = {
                'cv': cv_folds,
                'scoring': 'neg_mean_squared_error',
                'step': step_size,
                'min_features_to_select': min(5, n_features),
                'n_jobs': -1,
            }

        # Создаем модель с полными параметрами
        model = CatBoostRegressor(**model_params)

        logger.info(
            "Starting RFECV feature selection. "
            "Total features: %d, Min features to select: %d",
            n_features, rfecv_params['min_features_to_select']
        )

        feature_selector = RFECV(
            estimator=model,
            cv=rfecv_params['cv'],
            scoring=rfecv_params['scoring'],
            step=rfecv_params['step'],
            min_features_to_select=rfecv_params['min_features_to_select'],
            n_jobs=rfecv_params['n_jobs'],
            verbose=False
        )

        feature_selector.fit(factors_train, targets_train)

        selected_features = factors_train.columns[feature_selector.support_].tolist()

        logger.info(
            "Feature selection completed. Selected %d features out of %d.",
            len(selected_features), n_features
        )

        # Сохраняем селектор
        joblib.dump(feature_selector, './rfecv_selector.joblib')

        return selected_features

    except Exception as e:
        raise ModelTrainingException(
            "CatBoostRegressor",
            "Feature selection with RFECV failed"
        ) from e


def train_final_model(
    factors_train: pd.DataFrame,
    targets_train: pd.Series,
    selected_features: list,
    model_params: dict
) -> CatBoostRegressor:
    """
    Обучает финальную модель на отобранных признаках.

    :param factors_train: Полная матрица признаков.
    :param targets_train: Вектор целевой переменной.
    :param selected_features: Список отобранных признаков.
    :param model_params: ПОЛНЫЙ словарь параметров модели.
    :return: Обученная финальная модель.
    :raise ModelTrainingException: Если обучение завершилось с ошибкой.
    """
    try:
        factors_train_selected = factors_train[selected_features]

        logger.info(
            "Training final model on %d features and %d samples",
            len(selected_features), len(targets_train)
        )

        final_model = CatBoostRegressor(**model_params)
        final_model.fit(factors_train_selected, targets_train)

        logger.info("Final model training completed.")

        # Сохраняем модель и список признаков
        joblib.dump(final_model, './ml_model.joblib')
        joblib.dump(selected_features, './selected_features.joblib')

        return final_model

    except Exception as e:
        raise ModelTrainingException(
            "CatBoostRegressor",
            "Final model training failed"
        ) from e


def evaluate_model(
    final_model: CatBoostRegressor,
    factors_train: pd.DataFrame,
    targets_train: pd.Series,
    factors_validation: pd.DataFrame,
    targets_validation: pd.Series,
    factors_test: pd.DataFrame,
    targets_test: pd.Series,
    selected_features: list
) -> tuple[pd.DataFrame, dict]:
    """
    Оценивает качество обученной модели на всех выборках.

    :param final_model: Обученная модель.
    :param factors_train: Признаки тренировочной выборки (отобранные).
    :param targets_train: Целевая переменная тренировочной выборки.
    :param factors_validation: Признаки валидационной выборки (отобранные).
    :param targets_validation: Целевая переменная валидационной выборки.
    :param factors_test: Признаки тестовой выборки (отобранные).
    :param targets_test: Целевая переменная тестовой выборки.
    :param selected_features: Список отобранных признаков.
    :return: Кортеж (DataFrame с метриками, словарь с метриками).
    :raise ModelEvaluationException: Если расчет метрик завершился с ошибкой.
    """
    try:
        quality_metrics = [
            'R2_train', 'R2_validation', 'R2_test',
            'Adj_R2_train', 'Adj_R2_validation', 'Adj_R2_test',
            'MSE_train', 'MSE_validation', 'MSE_test',
            'RMSE_train', 'RMSE_validation', 'RMSE_test',
            'MAE_train', 'MAE_validation', 'MAE_test',
        ]

        models = pd.DataFrame(index=quality_metrics)

        target_train_pred = final_model.predict(factors_train)
        target_validation_pred = final_model.predict(factors_validation)
        target_test_pred = final_model.predict(factors_test)

        # R²
        r2_train = r2_score(targets_train, target_train_pred)
        r2_validation = r2_score(targets_validation, target_validation_pred)
        r2_test = r2_score(targets_test, target_test_pred)

        # Adjusted R²
        n_features = len(selected_features)
        adj_r2_train = calculate_adjusted_coefficient_of_determination(r2_train, len(targets_train), n_features)
        adj_r2_validation = calculate_adjusted_coefficient_of_determination(r2_validation, len(targets_validation), n_features)
        adj_r2_test = calculate_adjusted_coefficient_of_determination(r2_test, len(targets_test), n_features)

        # MSE
        mse_train = mean_squared_error(targets_train, target_train_pred)
        mse_validation = mean_squared_error(targets_validation, target_validation_pred)
        mse_test = mean_squared_error(targets_test, target_test_pred)

        # RMSE
        rmse_train = np.sqrt(mse_train)
        rmse_validation = np.sqrt(mse_validation)
        rmse_test = np.sqrt(mse_test)

        # MAE
        mae_train = mean_absolute_error(targets_train, target_train_pred)
        mae_validation = mean_absolute_error(targets_validation, target_validation_pred)
        mae_test = mean_absolute_error(targets_test, target_test_pred)

        models_list = [
            r2_train, r2_validation, r2_test,
            adj_r2_train, adj_r2_validation, adj_r2_test,
            mse_train, mse_validation, mse_test,
            rmse_train, rmse_validation, rmse_test,
            mae_train, mae_validation, mae_test,
        ]

        models[final_model.__class__.__name__] = models_list
        metrics_dict = dict(zip(quality_metrics, models_list))

        logger.info("Model evaluation completed. Test R²: %.4f, Test RMSE: %.2f", r2_test, rmse_test)

        return models, metrics_dict

    except Exception as e:
        raise ModelEvaluationException(
            final_model.__class__.__name__,
            "Metric calculation failed"
        ) from e


def log_to_mlflow(
    final_model: CatBoostRegressor,
    selected_features: list,
    metrics_dict: dict,
    model_params: dict,
    rfecv_params: dict,
    ohe_feature_names: list,
    frequency_encoded_columns: list,
    dataset_size_info: dict,
    ga_params: dict | None = None,
    ohe_encoder: Any = None,
    frequency_mappings: dict = None,
) -> None:
    """
    Логирует модель, метрики и артефакты в MLflow.
    """
    try:
        # Логируем все параметры
        mlflow.log_params(model_params)
        mlflow.log_params(rfecv_params)
        mlflow.log_params(dataset_size_info)
        mlflow.log_params({
            "frequency_encoded_columns": frequency_encoded_columns,
            "one_hot_encoded_columns": ohe_feature_names,
            "number_of_selected_features": len(selected_features),
            "selected_features": selected_features,
            "model_type": final_model.__class__.__name__,
        })

        if ga_params:
            mlflow.log_params({f"ga_{k}": v for k, v in ga_params.items()})

        # Логируем метрики
        mlflow.log_metrics(metrics_dict)

        # Логируем важность признаков
        feature_importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)

        feature_importance_df.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')

        # Логируем модель
        mlflow.catboost.log_model(
            final_model,
            "model",
            registered_model_name="car_price_model_catboost"
        )

        # Логируем основные артефакты
        for artifact_path in ['./ml_model.joblib', './selected_features.joblib', './rfecv_selector.joblib']:
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)

        # Сохраняем и логируем артефакты препроцессинга
        if ohe_encoder is not None:
            joblib.dump(ohe_encoder, './ohe_encoder.joblib')
            mlflow.log_artifact('./ohe_encoder.joblib')
            logger.info("OHE encoder logged to MLflow")

        if ohe_feature_names:
            joblib.dump(ohe_feature_names, './ohe_feature_names.joblib')
            mlflow.log_artifact('./ohe_feature_names.joblib')
            logger.info("OHE feature names logged to MLflow")

        if frequency_mappings:
            joblib.dump(frequency_mappings, './frequency_mappings.joblib')
            mlflow.log_artifact('./frequency_mappings.joblib')
            logger.info("Frequency mappings logged to MLflow")

        if os.path.exists('hyperparameter_optimization_history.csv'):
            mlflow.log_artifact('hyperparameter_optimization_history.csv')

        logger.info("All artifacts logged to MLflow successfully.")

    except Exception as e:
        raise MlflowLoggingException(
            "model/metrics/artifacts",
            "Failed to log artifacts to MLflow"
        ) from e


# Точка входа
if __name__ == '__main__':
    experiment_name = "car_price_prediction_catboost"

    try:
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow experiment set to '%s'.", experiment_name)
    except Exception as e:
        raise MlflowLoggingException("experiment", "Failed to set MLflow experiment") from e

    with mlflow.start_run(run_name="catboost_full_pipeline"):
        try:

            # 1. ЗАГРУЗКА ДАННЫХ
            logger.info("STEP 1: Loading data")

            dtype_dict = {
                'Make': 'category', 'Model': 'category', 'Price': 'int64',
                'Year': 'int64', 'Kilometer': 'int64', 'Fuel Type': 'category',
                'Transmission': 'category', 'Location': 'category', 'Color': 'category',
                'Owner': 'category', 'Seller Type': 'category', 'Engine': 'float64',
                'Max Power': 'float64', 'Max Torque': 'float64', 'Drivetrain': 'category',
                'Length': 'float64', 'Width': 'float64', 'Height': 'float64',
                'Seating Capacity': 'category', 'Fuel Tank Capacity': 'float64'
            }

            cb_dataset = load_data('machine_learning/data/preprocessed_dataset.csv', dtype_dict)
            initial_dataset_size = len(cb_dataset)

            # 2. ПРЕДОБРАБОТКА ДАННЫХ
            logger.info("STEP 2: Preprocessing")

            target_variable = 'Price'

            # Удаление выбросов
            cb_dataset = remove_outliers_iqr(cb_dataset, target_variable, multiplier=1.5)
            dataset_size_after_outliers = len(cb_dataset)

            # Группировка редких категорий
            rare_category_thresholds = [
                ('Make', 30), ('Model', 2), ('Fuel Type', 125),
                ('Location', 10), ('Color', 50), ('Owner', 150),
                ('Seller Type', 300), ('Drivetrain', 250), ('Seating Capacity', 150),
            ]

            for column, threshold in rare_category_thresholds:
                cb_dataset = group_rare_categories(cb_dataset, column, min_frequency=threshold)

            logger.info("Preprocessing completed.")

            # 3. РАЗДЕЛЕНИЕ ВЫБОРОК
            logger.info("STEP 3: Splitting data")

            target_variables = cb_dataset[target_variable]
            factor_variables = cb_dataset.drop([target_variable], axis=1)

            factors_train, factors_test, targets_train, targets_test = train_test_split(
                factor_variables, target_variables, test_size=0.2, random_state=0
            )

            factors_train, factors_validation, targets_train, targets_validation = train_test_split(
                factors_train, targets_train, test_size=0.25, random_state=42
            )

            logger.info(
                "Split completed. Train: %d, Validation: %d, Test: %d",
                len(factors_train), len(factors_validation), len(factors_test)
            )

            # 4. КОДИРОВАНИЕ ПРИЗНАКОВ
            logger.info("STEP 4: Feature encoding")

            high_cardinality_columns = ['Make', 'Model', 'Location', 'Color']

            factors_train, factors_validation, factors_test, frequency_mappings = apply_frequency_encoding(
                factors_train, factors_validation, factors_test, high_cardinality_columns
            )

            factors_train, factors_validation, factors_test, ohe_columns, ohe_feature_names, ohe_encoder = \
                apply_one_hot_encoding(factors_train, factors_validation, factors_test)

            # 5. ОТБОР ПРИЗНАКОВ на базовых параметрах
            logger.info("STEP 5: Feature selection (RFECV) with default params")

            n_features = factors_train.shape[1]
            step_size = max(1, n_features // 20)
            cv_folds = min(5, len(targets_train) // 10)

            rfecv_params = {
                'cv': cv_folds,
                'scoring': 'neg_mean_absolute_error',
                'step': step_size,
                'min_features_to_select': min(5, n_features),
                'n_jobs': -1,
            }

            # Используем фиксированные параметры (как базовый уровень)
            selected_features = select_features(
                factors_train=factors_train,
                targets_train=targets_train,
                model_params=get_fixed_model_params(),
                rfecv_params=rfecv_params
            )

            # Оставляем только отобранные признаки
            factors_train_selected = factors_train[selected_features]

            # 6. ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ на ОТОБРАННЫХ признаках
            logger.info("STEP 6: Hyperparameter optimization (GA) on selected features")

            ga_params = {
                'cv': min(5, len(targets_train) // 10),
                'scoring': 'neg_mean_absolute_error',
                'population_size': 8,
                'n_generations': 5,
                'tournament_size': 3,
                'mutation_rate': 0.1,
                'crossover_rate': 0.9,
                'n_jobs': -1,
                'verbose': True,
            }

            optimized_hyperparams = optimize_hyperparameters(
                factors_train=factors_train_selected,
                targets_train=targets_train,
                ga_params=ga_params
            )

            full_model_params = {**get_fixed_model_params(), **optimized_hyperparams}

            # 7. ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ (на отобранных признаках)
            logger.info("STEP 7: Training final model on selected features")

            final_model = train_final_model(
                factors_train=factors_train,
                targets_train=targets_train,
                selected_features=selected_features,
                model_params=full_model_params
            )

            # 8. ОЦЕНКА КАЧЕСТВА
            logger.info("STEP 8: Model evaluation")

            factors_train_selected = factors_train[selected_features]
            factors_validation_selected = factors_validation[selected_features]
            factors_test_selected = factors_test[selected_features]

            models_df, metrics_dict = evaluate_model(
                final_model,
                factors_train_selected, targets_train,
                factors_validation_selected, targets_validation,
                factors_test_selected, targets_test,
                selected_features
            )

            # 9. ЛОГИРОВАНИЕ В MLFLOW
            logger.info("STEP 9: Logging to MLflow")

            dataset_size_info = {
                "dataset_path": 'machine_learning/data/preprocessed_dataset.csv',
                "initial_dataset_size": initial_dataset_size,
                "dataset_size_after_outlier_removal": dataset_size_after_outliers,
                "outlier_filter_method": "IQR",
                "iqr_multiplier": 1.5,
                "train_set_size": len(targets_train),
                "validation_set_size": len(targets_validation),
                "test_set_size": len(targets_test),
            }

            log_to_mlflow(
                final_model=final_model,
                selected_features=selected_features,
                metrics_dict=metrics_dict,
                model_params=full_model_params,
                rfecv_params=rfecv_params,
                ohe_feature_names=ohe_feature_names,
                frequency_encoded_columns=high_cardinality_columns,
                dataset_size_info=dataset_size_info,
                ga_params=ga_params,
                ohe_encoder=ohe_encoder,
                frequency_mappings=frequency_mappings,
            )

            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")

        except DataLoadException:
            logger.exception("Failed to load data. Pipeline aborted.")
            raise
        except DataPreprocessingException:
            logger.exception("Failed during data preprocessing. Pipeline aborted.")
            raise
        except ModelTrainingException:
            logger.exception("Failed during model training/optimization. Pipeline aborted.")
            raise
        except ModelEvaluationException:
            logger.exception("Failed during model evaluation. Pipeline aborted.")
            raise
        except MlflowLoggingException:
            logger.exception("Failed to log to MLflow. Pipeline aborted.")
            raise
        except CarPriceBaseException:
            logger.exception("An unexpected pipeline error occurred.")
            raise
        except Exception:
            logger.exception("An unhandled error occurred. Pipeline aborted.")
            raise
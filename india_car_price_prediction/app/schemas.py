"""
Pydantic схемы для Car Price Prediction API
"""

from pydantic import BaseModel, Field


class CarFeatures(BaseModel):
    model_config = {"protected_namespaces": ()}

    Make: str = Field(..., description="Марка автомобиля", example="Honda")
    Model: str = Field(..., description="Модель автомобиля", example="City")
    Year: int = Field(..., ge=1980, le=2024, description="Год выпуска", example=2018)
    Kilometer: int = Field(..., ge=0, description="Пробег в километрах", example=45000)
    Fuel_Type: str = Field(..., alias="Fuel Type", description="Тип топлива", example="Petrol")
    Transmission: str = Field(..., description="Тип трансмиссии", example="Manual")
    Location: str = Field(..., description="Местоположение", example="Mumbai")
    Color: str = Field(..., description="Цвет автомобиля", example="White")
    Owner: str = Field(..., description="Тип владельца", example="First")
    Seller_Type: str = Field(..., alias="Seller Type", description="Тип продавца", example="Individual")
    Engine: float = Field(..., ge=0, description="Объем двигателя в CC", example=1498.0)
    Max_Power: float = Field(..., ge=0, alias="Max Power", description="Максимальная мощность в BHP", example=98.6)
    Max_Torque: float = Field(..., ge=0, alias="Max Torque", description="Максимальный крутящий момент в Nm",
                              example=200.0)
    Drivetrain: str = Field(..., description="Тип привода", example="FWD")
    Length: float = Field(..., ge=0, description="Длина в мм", example=3995.0)
    Width: float = Field(..., ge=0, description="Ширина в мм", example=1695.0)
    Height: float = Field(..., ge=0, description="Высота в мм", example=1505.0)
    Seating_Capacity: str = Field(..., alias="Seating Capacity", description="Количество мест", example="5")
    Fuel_Tank_Capacity: float = Field(..., ge=0, alias="Fuel Tank Capacity",
                                      description="Объем топливного бака в литрах", example=40.0)


class PredictionResponse(BaseModel):
    """Ответ с предсказанной ценой"""

    predicted_price: float = Field(..., description="Предсказанная цена в индийских рупиях")
    currency: str = Field(default="INR", description="Валюта")
    model: str | None = Field(None, description="Используемая модель")
    processing_time: float = Field(..., description="Время обработки запроса в секундах")


class HealthResponse(BaseModel):
    """Статус сервиса"""

    status: str = Field(..., description="Статус сервиса")
    model_loaded: bool = Field(..., description="Загружена ли модель")
    mlflow_connected: bool = Field(..., description="Доступен ли MLFlow")
    model: str | None = Field(None, description="Используемая модель")
    features_count: int | None = Field(None, description="Количество признаков модели")


class ErrorResponse(BaseModel):
    """
    Ответ при возникновении ошибки.
    """
    error: str = Field(..., description="Тип ошибки")
    detail: str = Field(..., description="Детали ошибки")
    timestamp: str | None = Field(None, description="Временная метка ошибки")

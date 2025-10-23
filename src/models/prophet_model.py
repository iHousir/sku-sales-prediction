"""
Prophet模型实现
Facebook开源的时间序列预测工具
"""

import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from src.models.base_model import BaseModel
from src.config import PROPHET_CONFIG


class ProphetModel(BaseModel):
    """Prophet时间序列预测模型"""

    def __init__(self, config: Dict = None):
        merged_config = {**PROPHET_CONFIG, **(config or {})}
        super().__init__(name="Prophet", config=merged_config)

    def fit(self, train_data: pd.DataFrame) -> 'ProphetModel':
        """训练Prophet模型"""
        from prophet import Prophet

        # Prophet需要特定的列名格式
        prophet_df = pd.DataFrame({
            'ds': train_data['date'],
            'y': train_data['quantity']
        })

        try:
            # 创建模型
            self.model = Prophet(
                growth=self.config.get('growth', 'linear'),
                changepoint_prior_scale=self.config.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=self.config.get('seasonality_prior_scale', 10),
                seasonality_mode=self.config.get('seasonality_mode', 'multiplicative'),
                daily_seasonality=self.config.get('daily_seasonality', True),
                weekly_seasonality=self.config.get('weekly_seasonality', True),
                yearly_seasonality=self.config.get('yearly_seasonality', True)
            )

            # 训练模型
            self.model.fit(prophet_df)
            self.is_fitted = True

            logger.info("Prophet 模型训练完成")

        except Exception as e:
            logger.error(f"Prophet 训练失败: {str(e)}")
            raise

        return self

    def predict(self, steps: int) -> np.ndarray:
        """预测未来值"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")

        try:
            # 创建未来日期框架
            future = self.model.make_future_dataframe(periods=steps)

            # 预测
            forecast = self.model.predict(future)

            # 提取预测值（只取未来的部分）
            predictions = forecast['yhat'].iloc[-steps:].values

            # 确保预测值非负
            predictions = np.maximum(predictions, 0)

            return predictions

        except Exception as e:
            logger.error(f"Prophet 预测失败: {str(e)}")
            raise

    def get_forecast_components(self, steps: int) -> pd.DataFrame:
        """
        获取预测组件（趋势、季节性等）

        Args:
            steps: 预测步数

        Returns:
            包含预测组件的DataFrame
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")

        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)

        return forecast.iloc[-steps:][['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]

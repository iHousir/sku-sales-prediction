"""
ARIMA模型实现
包括标准ARIMA和Auto-ARIMA
"""

import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from src.models.base_model import BaseModel
from src.config import ARIMA_CONFIG


class ARIMAModel(BaseModel):
    """ARIMA时间序列预测模型"""

    def __init__(self, config: Dict = None):
        super().__init__(name="ARIMA", config=config or {})
        self.order = self.config.get('order', (1, 1, 1))

    def fit(self, train_data: pd.DataFrame) -> 'ARIMAModel':
        """训练ARIMA模型"""
        from statsmodels.tsa.arima.model import ARIMA

        y = train_data['quantity'].values

        try:
            # 创建并训练模型
            self.model = ARIMA(y, order=self.order)
            self.model = self.model.fit()
            self.is_fitted = True

            logger.info(f"ARIMA{self.order} 模型训练完成")

        except Exception as e:
            logger.error(f"ARIMA 训练失败: {str(e)}")
            raise

        return self

    def predict(self, steps: int) -> np.ndarray:
        """预测未来值"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")

        try:
            # 预测
            forecast = self.model.forecast(steps=steps)

            # 确保预测值非负
            forecast = np.maximum(forecast, 0)

            return forecast

        except Exception as e:
            logger.error(f"ARIMA 预测失败: {str(e)}")
            raise


class AutoARIMAModel(BaseModel):
    """Auto-ARIMA模型 - 自动选择最优参数"""

    def __init__(self, config: Dict = None):
        merged_config = {**ARIMA_CONFIG, **(config or {})}
        super().__init__(name="Auto-ARIMA", config=merged_config)

    def fit(self, train_data: pd.DataFrame) -> 'AutoARIMAModel':
        """训练Auto-ARIMA模型"""
        from pmdarima import auto_arima

        y = train_data['quantity'].values

        try:
            # 自动选择最优ARIMA参数
            self.model = auto_arima(
                y,
                start_p=0,
                start_q=0,
                max_p=self.config.get('max_p', 5),
                max_q=self.config.get('max_q', 5),
                d=None,  # 自动确定差分阶数
                max_d=self.config.get('max_d', 2),
                seasonal=self.config.get('seasonal', True),
                m=self.config.get('m', 7),
                stepwise=self.config.get('stepwise', True),
                suppress_warnings=self.config.get('suppress_warnings', True),
                error_action=self.config.get('error_action', 'ignore'),
                trace=False
            )

            self.is_fitted = True
            logger.info(f"Auto-ARIMA 模型训练完成: {self.model.order}")

        except Exception as e:
            logger.error(f"Auto-ARIMA 训练失败: {str(e)}")
            raise

        return self

    def predict(self, steps: int) -> np.ndarray:
        """预测未来值"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")

        try:
            # 预测
            forecast = self.model.predict(n_periods=steps)

            # 确保预测值非负
            forecast = np.maximum(forecast, 0)

            return forecast

        except Exception as e:
            logger.error(f"Auto-ARIMA 预测失败: {str(e)}")
            raise

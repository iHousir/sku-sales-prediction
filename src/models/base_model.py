"""
基础模型类
所有预测模型的抽象基类
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger


class BaseModel(ABC):
    """预测模型基类"""

    def __init__(self, name: str, config: Dict = None):
        """
        初始化模型

        Args:
            name: 模型名称
            config: 模型配置参数
        """
        self.name = name
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.train_history = {}

    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> 'BaseModel':
        """
        训练模型

        Args:
            train_data: 训练数据，包含'date'和'quantity'列

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """
        预测未来值

        Args:
            steps: 预测步数

        Returns:
            预测值数组
        """
        pass

    def forecast(self, train_data: pd.DataFrame, steps: int) -> Dict:
        """
        完整的预测流程：训练 + 预测

        Args:
            train_data: 训练数据
            steps: 预测步数

        Returns:
            包含预测结果的字典
        """
        try:
            # 训练模型
            self.fit(train_data)

            # 预测
            predictions = self.predict(steps)

            # 生成预测日期
            last_date = train_data['date'].max()
            freq = pd.infer_freq(train_data['date'])
            if freq is None:
                freq = 'D'  # 默认为天

            forecast_dates = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq=freq
            )[1:]  # 跳过第一个日期（训练集最后一天）

            result = {
                'model_name': self.name,
                'dates': forecast_dates,
                'predictions': predictions,
                'success': True,
                'error': None
            }

            logger.info(f"{self.name} 预测完成，预测 {steps} 步")
            return result

        except Exception as e:
            logger.error(f"{self.name} 预测失败: {str(e)}")
            return {
                'model_name': self.name,
                'dates': None,
                'predictions': None,
                'success': False,
                'error': str(e)
            }

    def evaluate(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            train_data: 训练数据
            test_data: 测试数据

        Returns:
            评估指标字典
        """
        from src.utils import calculate_metrics

        try:
            # 训练模型
            self.fit(train_data)

            # 预测
            steps = len(test_data)
            predictions = self.predict(steps)

            # 计算指标
            y_true = test_data['quantity'].values
            metrics = calculate_metrics(y_true, predictions)

            logger.info(f"{self.name} 评估完成: MAPE={metrics['mape']:.2f}%")
            return metrics

        except Exception as e:
            logger.error(f"{self.name} 评估失败: {str(e)}")
            return {
                'mae': np.inf,
                'rmse': np.inf,
                'mape': np.inf,
                'smape': np.inf
            }

    def get_params(self) -> Dict:
        """获取模型参数"""
        return self.config.copy()

    def set_params(self, **params) -> 'BaseModel':
        """设置模型参数"""
        self.config.update(params)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

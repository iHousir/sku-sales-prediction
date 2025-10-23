"""
LSTM模型实现
基于深度学习的时间序列预测
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from src.models.base_model import BaseModel
from src.config import LSTM_CONFIG


class LSTMModel(BaseModel):
    """LSTM深度学习预测模型"""

    def __init__(self, config: Dict = None):
        merged_config = {**LSTM_CONFIG, **(config or {})}
        super().__init__(name="LSTM", config=merged_config)
        self.scaler = None
        self.lookback = self.config.get('lookback_window', 30)

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建LSTM训练序列

        Args:
            data: 原始数据

        Returns:
            X, y
        """
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback])

        return np.array(X), np.array(y)

    def fit(self, train_data: pd.DataFrame) -> 'LSTMModel':
        """训练LSTM模型"""
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import MinMaxScaler

        y = train_data['quantity'].values.reshape(-1, 1)

        try:
            # 数据标准化
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            y_scaled = self.scaler.fit_transform(y)

            # 创建序列
            X, y_train = self._create_sequences(y_scaled)

            if len(X) < 10:
                raise ValueError(f"训练样本太少: {len(X)}，需要至少 {self.lookback + 10} 个样本")

            # Reshape for LSTM [samples, time steps, features]
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # 构建LSTM模型
            lstm_units = self.config.get('lstm_units', [64, 32])
            dropout = self.config.get('dropout_rate', 0.2)

            model = keras.Sequential()

            # 第一层LSTM
            model.add(keras.layers.LSTM(
                lstm_units[0],
                return_sequences=len(lstm_units) > 1,
                input_shape=(self.lookback, 1)
            ))
            model.add(keras.layers.Dropout(dropout))

            # 额外的LSTM层
            for i in range(1, len(lstm_units)):
                return_seq = i < len(lstm_units) - 1
                model.add(keras.layers.LSTM(lstm_units[i], return_sequences=return_seq))
                model.add(keras.layers.Dropout(dropout))

            # 输出层
            model.add(keras.layers.Dense(1))

            # 编译模型
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.001)),
                loss='mse'
            )

            # 早停
            early_stop = keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=self.config.get('early_stopping_patience', 10),
                restore_best_weights=True
            )

            # 训练模型
            self.train_history = model.fit(
                X, y_train,
                epochs=self.config.get('epochs', 50),
                batch_size=self.config.get('batch_size', 32),
                validation_split=self.config.get('validation_split', 0.2),
                callbacks=[early_stop],
                verbose=0
            )

            self.model = model
            self.is_fitted = True
            self.last_sequence = y_scaled[-self.lookback:]

            logger.info("LSTM 模型训练完成")

        except Exception as e:
            logger.error(f"LSTM 训练失败: {str(e)}")
            raise

        return self

    def predict(self, steps: int) -> np.ndarray:
        """预测未来值"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")

        try:
            predictions = []
            current_sequence = self.last_sequence.copy()

            # 逐步预测
            for _ in range(steps):
                # Reshape for prediction
                X_pred = current_sequence.reshape(1, self.lookback, 1)

                # 预测下一个值
                next_pred = self.model.predict(X_pred, verbose=0)[0, 0]
                predictions.append(next_pred)

                # 更新序列
                current_sequence = np.append(current_sequence[1:], next_pred)

            # 反标准化
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()

            # 确保预测值非负
            predictions = np.maximum(predictions, 0)

            return predictions

        except Exception as e:
            logger.error(f"LSTM 预测失败: {str(e)}")
            raise

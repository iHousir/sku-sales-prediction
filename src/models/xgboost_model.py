"""
XGBoost和LightGBM模型实现
基于梯度提升的时间序列预测
"""

import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from src.models.base_model import BaseModel
from src.config import XGBOOST_CONFIG, LIGHTGBM_CONFIG
from src.utils import create_lag_features, create_rolling_features, create_time_features


class XGBoostModel(BaseModel):
    """XGBoost预测模型"""

    def __init__(self, config: Dict = None):
        merged_config = {**XGBOOST_CONFIG, **(config or {})}
        super().__init__(name="XGBoost", config=merged_config)
        self.lookback = self.config.get('lookback_window', 30)
        self.feature_cols = None

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建特征"""
        df = df.copy()

        # 滞后特征
        lags = [1, 7, 14, 21, 28]
        df = create_lag_features(df, 'quantity', lags)

        # 滚动窗口特征
        windows = [7, 14, 28]
        df = create_rolling_features(df, 'quantity', windows)

        # 时间特征
        df = create_time_features(df, 'date')

        # 移除包含NaN的行
        df = df.dropna()

        return df

    def fit(self, train_data: pd.DataFrame) -> 'XGBoostModel':
        """训练XGBoost模型"""
        import xgboost as xgb

        # 创建特征
        train_df = self._create_features(train_data)

        if len(train_df) < 10:
            raise ValueError("特征工程后训练样本太少")

        try:
            # 准备训练数据
            X = train_df.drop(['date', 'quantity'], axis=1)
            y = train_df['quantity']

            self.feature_cols = X.columns.tolist()

            # 创建并训练模型
            self.model = xgb.XGBRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                objective=self.config.get('objective', 'reg:squarederror'),
                n_jobs=self.config.get('n_jobs', -1),
                random_state=42
            )

            self.model.fit(X, y)
            self.is_fitted = True

            # 保存最后的数据用于预测
            self.last_data = train_data.copy()

            logger.info("XGBoost 模型训练完成")

        except Exception as e:
            logger.error(f"XGBoost 训练失败: {str(e)}")
            raise

        return self

    def predict(self, steps: int) -> np.ndarray:
        """预测未来值"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")

        try:
            predictions = []
            current_data = self.last_data.copy()

            # 逐步预测
            for i in range(steps):
                # 创建特征
                feature_df = self._create_features(current_data)

                if len(feature_df) == 0:
                    # 如果特征工程后没有数据，使用最后的平均值
                    pred = current_data['quantity'].tail(7).mean()
                else:
                    # 使用最后一行的特征进行预测
                    X = feature_df[self.feature_cols].iloc[-1:]
                    pred = self.model.predict(X)[0]

                predictions.append(max(pred, 0))  # 确保非负

                # 更新数据
                last_date = current_data['date'].max()
                freq = pd.infer_freq(current_data['date'])
                if freq is None:
                    freq = 'D'

                next_date = last_date + pd.Timedelta(1, unit=freq[0])

                new_row = pd.DataFrame({
                    'date': [next_date],
                    'quantity': [pred]
                })

                current_data = pd.concat([current_data, new_row], ignore_index=True)

            return np.array(predictions)

        except Exception as e:
            logger.error(f"XGBoost 预测失败: {str(e)}")
            raise


class LightGBMModel(BaseModel):
    """LightGBM预测模型"""

    def __init__(self, config: Dict = None):
        merged_config = {**LIGHTGBM_CONFIG, **(config or {})}
        super().__init__(name="LightGBM", config=merged_config)
        self.lookback = self.config.get('lookback_window', 30)
        self.feature_cols = None

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建特征"""
        df = df.copy()

        # 滞后特征
        lags = [1, 7, 14, 21, 28]
        df = create_lag_features(df, 'quantity', lags)

        # 滚动窗口特征
        windows = [7, 14, 28]
        df = create_rolling_features(df, 'quantity', windows)

        # 时间特征
        df = create_time_features(df, 'date')

        # 移除包含NaN的行
        df = df.dropna()

        return df

    def fit(self, train_data: pd.DataFrame) -> 'LightGBMModel':
        """训练LightGBM模型"""
        import lightgbm as lgb

        # 创建特征
        train_df = self._create_features(train_data)

        if len(train_df) < 10:
            raise ValueError("特征工程后训练样本太少")

        try:
            # 准备训练数据
            X = train_df.drop(['date', 'quantity'], axis=1)
            y = train_df['quantity']

            self.feature_cols = X.columns.tolist()

            # 创建并训练模型
            self.model = lgb.LGBMRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                objective=self.config.get('objective', 'regression'),
                n_jobs=self.config.get('n_jobs', -1),
                random_state=42,
                verbose=-1
            )

            self.model.fit(X, y)
            self.is_fitted = True

            # 保存最后的数据用于预测
            self.last_data = train_data.copy()

            logger.info("LightGBM 模型训练完成")

        except Exception as e:
            logger.error(f"LightGBM 训练失败: {str(e)}")
            raise

        return self

    def predict(self, steps: int) -> np.ndarray:
        """预测未来值"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")

        try:
            predictions = []
            current_data = self.last_data.copy()

            # 逐步预测
            for i in range(steps):
                # 创建特征
                feature_df = self._create_features(current_data)

                if len(feature_df) == 0:
                    # 如果特征工程后没有数据，使用最后的平均值
                    pred = current_data['quantity'].tail(7).mean()
                else:
                    # 使用最后一行的特征进行预测
                    X = feature_df[self.feature_cols].iloc[-1:]
                    pred = self.model.predict(X)[0]

                predictions.append(max(pred, 0))  # 确保非负

                # 更新数据
                last_date = current_data['date'].max()
                freq = pd.infer_freq(current_data['date'])
                if freq is None:
                    freq = 'D'

                next_date = last_date + pd.Timedelta(1, unit=freq[0])

                new_row = pd.DataFrame({
                    'date': [next_date],
                    'quantity': [pred]
                })

                current_data = pd.concat([current_data, new_row], ignore_index=True)

            return np.array(predictions)

        except Exception as e:
            logger.error(f"LightGBM 预测失败: {str(e)}")
            raise

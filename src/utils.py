"""
工具函数模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算预测评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        包含各项指标的字典
    """
    # 过滤无效值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "mae": np.inf,
            "rmse": np.inf,
            "mape": np.inf,
            "smape": np.inf
        }

    # MAE - 平均绝对误差
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE - 均方根误差
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE - 平均绝对百分比误差
    # 避免除以0
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # SMAPE - 对称平均绝对百分比误差
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "smape": float(smape)
    }


def detect_outliers(series: pd.Series, method: str = "iqr", threshold: float = 3.0) -> pd.Series:
    """
    检测异常值

    Args:
        series: 时间序列数据
        method: 检测方法 ('iqr' 或 'zscore')
        threshold: 阈值

    Returns:
        布尔序列，True表示异常值
    """
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)

    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold

    else:
        raise ValueError(f"Unknown method: {method}")


def fill_missing_dates(df: pd.DataFrame, date_col: str, freq: str = "D") -> pd.DataFrame:
    """
    填补缺失的日期

    Args:
        df: 数据框
        date_col: 日期列名
        freq: 频率 ('D', 'W', 'M')

    Returns:
        填补后的数据框
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # 创建完整的日期范围
    date_range = pd.date_range(
        start=df[date_col].min(),
        end=df[date_col].max(),
        freq=freq
    )

    # 重新索引
    df = df.set_index(date_col)
    df = df.reindex(date_range, fill_value=0)
    df.index.name = date_col
    df = df.reset_index()

    return df


def create_lag_features(df: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
    """
    创建滞后特征

    Args:
        df: 数据框
        target_col: 目标列名
        lags: 滞后期列表

    Returns:
        包含滞后特征的数据框
    """
    df = df.copy()

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df


def create_rolling_features(df: pd.DataFrame, target_col: str, windows: List[int]) -> pd.DataFrame:
    """
    创建滚动窗口特征

    Args:
        df: 数据框
        target_col: 目标列名
        windows: 窗口大小列表

    Returns:
        包含滚动特征的数据框
    """
    df = df.copy()

    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = df[target_col].rolling(window=window).mean()
        df[f"{target_col}_rolling_std_{window}"] = df[target_col].rolling(window=window).std()
        df[f"{target_col}_rolling_min_{window}"] = df[target_col].rolling(window=window).min()
        df[f"{target_col}_rolling_max_{window}"] = df[target_col].rolling(window=window).max()

    return df


def create_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    创建时间特征

    Args:
        df: 数据框
        date_col: 日期列名

    Returns:
        包含时间特征的数据框
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['quarter'] = df[date_col].dt.quarter
    df['weekofyear'] = df[date_col].dt.isocalendar().week
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)

    return df


def split_train_test(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序拆分训练集和测试集

    Args:
        df: 数据框
        test_size: 测试集比例

    Returns:
        训练集和测试集
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def check_stationarity(series: pd.Series) -> Dict[str, any]:
    """
    检查时间序列的平稳性 (使用ADF检验)

    Args:
        series: 时间序列数据

    Returns:
        包含检验结果的字典
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series.dropna())

    return {
        "is_stationary": result[1] < 0.05,  # p-value < 0.05 表示平稳
        "adf_statistic": result[0],
        "p_value": result[1],
        "critical_values": result[4]
    }


def make_stationary(series: pd.Series, method: str = "diff") -> pd.Series:
    """
    将时间序列转换为平稳序列

    Args:
        series: 时间序列数据
        method: 转换方法 ('diff' 或 'log_diff')

    Returns:
        平稳化后的序列
    """
    if method == "diff":
        return series.diff().dropna()
    elif method == "log_diff":
        return np.log(series + 1).diff().dropna()
    else:
        raise ValueError(f"Unknown method: {method}")

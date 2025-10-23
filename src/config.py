"""
配置文件 - 定义所有预测相关的参数
"""

import os
from typing import Dict, List

# 项目路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_CACHE_DIR = os.path.join(OUTPUT_DIR, "models")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 数据字段配置
# 必需字段
REQUIRED_COLUMNS = {
    "date": "下单时间",  # 日期字段
    "quantity": "数量",  # 销量字段（可以为负数，表示退货）
    "store_id": "送货专卖店卡号",  # 店铺ID
    "product_code": "货品代码",  # 产品代码（作为产品唯一标识）
    "province": "省",  # 省份
    "city": "市",  # 城市
}

# 可选字段（如果数据中不存在这些字段，系统会自动跳过）
OPTIONAL_COLUMNS = {
    "product_name": "货品名称",  # 产品名称（可选）
    "delivery_address": "送货地址",  # 送货地址（可选）
    "delivery_method": "配送方式",  # 配送方式（可选）
    "month": "月份"  # 月份（可选）
}

# 所有字段的合并（向后兼容）
DATA_COLUMNS = {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}

# 产品标识字段 - 用于聚合和识别产品
PRODUCT_ID_FIELD = "product_code"  # 使用货品代码作为产品唯一标识

# 时间粒度配置
TIME_GRANULARITIES = {
    "daily": "D",  # 天
    "weekly": "W",  # 周
    "monthly": "M"  # 月
}

# 聚合维度配置
AGGREGATION_DIMENSIONS = ["store_id", "province", "city"]

# 预测算法配置
AVAILABLE_MODELS = [
    "arima",
    "auto_arima",
    "prophet",
    "lstm",
    "xgboost",
    "lightgbm"
]

# 模型评估指标
EVALUATION_METRICS = ["mae", "rmse", "mape", "smape"]
PRIMARY_METRIC = "mape"  # 主要评估指标

# 默认预测参数
DEFAULT_FORECAST_CONFIG = {
    "test_size": 0.2,  # 测试集比例
    "validation_size": 0.1,  # 验证集比例
    "min_train_samples": 30,  # 最小训练样本数
    "forecast_horizon": {
        "daily": 30,  # 预测未来30天
        "weekly": 12,  # 预测未来12周
        "monthly": 6  # 预测未来6个月
    },
    "confidence_interval": 0.95  # 置信区间
}

# ARIMA模型参数
ARIMA_CONFIG = {
    "max_p": 5,
    "max_d": 2,
    "max_q": 5,
    "seasonal": True,
    "m": 7,  # 周季节性
    "stepwise": True,
    "suppress_warnings": True,
    "error_action": "ignore"
}

# Prophet模型参数
PROPHET_CONFIG = {
    "growth": "linear",
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10,
    "seasonality_mode": "multiplicative",
    "daily_seasonality": True,
    "weekly_seasonality": True,
    "yearly_seasonality": True
}

# LSTM模型参数
LSTM_CONFIG = {
    "lookback_window": 30,  # 回看窗口
    "lstm_units": [64, 32],  # LSTM层单元数
    "dropout_rate": 0.2,
    "batch_size": 32,
    "epochs": 50,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "learning_rate": 0.001
}

# XGBoost模型参数
XGBOOST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "lookback_window": 30  # 特征工程窗口
}

# LightGBM模型参数
LIGHTGBM_CONFIG = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "regression",
    "n_jobs": -1,
    "lookback_window": 30
}

# 并行处理配置
PARALLEL_CONFIG = {
    "n_jobs": -1,  # 使用所有CPU核心
    "verbose": 1
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    "rotation": "10 MB",
    "retention": "30 days"
}

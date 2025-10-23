"""
数据预处理模块
处理原始销售数据，支持多维度聚合和多时间粒度转换
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from src.config import (
    DATA_COLUMNS, REQUIRED_COLUMNS, OPTIONAL_COLUMNS,
    TIME_GRANULARITIES, AGGREGATION_DIMENSIONS,
    DEFAULT_FORECAST_CONFIG, PRODUCT_ID_FIELD
)
from src.utils import fill_missing_dates, detect_outliers


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, data_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        """
        初始化数据预处理器

        Args:
            data_path: 数据文件路径
            df: 或直接传入DataFrame
        """
        if data_path:
            self.raw_data = self._load_data(data_path)
        elif df is not None:
            self.raw_data = df.copy()
        else:
            raise ValueError("Must provide either data_path or df")

        self.processed_data = None
        logger.info(f"数据加载成功，共 {len(self.raw_data)} 行")

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """
        加载数据文件

        Args:
            data_path: 数据文件路径

        Returns:
            DataFrame
        """
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError(f"不支持的文件格式: {data_path}")

        return df

    def validate_columns(self) -> bool:
        """
        验证数据列是否完整（只验证必需字段）

        Returns:
            是否验证通过
        """
        # 只验证必需字段
        required_cols = list(REQUIRED_COLUMNS.values())
        missing_cols = [col for col in required_cols if col not in self.raw_data.columns]

        if missing_cols:
            logger.error(f"缺少必需的列: {missing_cols}")
            logger.info(f"当前数据列: {list(self.raw_data.columns)}")
            return False

        # 检查可选字段
        optional_cols = list(OPTIONAL_COLUMNS.values())
        existing_optional = [col for col in optional_cols if col in self.raw_data.columns]
        missing_optional = [col for col in optional_cols if col not in self.raw_data.columns]

        if existing_optional:
            logger.info(f"找到可选字段: {existing_optional}")
        if missing_optional:
            logger.info(f"缺少可选字段（将跳过）: {missing_optional}")

        logger.info("数据列验证通过")
        return True

    def clean_data(self, remove_outliers: bool = False) -> pd.DataFrame:
        """
        清洗数据

        Args:
            remove_outliers: 是否移除异常值

        Returns:
            清洗后的数据
        """
        df = self.raw_data.copy()

        # 转换日期格式
        date_col = DATA_COLUMNS['date']
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # 移除日期无效的行
        before_count = len(df)
        df = df.dropna(subset=[date_col])
        logger.info(f"移除无效日期行: {before_count - len(df)} 行")

        # 确保数量是数值类型
        quantity_col = DATA_COLUMNS['quantity']
        df[quantity_col] = pd.to_numeric(df[quantity_col], errors='coerce')

        # 移除数量为NaN的行
        before_count = len(df)
        df = df.dropna(subset=[quantity_col])
        if len(df) < before_count:
            logger.info(f"移除数量无效行: {before_count - len(df)} 行")

        # 注意：保留负数销量（可能表示退货等情况）
        logger.info(f"数据包含负数销量记录: {(df[quantity_col] < 0).sum()} 条")

        # 处理异常值（可选）
        if remove_outliers:
            original_count = len(df)
            # 只对正数销量检测异常值
            positive_mask = df[quantity_col] > 0
            if positive_mask.sum() > 0:
                outliers = detect_outliers(df.loc[positive_mask, quantity_col])
                # 创建一个与df索引对齐的outliers series
                outliers_full = pd.Series(False, index=df.index)
                outliers_full.loc[positive_mask] = outliers
                df = df[~outliers_full]
                logger.info(f"移除异常值: {original_count - len(df)} 行")

        # 填充缺失的必需分类字段
        required_cat_fields = ['store_id', 'product_code', 'province', 'city']
        for field in required_cat_fields:
            col = DATA_COLUMNS[field]
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')

        # 填充存在的可选分类字段
        optional_cat_fields = ['product_name', 'delivery_address', 'delivery_method']
        for field in optional_cat_fields:
            if field in DATA_COLUMNS and DATA_COLUMNS[field] in df.columns:
                df[DATA_COLUMNS[field]] = df[DATA_COLUMNS[field]].fillna('Unknown')

        logger.info(f"数据清洗完成，剩余 {len(df)} 行")
        self.processed_data = df
        return df

    def aggregate_by_dimension(
        self,
        df: pd.DataFrame,
        dimension: str,
        product_col: str,
        time_granularity: str = "daily"
    ) -> Dict[str, pd.DataFrame]:
        """
        按维度聚合数据

        Args:
            df: 数据框
            dimension: 聚合维度 ('store_id', 'province', 'city')
            product_col: 产品列名 (使用货品代码作为产品标识)
            time_granularity: 时间粒度 ('daily', 'weekly', 'monthly')

        Returns:
            字典，key为维度值+产品，value为聚合后的时间序列数据
        """
        date_col = DATA_COLUMNS['date']
        quantity_col = DATA_COLUMNS['quantity']

        # 获取维度列名
        if dimension == "store_id":
            dim_col = DATA_COLUMNS['store_id']
        elif dimension == "province":
            dim_col = DATA_COLUMNS['province']
        elif dimension == "city":
            dim_col = DATA_COLUMNS['city']
        else:
            raise ValueError(f"不支持的维度: {dimension}")

        # 获取时间频率
        freq = TIME_GRANULARITIES.get(time_granularity, "D")

        # 按维度和产品分组
        grouped_data = {}

        for (dim_value, product), group in df.groupby([dim_col, product_col]):
            # 按时间聚合
            time_series = group.groupby(pd.Grouper(key=date_col, freq=freq))[quantity_col].sum()
            time_series = time_series.reset_index()
            time_series.columns = ['date', 'quantity']

            # 填补缺失日期
            time_series = fill_missing_dates(time_series, 'date', freq)

            # 生成唯一键
            key = f"{dim_value}_{product}"
            grouped_data[key] = time_series

        logger.info(f"按 {dimension} 聚合完成，共 {len(grouped_data)} 个时间序列")
        return grouped_data

    def prepare_forecast_data(
        self,
        dimension: str = "store_id",
        time_granularity: str = "daily",
        min_samples: int = None
    ) -> Dict[str, pd.DataFrame]:
        """
        准备预测数据

        Args:
            dimension: 聚合维度
            time_granularity: 时间粒度
            min_samples: 最小样本数要求

        Returns:
            准备好的预测数据集
        """
        if self.processed_data is None:
            raise ValueError("请先运行 clean_data() 进行数据清洗")

        if min_samples is None:
            min_samples = DEFAULT_FORECAST_CONFIG['min_train_samples']

        # 使用货品代码作为产品标识
        product_col = DATA_COLUMNS[PRODUCT_ID_FIELD]

        # 聚合数据
        grouped_data = self.aggregate_by_dimension(
            self.processed_data,
            dimension,
            product_col,
            time_granularity
        )

        # 过滤样本数不足的序列
        filtered_data = {}
        for key, ts_data in grouped_data.items():
            if len(ts_data) >= min_samples:
                filtered_data[key] = ts_data
            else:
                logger.warning(f"序列 {key} 样本数不足 ({len(ts_data)} < {min_samples})，已过滤")

        logger.info(f"过滤后剩余 {len(filtered_data)} 个有效时间序列")
        return filtered_data

    def get_summary_statistics(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        获取数据摘要统计

        Args:
            df: 数据框，如果为None则使用processed_data

        Returns:
            统计摘要
        """
        if df is None:
            df = self.processed_data if self.processed_data is not None else self.raw_data

        quantity_col = DATA_COLUMNS['quantity']
        product_col = DATA_COLUMNS[PRODUCT_ID_FIELD]

        summary = {
            "总记录数": len(df),
            "总销量": df[quantity_col].sum(),
            "平均销量": df[quantity_col].mean(),
            "销量中位数": df[quantity_col].median(),
            "销量标准差": df[quantity_col].std(),
            "最小销量": df[quantity_col].min(),
            "最大销量": df[quantity_col].max(),
            "唯一产品数": df[product_col].nunique(),
            "唯一店铺数": df[DATA_COLUMNS['store_id']].nunique(),
            "时间跨度": f"{df[DATA_COLUMNS['date']].min()} 至 {df[DATA_COLUMNS['date']].max()}",
            "负数销量记录数": (df[quantity_col] < 0).sum()
        }

        return pd.Series(summary)

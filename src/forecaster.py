"""
销售预测器
统一的预测接口，支持多维度、多时间粒度的预测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from loguru import logger
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import DataPreprocessor
from src.model_selector import ModelSelector, EnsembleSelector
from src.config import (
    TIME_GRANULARITIES, AGGREGATION_DIMENSIONS,
    DEFAULT_FORECAST_CONFIG, OUTPUT_DIR
)
import os


class SalesForecaster:
    """销售预测器 - 主要预测接口"""

    def __init__(
        self,
        data_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        auto_clean: bool = True
    ):
        """
        初始化预测器

        Args:
            data_path: 数据文件路径
            df: 或直接传入DataFrame
            auto_clean: 是否自动清洗数据
        """
        # 初始化数据预处理器
        self.preprocessor = DataPreprocessor(data_path=data_path, df=df)

        if auto_clean:
            self.preprocessor.clean_data()

        self.forecast_results = {}

    def forecast_by_dimension(
        self,
        dimension: str,
        time_granularity: str = "daily",
        forecast_horizon: Optional[int] = None,
        use_ensemble: bool = False,
        models: Optional[List[str]] = None,
        save_results: bool = True
    ) -> Dict[str, Dict]:
        """
        按指定维度进行预测

        Args:
            dimension: 聚合维度 ('store_id', 'province', 'city')
            time_granularity: 时间粒度 ('daily', 'weekly', 'monthly')
            forecast_horizon: 预测步数（天/周/月），None则使用默认值
            use_ensemble: 是否使用集成模型
            models: 要使用的模型列表，None则使用所有模型
            save_results: 是否保存结果

        Returns:
            预测结果字典
        """
        logger.info(f"开始预测 - 维度: {dimension}, 时间粒度: {time_granularity}")

        # 准备预测数据
        grouped_data = self.preprocessor.prepare_forecast_data(
            dimension=dimension,
            time_granularity=time_granularity
        )

        if not grouped_data:
            logger.warning("没有可用的时间序列数据")
            return {}

        # 获取预测步数
        if forecast_horizon is None:
            forecast_horizon = DEFAULT_FORECAST_CONFIG['forecast_horizon'].get(
                time_granularity, 30
            )

        # 对每个时间序列进行预测
        results = {}

        for series_key, ts_data in tqdm(grouped_data.items(), desc="预测进度"):
            try:
                # 选择模型
                if use_ensemble:
                    selector = EnsembleSelector(models=models)
                    selector.fit(ts_data)
                    predictions = selector.predict(forecast_horizon)

                    result = {
                        'model_type': 'Ensemble',
                        'predictions': predictions,
                        'success': True
                    }
                else:
                    selector = ModelSelector(models=models)
                    best_model, eval_results = selector.select_best_model(ts_data)

                    # 预测
                    predictions = best_model.predict(forecast_horizon)

                    result = {
                        'model_type': 'Single',
                        'best_model': selector.best_model_name,
                        'best_score': selector.best_score,
                        'predictions': predictions,
                        'evaluation': eval_results,
                        'success': True
                    }

                # 生成预测日期
                last_date = ts_data['date'].max()
                freq = TIME_GRANULARITIES[time_granularity]
                forecast_dates = pd.date_range(
                    start=last_date,
                    periods=forecast_horizon + 1,
                    freq=freq
                )[1:]

                result['dates'] = forecast_dates
                result['historical_data'] = ts_data

                results[series_key] = result

                logger.info(f"✓ {series_key} 预测完成")

            except Exception as e:
                logger.error(f"✗ {series_key} 预测失败: {str(e)}")
                results[series_key] = {
                    'success': False,
                    'error': str(e)
                }

        # 保存结果
        if save_results:
            self._save_results(results, dimension, time_granularity)

        # 存储到实例变量
        key = f"{dimension}_{time_granularity}"
        self.forecast_results[key] = results

        logger.info(f"预测完成 - 成功: {sum(1 for r in results.values() if r.get('success', False))}/{len(results)}")

        return results

    def forecast_all_dimensions(
        self,
        time_granularities: Optional[List[str]] = None,
        forecast_horizons: Optional[Dict[str, int]] = None,
        use_ensemble: bool = False,
        models: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        对所有维度进行预测

        Args:
            time_granularities: 时间粒度列表，None则使用所有粒度
            forecast_horizons: 各时间粒度的预测步数
            use_ensemble: 是否使用集成模型
            models: 要使用的模型列表

        Returns:
            所有预测结果
        """
        if time_granularities is None:
            time_granularities = list(TIME_GRANULARITIES.keys())

        all_results = {}

        for dimension in AGGREGATION_DIMENSIONS:
            for granularity in time_granularities:
                horizon = None
                if forecast_horizons:
                    horizon = forecast_horizons.get(granularity)

                results = self.forecast_by_dimension(
                    dimension=dimension,
                    time_granularity=granularity,
                    forecast_horizon=horizon,
                    use_ensemble=use_ensemble,
                    models=models
                )

                key = f"{dimension}_{granularity}"
                all_results[key] = results

        return all_results

    def get_forecast_dataframe(
        self,
        dimension: str,
        time_granularity: str,
        series_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取预测结果的DataFrame格式

        Args:
            dimension: 维度
            time_granularity: 时间粒度
            series_key: 特定的时间序列key，None则返回所有

        Returns:
            预测结果DataFrame
        """
        key = f"{dimension}_{time_granularity}"

        if key not in self.forecast_results:
            raise ValueError(f"未找到预测结果: {key}")

        results = self.forecast_results[key]

        if series_key:
            if series_key not in results:
                raise ValueError(f"未找到时间序列: {series_key}")
            results = {series_key: results[series_key]}

        # 转换为DataFrame
        all_data = []

        for sk, result in results.items():
            if not result.get('success', False):
                continue

            # 拆分series_key
            parts = sk.split('_', 1)
            dim_value = parts[0] if len(parts) > 0 else sk
            product = parts[1] if len(parts) > 1 else 'Unknown'

            df = pd.DataFrame({
                'date': result['dates'],
                'predicted_quantity': result['predictions'],
                dimension: dim_value,
                'product': product,
                'model': result.get('best_model', result.get('model_type', 'Unknown')),
                'time_granularity': time_granularity
            })

            all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)

        return combined_df

    def _save_results(
        self,
        results: Dict,
        dimension: str,
        time_granularity: str
    ):
        """
        保存预测结果到CSV

        Args:
            results: 预测结果
            dimension: 维度
            time_granularity: 时间粒度
        """
        try:
            df = pd.DataFrame()

            for series_key, result in results.items():
                if not result.get('success', False):
                    continue

                parts = series_key.split('_', 1)
                dim_value = parts[0] if len(parts) > 0 else series_key
                product = parts[1] if len(parts) > 1 else 'Unknown'

                temp_df = pd.DataFrame({
                    'date': result['dates'],
                    'predicted_quantity': result['predictions'],
                    dimension: dim_value,
                    'product': product,
                    'model': result.get('best_model', result.get('model_type', 'Unknown')),
                    'time_granularity': time_granularity
                })

                df = pd.concat([df, temp_df], ignore_index=True)

            if not df.empty:
                output_file = os.path.join(
                    OUTPUT_DIR,
                    f"forecast_{dimension}_{time_granularity}.csv"
                )
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                logger.info(f"预测结果已保存至: {output_file}")

        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")

    def get_summary_statistics(self) -> pd.DataFrame:
        """获取数据摘要统计"""
        return self.preprocessor.get_summary_statistics()

    def export_all_results(self, output_dir: Optional[str] = None):
        """
        导出所有预测结果

        Args:
            output_dir: 输出目录，None则使用默认目录
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR

        os.makedirs(output_dir, exist_ok=True)

        for key, results in self.forecast_results.items():
            dimension, granularity = key.split('_', 1)

            df = self.get_forecast_dataframe(dimension, granularity)

            if not df.empty:
                output_file = os.path.join(output_dir, f"forecast_{key}.csv")
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                logger.info(f"导出: {output_file}")

        logger.info(f"所有结果已导出至: {output_dir}")

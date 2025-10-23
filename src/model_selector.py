"""
模型选择器
自动评估多个模型并选择最优模型
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from src.models import (
    ARIMAModel, AutoARIMAModel, ProphetModel,
    LSTMModel, XGBoostModel, LightGBMModel
)
from src.config import AVAILABLE_MODELS, PRIMARY_METRIC, DEFAULT_FORECAST_CONFIG
from src.utils import split_train_test


class ModelSelector:
    """模型选择器 - 自动选择最优预测模型"""

    def __init__(self, models: Optional[List[str]] = None, metric: str = PRIMARY_METRIC):
        """
        初始化模型选择器

        Args:
            models: 要评估的模型列表，None则使用所有可用模型
            metric: 评估指标 ('mae', 'rmse', 'mape', 'smape')
        """
        self.models = models or AVAILABLE_MODELS
        self.metric = metric
        self.evaluation_results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = np.inf

    def _create_model(self, model_name: str):
        """
        创建模型实例

        Args:
            model_name: 模型名称

        Returns:
            模型实例
        """
        model_map = {
            'arima': ARIMAModel,
            'auto_arima': AutoARIMAModel,
            'prophet': ProphetModel,
            'lstm': LSTMModel,
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel
        }

        if model_name.lower() not in model_map:
            raise ValueError(f"未知的模型: {model_name}")

        return model_map[model_name.lower()]()

    def _evaluate_single_model(
        self,
        model_name: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Tuple[str, Dict]:
        """
        评估单个模型

        Args:
            model_name: 模型名称
            train_data: 训练数据
            test_data: 测试数据

        Returns:
            (模型名称, 评估结果字典)
        """
        try:
            logger.info(f"评估模型: {model_name}")

            # 创建模型
            model = self._create_model(model_name)

            # 评估模型
            metrics = model.evaluate(train_data, test_data)

            result = {
                'model': model,
                'metrics': metrics,
                'success': True,
                'error': None
            }

            logger.info(f"{model_name} 评估完成: {self.metric.upper()}={metrics[self.metric]:.2f}")

        except Exception as e:
            logger.error(f"{model_name} 评估失败: {str(e)}")
            result = {
                'model': None,
                'metrics': {
                    'mae': np.inf,
                    'rmse': np.inf,
                    'mape': np.inf,
                    'smape': np.inf
                },
                'success': False,
                'error': str(e)
            }

        return model_name, result

    def select_best_model(
        self,
        data: pd.DataFrame,
        test_size: float = None,
        parallel: bool = True
    ) -> Tuple[object, Dict]:
        """
        选择最优模型

        Args:
            data: 时间序列数据
            test_size: 测试集比例
            parallel: 是否并行评估

        Returns:
            (最优模型, 评估结果字典)
        """
        if test_size is None:
            test_size = DEFAULT_FORECAST_CONFIG['test_size']

        # 拆分数据
        train_data, test_data = split_train_test(data, test_size)

        logger.info(f"训练集: {len(train_data)} 样本, 测试集: {len(test_data)} 样本")
        logger.info(f"开始评估 {len(self.models)} 个模型...")

        # 评估所有模型
        if parallel:
            # 并行评估
            with ThreadPoolExecutor(max_workers=min(len(self.models), 4)) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_single_model,
                        model_name,
                        train_data,
                        test_data
                    ): model_name
                    for model_name in self.models
                }

                for future in as_completed(futures):
                    model_name, result = future.result()
                    self.evaluation_results[model_name] = result
        else:
            # 串行评估
            for model_name in self.models:
                model_name, result = self._evaluate_single_model(
                    model_name, train_data, test_data
                )
                self.evaluation_results[model_name] = result

        # 选择最优模型
        for model_name, result in self.evaluation_results.items():
            if result['success']:
                score = result['metrics'][self.metric]
                if score < self.best_score:
                    self.best_score = score
                    self.best_model_name = model_name
                    self.best_model = result['model']

        if self.best_model is None:
            raise ValueError("所有模型评估都失败了")

        logger.info(f"最优模型: {self.best_model_name}, {self.metric.upper()}={self.best_score:.2f}")

        # 使用完整数据重新训练最优模型
        self.best_model.fit(data)

        return self.best_model, self.evaluation_results

    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        获取评估摘要

        Returns:
            评估结果DataFrame
        """
        if not self.evaluation_results:
            return pd.DataFrame()

        summary_data = []
        for model_name, result in self.evaluation_results.items():
            row = {
                'model': model_name,
                'success': result['success'],
                **result['metrics']
            }
            summary_data.append(row)

        df = pd.DataFrame(summary_data)
        df = df.sort_values(by=self.metric)

        return df

    def forecast_with_best_model(self, steps: int) -> Dict:
        """
        使用最优模型进行预测

        Args:
            steps: 预测步数

        Returns:
            预测结果字典
        """
        if self.best_model is None:
            raise ValueError("请先运行 select_best_model()")

        predictions = self.best_model.predict(steps)

        return {
            'model_name': self.best_model_name,
            'predictions': predictions,
            'metric': self.metric,
            'score': self.best_score
        }


class EnsembleSelector:
    """集成选择器 - 使用多个模型的加权平均"""

    def __init__(self, models: Optional[List[str]] = None, top_k: int = 3):
        """
        初始化集成选择器

        Args:
            models: 要评估的模型列表
            top_k: 使用前k个最优模型
        """
        self.selector = ModelSelector(models)
        self.top_k = top_k
        self.top_models = []
        self.weights = []

    def fit(self, data: pd.DataFrame, test_size: float = None) -> 'EnsembleSelector':
        """
        训练集成模型

        Args:
            data: 时间序列数据
            test_size: 测试集比例

        Returns:
            self
        """
        # 选择最优模型
        self.selector.select_best_model(data, test_size)

        # 获取评估结果
        summary = self.selector.get_evaluation_summary()
        summary = summary[summary['success'] == True]

        # 选择前k个模型
        top_k_results = summary.head(self.top_k)

        # 计算权重（基于性能的倒数）
        scores = top_k_results[self.selector.metric].values
        # 避免除以0
        scores = np.maximum(scores, 1e-10)
        # 权重与误差成反比
        weights = 1.0 / scores
        weights = weights / weights.sum()  # 归一化

        # 保存模型和权重
        for idx, row in top_k_results.iterrows():
            model_name = row['model']
            model = self.selector.evaluation_results[model_name]['model']

            # 使用完整数据重新训练
            model.fit(data)

            self.top_models.append(model)

        self.weights = weights

        logger.info(f"集成模型使用 {len(self.top_models)} 个模型")
        for i, (model, weight) in enumerate(zip(self.top_models, self.weights)):
            logger.info(f"  模型{i+1}: {model.name}, 权重={weight:.3f}")

        return self

    def predict(self, steps: int) -> np.ndarray:
        """
        集成预测

        Args:
            steps: 预测步数

        Returns:
            预测值数组
        """
        if not self.top_models:
            raise ValueError("请先运行 fit()")

        # 收集所有模型的预测
        all_predictions = []
        for model in self.top_models:
            try:
                pred = model.predict(steps)
                all_predictions.append(pred)
            except Exception as e:
                logger.warning(f"模型 {model.name} 预测失败: {str(e)}")
                # 使用0填充
                all_predictions.append(np.zeros(steps))

        # 加权平均
        all_predictions = np.array(all_predictions)
        ensemble_pred = np.average(all_predictions, axis=0, weights=self.weights)

        return ensemble_pred

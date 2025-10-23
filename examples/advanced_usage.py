"""
高级使用示例
演示更复杂的使用场景
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from loguru import logger

from src.forecaster import SalesForecaster
from src.model_selector import ModelSelector
from src.data_preprocessing import DataPreprocessor


def example_advanced_1_custom_preprocessing():
    """高级示例1: 自定义数据预处理"""
    print("=" * 80)
    print("高级示例1: 自定义数据预处理流程")
    print("=" * 80)

    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    sample_data = pd.DataFrame({
        '下单时间': dates,
        '数量': [150 + i % 100 + np.random.randint(-20, 20) for i in range(300)],
        '送货专卖店卡号': ['店铺001'] * 300,
        '货品名称': ['产品A'] * 300,
        '省': ['广东省'] * 300,
        '市': ['深圳市'] * 300,
        '货品代码': ['SKU001'] * 300,
        '送货地址': ['深圳市南山区'] * 300,
        '配送方式': ['快递'] * 300,
        '月份': dates.to_period('M').astype(str)
    })

    # 手动控制预处理流程
    preprocessor = DataPreprocessor(df=sample_data)

    # 验证列
    if not preprocessor.validate_columns():
        print("数据列验证失败!")
        return

    # 清洗数据（可选是否移除异常值）
    cleaned_data = preprocessor.clean_data(remove_outliers=True)
    print(f"\n清洗后数据: {len(cleaned_data)} 行")

    # 查看摘要统计
    print("\n数据摘要统计:")
    print(preprocessor.get_summary_statistics())

    # 准备预测数据
    forecast_data = preprocessor.prepare_forecast_data(
        dimension='store_id',
        time_granularity='daily',
        min_samples=60  # 自定义最小样本数
    )

    print(f"\n准备好 {len(forecast_data)} 个时间序列用于预测")


def example_advanced_2_model_comparison():
    """高级示例2: 详细的模型比较"""
    print("\n" + "=" * 80)
    print("高级示例2: 详细比较所有模型的性能")
    print("=" * 80)

    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'quantity': [120 + i * 0.3 + np.sin(i / 7) * 15 + np.random.normal(0, 5) for i in range(200)]
    })

    # 使用ModelSelector进行详细比较
    selector = ModelSelector(
        models=['auto_arima', 'prophet', 'xgboost', 'lightgbm'],
        metric='mape'
    )

    # 选择最优模型
    best_model, eval_results = selector.select_best_model(
        sample_data,
        test_size=0.2,
        parallel=True
    )

    # 查看评估摘要
    print("\n模型评估摘要:")
    summary = selector.get_evaluation_summary()
    print(summary)

    # 查看详细评估结果
    print("\n详细评估结果:")
    for model_name, result in eval_results.items():
        print(f"\n{model_name}:")
        print(f"  成功: {result['success']}")
        if result['success']:
            print(f"  指标: {result['metrics']}")
        else:
            print(f"  错误: {result['error']}")

    # 使用最优模型预测
    print(f"\n使用最优模型 {selector.best_model_name} 进行预测...")
    predictions = selector.forecast_with_best_model(steps=30)
    print(f"预测结果: {predictions['predictions'][:5]}... (前5个)")


def example_advanced_3_batch_forecast():
    """高级示例3: 批量预测多个产品"""
    print("\n" + "=" * 80)
    print("高级示例3: 批量预测多个产品和店铺")
    print("=" * 80)

    # 创建多产品、多店铺数据
    dates = pd.date_range('2023-01-01', periods=180, freq='D')

    all_data = []
    stores = ['店铺A', '店铺B', '店铺C']
    products = ['产品X', '产品Y', '产品Z']

    for store in stores:
        for product in products:
            for date in dates:
                # 为每个组合生成不同的销量模式
                base = hash(store + product) % 100 + 50
                trend = (dates.get_loc(date) * 0.1)
                seasonal = np.sin(dates.get_loc(date) / 7) * 20
                noise = np.random.normal(0, 10)

                quantity = max(0, base + trend + seasonal + noise)

                all_data.append({
                    '下单时间': date,
                    '数量': quantity,
                    '送货专卖店卡号': store,
                    '货品名称': product,
                    '省': '广东省',
                    '市': '深圳市',
                    '货品代码': f'SKU_{product}',
                    '送货地址': f'{store}地址',
                    '配送方式': '快递',
                    '月份': date.to_period('M')
                })

    sample_data = pd.DataFrame(all_data)

    # 批量预测
    forecaster = SalesForecaster(df=sample_data)

    results = forecaster.forecast_by_dimension(
        dimension='store_id',
        time_granularity='daily',
        forecast_horizon=30,
        use_ensemble=False
    )

    # 统计结果
    success_count = sum(1 for r in results.values() if r.get('success', False))
    print(f"\n批量预测完成: {success_count}/{len(results)} 个时间序列成功")

    # 查看每个产品使用的最优模型
    print("\n各产品最优模型:")
    for series_key, result in results.items():
        if result.get('success'):
            print(f"  {series_key}: {result.get('best_model')} (MAPE: {result.get('best_score', 0):.2f}%)")


def example_advanced_4_forecast_with_confidence():
    """高级示例4: 带置信区间的预测（使用Prophet）"""
    print("\n" + "=" * 80)
    print("高级示例4: 获取预测的置信区间")
    print("=" * 80)

    from src.models import ProphetModel

    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=150, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'quantity': [100 + i * 0.5 + np.sin(i / 7) * 10 + np.random.normal(0, 5) for i in range(150)]
    })

    # 使用Prophet模型（支持置信区间）
    model = ProphetModel()
    model.fit(sample_data)

    # 获取带置信区间的预测
    forecast_components = model.get_forecast_components(steps=30)

    print("\n带置信区间的预测结果:")
    print(forecast_components[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10))


def example_advanced_5_export_and_visualize():
    """高级示例5: 导出和可视化预测结果"""
    print("\n" + "=" * 80)
    print("高级示例5: 导出和可视化预测结果")
    print("=" * 80)

    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=180, freq='D')
    sample_data = pd.DataFrame({
        '下单时间': dates,
        '数量': [150 + i * 0.3 + np.sin(i / 7) * 20 for i in range(180)],
        '送货专卖店卡号': ['店铺001'] * 180,
        '货品名称': ['产品A'] * 180,
        '省': ['广东省'] * 180,
        '市': ['深圳市'] * 180,
        '货品代码': ['SKU001'] * 180,
        '送货地址': ['深圳市南山区'] * 180,
        '配送方式': ['快递'] * 180,
        '月份': dates.to_period('M').astype(str)
    })

    # 预测
    forecaster = SalesForecaster(df=sample_data)

    results = forecaster.forecast_by_dimension(
        dimension='store_id',
        time_granularity='daily',
        forecast_horizon=30,
        save_results=True
    )

    # 获取预测结果DataFrame
    df_result = forecaster.get_forecast_dataframe('store_id', 'daily')

    print("\n预测结果样例:")
    print(df_result.head(10))

    # 导出所有结果
    forecaster.export_all_results(output_dir='output/advanced_example')
    print("\n结果已导出至 output/advanced_example/")

    # 简单的可视化（如果安装了matplotlib）
    try:
        import matplotlib.pyplot as plt

        for series_key, result in results.items():
            if not result.get('success'):
                continue

            # 历史数据
            hist_data = result['historical_data']

            # 预测数据
            forecast_dates = result['dates']
            forecast_values = result['predictions']

            plt.figure(figsize=(12, 6))
            plt.plot(hist_data['date'], hist_data['quantity'], label='历史数据', marker='o', markersize=2)
            plt.plot(forecast_dates, forecast_values, label='预测', marker='s', markersize=3, linestyle='--')
            plt.xlabel('日期')
            plt.ylabel('销量')
            plt.title(f'{series_key} - 销量预测')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # 保存图表
            output_file = f'output/advanced_example/{series_key}_forecast.png'
            plt.savefig(output_file, dpi=150)
            print(f"图表已保存: {output_file}")
            plt.close()

    except ImportError:
        print("\n未安装matplotlib，跳过可视化")


if __name__ == "__main__":
    # 配置日志
    logger.add("logs/advanced_forecast.log", rotation="10 MB")

    print("\n" + "=" * 80)
    print("SKU销售预测系统 - 高级使用示例")
    print("=" * 80)

    try:
        example_advanced_1_custom_preprocessing()
        # example_advanced_2_model_comparison()
        # example_advanced_3_batch_forecast()
        # example_advanced_4_forecast_with_confidence()
        # example_advanced_5_export_and_visualize()

        print("\n" + "=" * 80)
        print("高级示例运行完成!")
        print("=" * 80)

    except Exception as e:
        logger.error(f"运行示例时出错: {str(e)}")
        import traceback
        traceback.print_exc()

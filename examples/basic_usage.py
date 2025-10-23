"""
基础使用示例
演示如何使用销售预测系统
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forecaster import SalesForecaster
from loguru import logger
import pandas as pd


def example_1_basic_forecast():
    """示例1: 基础预测 - 按店铺预测"""
    print("=" * 80)
    print("示例1: 基础预测 - 按店铺预测每日销量")
    print("=" * 80)

    # 加载数据（请替换为你的数据文件路径）
    data_path = "data/sample/sales_data.csv"

    # 或者使用示例DataFrame
    # 创建示例数据（只包含必需字段）
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    sample_data = pd.DataFrame({
        '下单时间': dates,
        '数量': [100 + i % 50 + (i % 7) * 10 for i in range(365)],
        '送货专卖店卡号': ['店铺001'] * 365,
        '货品代码': ['SKU001'] * 365,
        '省': ['广东省'] * 365,
        '市': ['深圳市'] * 365,
        '配送方式': ['快递'] * 365,
        '月份': dates.to_period('M').astype(str)
    })

    # 初始化预测器
    forecaster = SalesForecaster(df=sample_data, auto_clean=True)

    # 查看数据摘要
    print("\n数据摘要:")
    print(forecaster.get_summary_statistics())

    # 按店铺进行每日预测
    results = forecaster.forecast_by_dimension(
        dimension='store_id',  # 按店铺
        time_granularity='daily',  # 每日粒度
        forecast_horizon=30,  # 预测未来30天
        use_ensemble=False,  # 使用单一最优模型
        save_results=True  # 保存结果
    )

    # 查看结果
    for series_key, result in results.items():
        if result.get('success'):
            print(f"\n{series_key}:")
            print(f"  最优模型: {result.get('best_model')}")
            print(f"  评估分数(MAPE): {result.get('best_score', 0):.2f}%")
            print(f"  预测未来30天销量: {result['predictions'][:5]}... (前5天)")

    # 获取DataFrame格式的结果
    df_result = forecaster.get_forecast_dataframe('store_id', 'daily')
    print(f"\n预测结果 DataFrame:\n{df_result.head(10)}")


def example_2_multi_granularity():
    """示例2: 多时间粒度预测"""
    print("\n" + "=" * 80)
    print("示例2: 多时间粒度预测 - 按省份预测")
    print("=" * 80)

    # 创建示例数据（只包含必需字段）
    dates = pd.date_range('2023-01-01', periods=180, freq='D')
    sample_data = pd.DataFrame({
        '下单时间': dates,
        '数量': [200 + i % 100 for i in range(180)],
        '送货专卖店卡号': ['店铺002'] * 180,
        '货品代码': ['SKU002'] * 180,
        '省': ['北京市'] * 180,
        '市': ['北京市'] * 180,
        '配送方式': ['自提'] * 180,
        '月份': dates.to_period('M').astype(str)
    })

    forecaster = SalesForecaster(df=sample_data)

    # 按周预测
    print("\n按周预测...")
    weekly_results = forecaster.forecast_by_dimension(
        dimension='province',
        time_granularity='weekly',
        forecast_horizon=12,  # 预测未来12周
        use_ensemble=False
    )

    # 按月预测
    print("\n按月预测...")
    monthly_results = forecaster.forecast_by_dimension(
        dimension='province',
        time_granularity='monthly',
        forecast_horizon=6,  # 预测未来6个月
        use_ensemble=False
    )

    print("\n预测完成!")


def example_3_ensemble_forecast():
    """示例3: 使用集成模型预测"""
    print("\n" + "=" * 80)
    print("示例3: 使用集成模型提高预测准确性")
    print("=" * 80)

    # 创建示例数据（只包含必需字段）
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    sample_data = pd.DataFrame({
        '下单时间': dates,
        '数量': [150 + i % 80 + (i % 30) * 5 for i in range(200)],
        '送货专卖店卡号': ['店铺003'] * 200,
        '货品代码': ['SKU003'] * 200,
        '省': ['上海市'] * 200,
        '市': ['上海市'] * 200,
        '配送方式': ['快递'] * 200,
        '月份': dates.to_period('M').astype(str)
    })

    forecaster = SalesForecaster(df=sample_data)

    # 使用集成模型
    results = forecaster.forecast_by_dimension(
        dimension='city',
        time_granularity='daily',
        forecast_horizon=30,
        use_ensemble=True,  # 启用集成模型
        models=['auto_arima', 'prophet', 'xgboost']  # 指定使用的模型
    )

    print("\n集成模型预测完成!")


def example_4_all_dimensions():
    """示例4: 对所有维度进行预测"""
    print("\n" + "=" * 80)
    print("示例4: 对所有维度和时间粒度进行批量预测")
    print("=" * 80)

    # 创建更复杂的示例数据（多个店铺、产品）
    dates = pd.date_range('2023-01-01', periods=150, freq='D')

    data_list = []
    for store in ['店铺A', '店铺B']:
        for product_code in ['SKU_X', 'SKU_Y']:
            for date in dates:
                data_list.append({
                    '下单时间': date,
                    '数量': np.random.randint(50, 200),
                    '送货专卖店卡号': store,
                    '货品代码': product_code,
                    '省': '广东省',
                    '市': '深圳市' if store == '店铺A' else '广州市',
                    '配送方式': '快递',
                    '月份': date.to_period('M')
                })

    sample_data = pd.DataFrame(data_list)

    forecaster = SalesForecaster(df=sample_data)

    # 对所有维度进行预测
    all_results = forecaster.forecast_all_dimensions(
        time_granularities=['daily', 'weekly'],  # 只预测日和周
        use_ensemble=False
    )

    print(f"\n完成预测，共 {len(all_results)} 个维度组合")

    # 导出所有结果
    forecaster.export_all_results()
    print("\n所有结果已导出!")


def example_5_custom_models():
    """示例5: 自定义选择特定模型"""
    print("\n" + "=" * 80)
    print("示例5: 只使用特定的预测模型")
    print("=" * 80)

    # 创建示例数据（只包含必需字段）
    dates = pd.date_range('2023-01-01', periods=120, freq='D')
    sample_data = pd.DataFrame({
        '下单时间': dates,
        '数量': [100 + i * 0.5 + np.sin(i / 7) * 20 for i in range(120)],
        '送货专卖店卡号': ['店铺004'] * 120,
        '货品代码': ['SKU004'] * 120,
        '省': ['江苏省'] * 120,
        '市': ['南京市'] * 120,
        '配送方式': ['自提'] * 120,
        '月份': dates.to_period('M').astype(str)
    })

    forecaster = SalesForecaster(df=sample_data)

    # 只使用ARIMA和Prophet模型
    results = forecaster.forecast_by_dimension(
        dimension='store_id',
        time_granularity='daily',
        forecast_horizon=15,
        models=['auto_arima', 'prophet']  # 只使用这两个模型
    )

    print("\n自定义模型预测完成!")


if __name__ == "__main__":
    import numpy as np

    # 配置日志
    logger.add("logs/forecast.log", rotation="10 MB")

    print("\n" + "=" * 80)
    print("SKU销售预测系统 - 使用示例")
    print("=" * 80)

    # 运行示例
    try:
        example_1_basic_forecast()
        # example_2_multi_granularity()
        # example_3_ensemble_forecast()
        # example_4_all_dimensions()
        # example_5_custom_models()

        print("\n" + "=" * 80)
        print("所有示例运行完成!")
        print("=" * 80)

    except Exception as e:
        logger.error(f"运行示例时出错: {str(e)}")
        import traceback
        traceback.print_exc()

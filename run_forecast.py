"""
SKU销售预测 - 主运行脚本
直接修改数据路径即可使用
"""

from src.forecaster import SalesForecaster
import os
import sys

# ============================================================================
# 配置区域 - 请根据你的实际情况修改以下参数
# ============================================================================

# 数据文件路径（请修改为你的CSV文件路径）
DATA_PATH = "data/sample/sales_data.csv"  # 修改这里！

# 预测配置
FORECAST_CONFIG = {
    # 是否预测按店铺的销量
    "forecast_by_store": True,
    "store_daily_days": 30,      # 按店铺预测未来多少天
    "store_weekly_weeks": 12,    # 按店铺预测未来多少周
    "store_monthly_months": 6,   # 按店铺预测未来多少月

    # 是否预测按省份的销量
    "forecast_by_province": True,
    "province_daily_days": 30,
    "province_weekly_weeks": 12,
    "province_monthly_months": 6,

    # 是否预测按城市的销量
    "forecast_by_city": True,
    "city_daily_days": 30,
    "city_weekly_weeks": 12,
    "city_monthly_months": 6,

    # 模型配置
    "use_ensemble": False,  # True=使用集成模型(更准确但更慢), False=使用单一最优模型(更快)
    "models": ['auto_arima', 'prophet', 'xgboost'],  # 要使用的模型列表

    # 输出配置
    "output_dir": "output",  # 结果输出目录
}

# ============================================================================
# 主程序 - 通常不需要修改以下代码
# ============================================================================

def print_section(title):
    """打印分隔线标题"""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def main():
    """主预测流程"""

    print_section("SKU销售预测系统")
    print(f"数据文件: {DATA_PATH}")

    # 1. 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"\n✗ 错误：找不到数据文件 '{DATA_PATH}'")
        print("\n请执行以下操作：")
        print("1. 检查文件路径是否正确")
        print("2. 如果要使用示例数据，先运行: python examples/generate_sample_data.py")
        print("3. 或者修改本脚本开头的 DATA_PATH 变量为你的CSV文件路径")
        return 1

    # 2. 加载数据
    print_section("1/5 数据加载与验证")
    try:
        forecaster = SalesForecaster(data_path=DATA_PATH, auto_clean=True)
        print("✓ 数据加载成功")
    except Exception as e:
        print(f"✗ 数据加载失败: {str(e)}")
        print("\n常见问题：")
        print("1. CSV文件编码问题 - 尝试用UTF-8或GBK编码保存")
        print("2. 缺少必需字段 - 确保包含：下单时间、数量、送货专卖店卡号、货品代码、省、市")
        print("3. 日期格式问题 - 确保日期格式正确（如：2023-01-01）")
        return 1

    # 3. 显示数据摘要
    print_section("2/5 数据摘要统计")
    try:
        summary = forecaster.get_summary_statistics()
        for key, value in summary.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"✗ 获取摘要失败: {str(e)}")

    # 4. 执行预测
    print_section("3/5 执行预测")

    total_forecasts = 0
    successful_forecasts = 0

    # 按店铺预测
    if FORECAST_CONFIG['forecast_by_store']:
        print("\n按店铺预测:")

        # 每日
        print("  - 每日预测（未来{}天）...".format(FORECAST_CONFIG['store_daily_days']), end=' ')
        try:
            results = forecaster.forecast_by_dimension(
                dimension='store_id',
                time_granularity='daily',
                forecast_horizon=FORECAST_CONFIG['store_daily_days'],
                use_ensemble=FORECAST_CONFIG['use_ensemble'],
                models=FORECAST_CONFIG['models'],
                save_results=True
            )
            success = sum(1 for r in results.values() if r.get('success', False))
            total_forecasts += len(results)
            successful_forecasts += success
            print(f"✓ 完成 ({success}/{len(results)})")
        except Exception as e:
            print(f"✗ 失败: {str(e)}")

        # 每周
        print("  - 每周预测（未来{}周）...".format(FORECAST_CONFIG['store_weekly_weeks']), end=' ')
        try:
            results = forecaster.forecast_by_dimension(
                dimension='store_id',
                time_granularity='weekly',
                forecast_horizon=FORECAST_CONFIG['store_weekly_weeks'],
                use_ensemble=FORECAST_CONFIG['use_ensemble'],
                models=FORECAST_CONFIG['models'],
                save_results=True
            )
            success = sum(1 for r in results.values() if r.get('success', False))
            total_forecasts += len(results)
            successful_forecasts += success
            print(f"✓ 完成 ({success}/{len(results)})")
        except Exception as e:
            print(f"✗ 失败: {str(e)}")

        # 每月
        print("  - 每月预测（未来{}月）...".format(FORECAST_CONFIG['store_monthly_months']), end=' ')
        try:
            results = forecaster.forecast_by_dimension(
                dimension='store_id',
                time_granularity='monthly',
                forecast_horizon=FORECAST_CONFIG['store_monthly_months'],
                use_ensemble=FORECAST_CONFIG['use_ensemble'],
                models=FORECAST_CONFIG['models'],
                save_results=True
            )
            success = sum(1 for r in results.values() if r.get('success', False))
            total_forecasts += len(results)
            successful_forecasts += success
            print(f"✓ 完成 ({success}/{len(results)})")
        except Exception as e:
            print(f"✗ 失败: {str(e)}")

    # 按省份预测
    if FORECAST_CONFIG['forecast_by_province']:
        print("\n按省份预测:")

        # 每日
        print("  - 每日预测（未来{}天）...".format(FORECAST_CONFIG['province_daily_days']), end=' ')
        try:
            results = forecaster.forecast_by_dimension(
                dimension='province',
                time_granularity='daily',
                forecast_horizon=FORECAST_CONFIG['province_daily_days'],
                use_ensemble=FORECAST_CONFIG['use_ensemble'],
                models=FORECAST_CONFIG['models'],
                save_results=True
            )
            success = sum(1 for r in results.values() if r.get('success', False))
            total_forecasts += len(results)
            successful_forecasts += success
            print(f"✓ 完成 ({success}/{len(results)})")
        except Exception as e:
            print(f"✗ 失败: {str(e)}")

        # 每周
        print("  - 每周预测（未来{}周）...".format(FORECAST_CONFIG['province_weekly_weeks']), end=' ')
        try:
            results = forecaster.forecast_by_dimension(
                dimension='province',
                time_granularity='weekly',
                forecast_horizon=FORECAST_CONFIG['province_weekly_weeks'],
                use_ensemble=FORECAST_CONFIG['use_ensemble'],
                models=FORECAST_CONFIG['models'],
                save_results=True
            )
            success = sum(1 for r in results.values() if r.get('success', False))
            total_forecasts += len(results)
            successful_forecasts += success
            print(f"✓ 完成 ({success}/{len(results)})")
        except Exception as e:
            print(f"✗ 失败: {str(e)}")

        # 每月
        print("  - 每月预测（未来{}月）...".format(FORECAST_CONFIG['province_monthly_months']), end=' ')
        try:
            results = forecaster.forecast_by_dimension(
                dimension='province',
                time_granularity='monthly',
                forecast_horizon=FORECAST_CONFIG['province_monthly_months'],
                use_ensemble=FORECAST_CONFIG['use_ensemble'],
                models=FORECAST_CONFIG['models'],
                save_results=True
            )
            success = sum(1 for r in results.values() if r.get('success', False))
            total_forecasts += len(results)
            successful_forecasts += success
            print(f"✓ 完成 ({success}/{len(results)})")
        except Exception as e:
            print(f"✗ 失败: {str(e)}")

    # 按城市预测
    if FORECAST_CONFIG['forecast_by_city']:
        print("\n按城市预测:")

        # 每日
        print("  - 每日预测（未来{}天）...".format(FORECAST_CONFIG['city_daily_days']), end=' ')
        try:
            results = forecaster.forecast_by_dimension(
                dimension='city',
                time_granularity='daily',
                forecast_horizon=FORECAST_CONFIG['city_daily_days'],
                use_ensemble=FORECAST_CONFIG['use_ensemble'],
                models=FORECAST_CONFIG['models'],
                save_results=True
            )
            success = sum(1 for r in results.values() if r.get('success', False))
            total_forecasts += len(results)
            successful_forecasts += success
            print(f"✓ 完成 ({success}/{len(results)})")
        except Exception as e:
            print(f"✗ 失败: {str(e)}")

        # 每周
        print("  - 每周预测（未来{}周）...".format(FORECAST_CONFIG['city_weekly_weeks']), end=' ')
        try:
            results = forecaster.forecast_by_dimension(
                dimension='city',
                time_granularity='weekly',
                forecast_horizon=FORECAST_CONFIG['city_weekly_weeks'],
                use_ensemble=FORECAST_CONFIG['use_ensemble'],
                models=FORECAST_CONFIG['models'],
                save_results=True
            )
            success = sum(1 for r in results.values() if r.get('success', False))
            total_forecasts += len(results)
            successful_forecasts += success
            print(f"✓ 完成 ({success}/{len(results)})")
        except Exception as e:
            print(f"✗ 失败: {str(e)}")

        # 每月
        print("  - 每月预测（未来{}月）...".format(FORECAST_CONFIG['city_monthly_months']), end=' ')
        try:
            results = forecaster.forecast_by_dimension(
                dimension='city',
                time_granularity='monthly',
                forecast_horizon=FORECAST_CONFIG['city_monthly_months'],
                use_ensemble=FORECAST_CONFIG['use_ensemble'],
                models=FORECAST_CONFIG['models'],
                save_results=True
            )
            success = sum(1 for r in results.values() if r.get('success', False))
            total_forecasts += len(results)
            successful_forecasts += success
            print(f"✓ 完成 ({success}/{len(results)})")
        except Exception as e:
            print(f"✗ 失败: {str(e)}")

    # 5. 汇总统计
    print_section("4/5 预测统计")
    print(f"  总预测任务: {total_forecasts}")
    print(f"  成功: {successful_forecasts}")
    print(f"  失败: {total_forecasts - successful_forecasts}")
    if total_forecasts > 0:
        print(f"  成功率: {successful_forecasts/total_forecasts*100:.1f}%")

    # 6. 导出结果
    print_section("5/5 导出结果")
    try:
        output_dir = FORECAST_CONFIG['output_dir']
        forecaster.export_all_results(output_dir=output_dir)
        print(f"✓ 所有结果已导出到 {output_dir}/ 目录\n")

        # 列出输出文件
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
            if files:
                print("输出文件列表:")
                for file in sorted(files):
                    file_path = os.path.join(output_dir, file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  - {file} ({file_size:.1f} KB)")
            else:
                print("未生成输出文件")
    except Exception as e:
        print(f"✗ 导出失败: {str(e)}")

    # 7. 完成
    print_section("预测完成")
    print(f"✓ 所有预测任务已完成")
    print(f"✓ 结果已保存到 {FORECAST_CONFIG['output_dir']}/ 目录")
    print("\n下一步操作：")
    print(f"1. 查看预测结果: 打开 {FORECAST_CONFIG['output_dir']}/ 目录中的CSV文件")
    print("2. 使用Excel或其他工具分析预测数据")
    print("3. 如需调整预测参数，修改本脚本开头的 FORECAST_CONFIG")

    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n预测已被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ 发生未预期的错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

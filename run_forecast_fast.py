"""
SKU销售预测 - 快速预测脚本（优化版）
专注速度和效率，适合大数据量
"""

from src.forecaster import SalesForecaster
import os
import sys

# ============================================================================
# 配置区域 - 快速预测优化配置
# ============================================================================

# 数据文件路径
DATA_PATH = r"E:\03 Python xm-1\嘉兴数据V1.csv"

# 预测配置 - 按省份输出每个产品的分月销售预测
FORECAST_CONFIG = {
    # 选择要预测的维度 - 按省份
    "forecast_by_store": False,     # 不预测店铺
    "forecast_by_province": True,   # ✓ 按省份预测
    "forecast_by_city": False,      # 不预测城市

    # 选择要预测的时间粒度 - 按月
    "forecast_daily": False,        # 不预测每日
    "forecast_weekly": False,       # 不预测每周
    "forecast_monthly": True,       # ✓ 按月预测

    # 预测时长
    "daily_days": 30,               # 每日预测天数（未使用）
    "weekly_weeks": 12,             # 每周预测周数（未使用）
    "monthly_months": 6,            # 每月预测6个月

    # 模型配置 - 只使用最快的2个模型
    "use_ensemble": False,          # 关闭集成模型（加快速度）
    "models": ['prophet', 'xgboost'],  # 只用2个最快的模型

    # 数据过滤 - 只预测重要的产品/店铺
    "min_samples": 60,              # 提高最小样本数要求（过滤掉数据少的）

    # 输出配置
    "output_dir": "output",
}

# ============================================================================
# 主程序
# ============================================================================

def print_section(title):
    """打印分隔线标题"""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def main():
    """主预测流程"""

    print_section("SKU销售预测系统 - 快速版")
    print(f"数据文件: {DATA_PATH}")
    print("\n优化配置：")
    print(f"  - 预测维度: {'店铺' if FORECAST_CONFIG['forecast_by_store'] else ''}"
          f"{'、省份' if FORECAST_CONFIG['forecast_by_province'] else ''}"
          f"{'、城市' if FORECAST_CONFIG['forecast_by_city'] else ''}")
    print(f"  - 时间粒度: {'每日' if FORECAST_CONFIG['forecast_daily'] else ''}"
          f"{'、每周' if FORECAST_CONFIG['forecast_weekly'] else ''}"
          f"{'、每月' if FORECAST_CONFIG['forecast_monthly'] else ''}")
    print(f"  - 使用模型: {', '.join(FORECAST_CONFIG['models'])}")
    print(f"  - 最小样本数: {FORECAST_CONFIG['min_samples']}")

    # 1. 检查文件
    if not os.path.exists(DATA_PATH):
        print(f"\n✗ 错误：找不到数据文件 '{DATA_PATH}'")
        return 1

    # 2. 加载数据（自动检测编码）
    print_section("1/4 数据加载")

    import pandas as pd
    df = None
    encodings = ['gbk', 'gb2312', 'utf-8', 'utf-8-sig']

    for encoding in encodings:
        try:
            print(f"尝试 {encoding} 编码...", end=' ')
            df = pd.read_csv(DATA_PATH, encoding=encoding)
            print(f"✓ 成功")
            break
        except:
            print(f"✗")
            continue

    if df is None:
        print(f"\n✗ 无法读取文件")
        return 1

    try:
        forecaster = SalesForecaster(df=df, auto_clean=True)
        print("✓ 数据验证成功")
    except Exception as e:
        print(f"✗ 数据验证失败: {str(e)}")
        return 1

    # 3. 数据摘要
    print_section("2/4 数据摘要")
    try:
        summary = forecaster.get_summary_statistics()
        print(f"  总记录数: {summary['总记录数']}")
        print(f"  唯一产品数: {summary['唯一产品数']}")
        print(f"  唯一店铺数: {summary['唯一店铺数']}")
        print(f"  时间跨度: {summary['时间跨度']}")
    except Exception as e:
        print(f"✗ 获取摘要失败: {str(e)}")

    # 4. 执行预测
    print_section("3/4 执行预测")

    total_forecasts = 0
    successful_forecasts = 0

    dimensions = []
    if FORECAST_CONFIG['forecast_by_store']:
        dimensions.append(('store_id', '店铺'))
    if FORECAST_CONFIG['forecast_by_province']:
        dimensions.append(('province', '省份'))
    if FORECAST_CONFIG['forecast_by_city']:
        dimensions.append(('city', '城市'))

    granularities = []
    if FORECAST_CONFIG['forecast_daily']:
        granularities.append(('daily', '每日', FORECAST_CONFIG['daily_days']))
    if FORECAST_CONFIG['forecast_weekly']:
        granularities.append(('weekly', '每周', FORECAST_CONFIG['weekly_weeks']))
    if FORECAST_CONFIG['forecast_monthly']:
        granularities.append(('monthly', '每月', FORECAST_CONFIG['monthly_months']))

    total_tasks = len(dimensions) * len(granularities)
    current_task = 0

    for dim_key, dim_name in dimensions:
        print(f"\n按{dim_name}预测:")

        for gran_key, gran_name, horizon in granularities:
            current_task += 1
            print(f"  [{current_task}/{total_tasks}] {gran_name}预测（未来{horizon}{'天' if gran_key=='daily' else '周' if gran_key=='weekly' else '月'}）...", end=' ', flush=True)

            try:
                results = forecaster.forecast_by_dimension(
                    dimension=dim_key,
                    time_granularity=gran_key,
                    forecast_horizon=horizon,
                    use_ensemble=FORECAST_CONFIG['use_ensemble'],
                    models=FORECAST_CONFIG['models'],
                    save_results=True
                )

                success = sum(1 for r in results.values() if r.get('success', False))
                total_forecasts += len(results)
                successful_forecasts += success
                print(f"✓ ({success}/{len(results)})")

            except Exception as e:
                print(f"✗ {str(e)}")

    # 5. 结果汇总
    print_section("4/4 完成")
    print(f"  预测任务: {total_tasks}")
    print(f"  时间序列: {total_forecasts}")
    print(f"  成功: {successful_forecasts}")
    print(f"  失败: {total_forecasts - successful_forecasts}")

    if total_forecasts > 0:
        print(f"  成功率: {successful_forecasts/total_forecasts*100:.1f}%")

    # 6. 导出结果
    try:
        forecaster.export_all_results(output_dir=FORECAST_CONFIG['output_dir'])
        print(f"\n✓ 结果已保存到 {FORECAST_CONFIG['output_dir']}/ 目录")

        if os.path.exists(FORECAST_CONFIG['output_dir']):
            files = [f for f in os.listdir(FORECAST_CONFIG['output_dir']) if f.endswith('.csv')]
            if files:
                print(f"\n生成的文件 ({len(files)} 个):")
                for f in sorted(files):
                    print(f"  - {f}")
    except Exception as e:
        print(f"✗ 导出失败: {str(e)}")

    print("\n" + "="*80)
    print("预测完成！")
    print("="*80)

    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断预测（按了 Ctrl+C）")
        print("已停止运行")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

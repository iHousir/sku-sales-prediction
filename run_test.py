"""
SKU销售预测 - 测试脚本
专为您的环境配置
"""

from src.forecaster import SalesForecaster
import os
import sys
import glob

# ============================================================================
# 配置区域 - 已为您的环境配置
# ============================================================================

# 数据目录路径（您的实际路径）
DATA_DIR = r"E:\04 数据分析Python"

# 预测配置 - 按省份输出每个产品的分月销售预测
FORECAST_CONFIG = {
    # 预测维度 - 按省份
    "forecast_by_store": False,
    "forecast_by_province": True,   # ✓ 按省份预测
    "forecast_by_city": False,

    # 时间粒度 - 按月
    "forecast_daily": False,
    "forecast_weekly": False,
    "forecast_monthly": True,       # ✓ 按月预测

    # 预测时长
    "monthly_months": 6,            # 预测6个月

    # 模型配置 - 使用2个最快的模型
    "use_ensemble": False,
    "models": ['prophet', 'xgboost'],

    # 数据过滤
    "min_samples": 60,              # 最少需要60个历史数据点

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

def find_csv_files(directory):
    """在目录中查找CSV文件"""
    if os.path.isfile(directory) and directory.endswith('.csv'):
        # 如果直接指定了文件路径
        return directory

    if not os.path.isdir(directory):
        return None

    # 在目录中查找CSV文件
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    if len(csv_files) == 0:
        return None
    elif len(csv_files) == 1:
        return csv_files[0]
    else:
        # 多个CSV文件，让用户选择
        print("\n找到多个CSV文件：")
        for i, f in enumerate(csv_files, 1):
            size = os.path.getsize(f) / (1024 * 1024)  # MB
            print(f"  [{i}] {os.path.basename(f)} ({size:.2f} MB)")

        while True:
            try:
                choice = input("\n请选择文件编号 (1-{}): ".format(len(csv_files)))
                idx = int(choice) - 1
                if 0 <= idx < len(csv_files):
                    return csv_files[idx]
                else:
                    print("无效的选择，请重新输入")
            except (ValueError, KeyboardInterrupt):
                print("\n取消运行")
                sys.exit(1)

def main():
    """主预测流程"""

    print_section("SKU销售预测系统 - 测试版")
    print(f"数据目录: {DATA_DIR}")
    print("\n当前配置：")
    print(f"  - 预测维度: 省份")
    print(f"  - 时间粒度: 每月")
    print(f"  - 预测时长: {FORECAST_CONFIG['monthly_months']}个月")
    print(f"  - 使用模型: {', '.join(FORECAST_CONFIG['models'])}")
    print(f"  - 最小样本数: {FORECAST_CONFIG['min_samples']}")

    # 1. 查找数据文件
    print_section("1/4 查找数据文件")

    data_path = find_csv_files(DATA_DIR)

    if data_path is None:
        print(f"\n✗ 错误：在 '{DATA_DIR}' 目录下找不到CSV文件")
        print("\n请检查：")
        print("1. 目录路径是否正确")
        print("2. 目录下是否有 .csv 文件")
        print("3. 或者直接修改 run_test.py 第13行 DATA_DIR 为具体的CSV文件路径")
        print("\n示例：")
        print('   DATA_DIR = r"E:\\04 数据分析Python\\您的文件名.csv"')
        return 1

    print(f"✓ 找到数据文件: {os.path.basename(data_path)}")
    file_size = os.path.getsize(data_path) / (1024 * 1024)
    print(f"  文件大小: {file_size:.2f} MB")

    # 2. 加载数据（自动检测编码）
    print_section("2/4 数据加载与验证")

    import pandas as pd
    df = None
    encodings = ['gbk', 'gb2312', 'utf-8', 'utf-8-sig', 'gb18030']

    for encoding in encodings:
        try:
            print(f"尝试 {encoding} 编码...", end=' ')
            df = pd.read_csv(data_path, encoding=encoding)
            print(f"✓ 成功")
            break
        except:
            print(f"✗")
            continue

    if df is None:
        print(f"\n✗ 无法读取CSV文件")
        print("\n建议：")
        print("1. 用Excel打开CSV文件，另存为 UTF-8 编码")
        print("2. 或检查文件是否损坏")
        return 1

    print(f"✓ 读取成功，共 {len(df)} 条记录")

    # 验证数据
    try:
        forecaster = SalesForecaster(df=df, auto_clean=True)
        print("✓ 数据验证成功")
    except Exception as e:
        print(f"✗ 数据验证失败: {str(e)}")
        print("\n请检查CSV文件是否包含必需字段：")
        print("  - 下单时间")
        print("  - 数量")
        print("  - 送货专卖店卡号")
        print("  - 货品代码")
        print("  - 省")
        print("  - 市")
        print(f"\n当前数据列: {list(df.columns)}")
        return 1

    # 3. 数据摘要
    print_section("3/4 数据摘要")
    try:
        summary = forecaster.get_summary_statistics()
        print(f"  总记录数: {summary.get('总记录数', 'N/A')}")
        print(f"  唯一产品数: {summary.get('唯一产品数', 'N/A')}")
        print(f"  唯一店铺数: {summary.get('唯一店铺数', 'N/A')}")
        if '唯一省份数' in summary:
            print(f"  唯一省份数: {summary['唯一省份数']}")
        print(f"  时间跨度: {summary.get('时间跨度', 'N/A')}")

        # 预估预测数量
        n_provinces = summary.get('唯一省份数', 0)
        n_products = summary.get('唯一产品数', 0)
        estimated_forecasts = n_provinces * n_products
        print(f"\n  预计生成预测: 约 {estimated_forecasts} 个时间序列")

        # 预估运行时间
        if estimated_forecasts < 100:
            time_estimate = "5-10分钟"
        elif estimated_forecasts < 500:
            time_estimate = "10-30分钟"
        else:
            time_estimate = "30-60分钟"
        print(f"  预估运行时间: {time_estimate}")

    except Exception as e:
        print(f"✗ 获取摘要失败: {str(e)}")

    # 4. 执行预测
    print_section("4/4 执行预测")
    print("\n开始预测，请耐心等待...\n")

    total_forecasts = 0
    successful_forecasts = 0

    print("按省份预测:")
    print(f"  [1/1] 每月预测（未来{FORECAST_CONFIG['monthly_months']}月）...", end=' ', flush=True)

    try:
        results = forecaster.forecast_by_dimension(
            dimension='province',
            time_granularity='monthly',
            forecast_horizon=FORECAST_CONFIG['monthly_months'],
            use_ensemble=FORECAST_CONFIG['use_ensemble'],
            models=FORECAST_CONFIG['models'],
            save_results=True
        )

        success = sum(1 for r in results.values() if r.get('success', False))
        total_forecasts = len(results)
        successful_forecasts = success
        print(f"✓ ({success}/{len(results)})")

    except Exception as e:
        print(f"✗ {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    # 5. 结果汇总
    print_section("预测完成")
    print(f"  预测任务: 1")
    print(f"  时间序列: {total_forecasts}")
    print(f"  成功: {successful_forecasts}")
    print(f"  失败: {total_forecasts - successful_forecasts}")

    if total_forecasts > 0:
        print(f"  成功率: {successful_forecasts/total_forecasts*100:.1f}%")

    # 6. 导出结果
    try:
        forecaster.export_all_results(output_dir=FORECAST_CONFIG['output_dir'])
        print(f"\n✓ 结果已保存到 {FORECAST_CONFIG['output_dir']}/ 目录")

        output_path = os.path.abspath(FORECAST_CONFIG['output_dir'])
        print(f"\n完整路径: {output_path}")

        if os.path.exists(FORECAST_CONFIG['output_dir']):
            files = [f for f in os.listdir(FORECAST_CONFIG['output_dir']) if f.endswith('.csv')]
            if files:
                print(f"\n生成的文件 ({len(files)} 个):")
                for f in sorted(files):
                    file_path = os.path.join(FORECAST_CONFIG['output_dir'], f)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  - {f} ({file_size:.1f} KB)")

                print("\n用Excel打开CSV文件即可查看预测结果")
    except Exception as e:
        print(f"✗ 导出失败: {str(e)}")

    print("\n" + "="*80)
    print("预测完成！")
    print("="*80)
    print("\n下一步：")
    print(f"1. 打开文件夹: {os.path.abspath(FORECAST_CONFIG['output_dir'])}")
    print("2. 用Excel打开 forecast_province_monthly.csv 查看预测结果")
    print("3. 如需调整配置，修改 run_test.py 第17-37行")

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

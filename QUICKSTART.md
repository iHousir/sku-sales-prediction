# 快速开始指南

本指南将帮助你在5分钟内开始使用SKU销售预测系统。

## 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖包
pip install -r requirements.txt
```

**注意**: 安装可能需要5-10分钟，特别是TensorFlow和Prophet等包。

## 2. 生成示例数据

如果你还没有自己的数据，可以先生成示例数据：

```bash
python examples/generate_sample_data.py
```

这将在 `data/sample/sales_data.csv` 生成一年的示例销售数据，包含：
- 3个店铺
- 5个产品
- 365天的每日销售记录
- 包含趋势、季节性和随机波动

## 3. 运行你的第一个预测

### 方法1: 使用示例脚本

```bash
# 基础示例
python examples/basic_usage.py

# 高级示例
python examples/advanced_usage.py
```

### 方法2: 交互式Python

```python
from src.forecaster import SalesForecaster

# 加载数据（使用生成的示例数据）
forecaster = SalesForecaster(
    data_path="data/sample/sales_data.csv",
    auto_clean=True
)

# 查看数据摘要
print(forecaster.get_summary_statistics())

# 预测：按店铺预测未来30天的每日销量
results = forecaster.forecast_by_dimension(
    dimension='store_id',      # 按店铺
    time_granularity='daily',  # 每日
    forecast_horizon=30,       # 预测30天
    use_ensemble=False         # 使用单一最优模型（更快）
)

# 查看结果
for key, result in results.items():
    if result.get('success'):
        print(f"\n{key}:")
        print(f"  最优模型: {result['best_model']}")
        print(f"  MAPE误差: {result['best_score']:.2f}%")
        print(f"  预测前5天: {result['predictions'][:5]}")

# 导出结果到CSV
forecaster.export_all_results()
print("\n预测结果已保存到 output/ 目录")
```

## 4. 使用你自己的数据

### 准备数据

确保你的CSV或Excel文件包含以下列：

| 列名 | 说明 |
|------|------|
| 下单时间 | 订单日期 |
| 数量 | 销量 |
| 送货专卖店卡号 | 店铺ID |
| 货品名称 | 产品名称 |
| 省 | 省份 |
| 市 | 城市 |

### 运行预测

```python
from src.forecaster import SalesForecaster

# 加载你的数据
forecaster = SalesForecaster(data_path="你的数据.csv")

# 按店铺预测每日销量
daily_results = forecaster.forecast_by_dimension(
    dimension='store_id',
    time_granularity='daily',
    forecast_horizon=30
)

# 按省份预测每周销量
weekly_results = forecaster.forecast_by_dimension(
    dimension='province',
    time_granularity='weekly',
    forecast_horizon=12
)

# 按城市预测每月销量
monthly_results = forecaster.forecast_by_dimension(
    dimension='city',
    time_granularity='monthly',
    forecast_horizon=6
)

# 导出所有结果
forecaster.export_all_results()
```

## 5. 查看结果

预测结果会自动保存到 `output/` 目录，包含：

- `forecast_store_id_daily.csv` - 按店铺的每日预测
- `forecast_province_weekly.csv` - 按省份的每周预测
- `forecast_city_monthly.csv` - 按城市的每月预测

CSV文件包含以下列：
- `date`: 预测日期
- `predicted_quantity`: 预测销量
- `store_id/province/city`: 维度值
- `product`: 产品名称
- `model`: 使用的最优模型
- `time_granularity`: 时间粒度

## 6. 进阶功能

### 使用集成模型（提高准确性）

```python
results = forecaster.forecast_by_dimension(
    dimension='store_id',
    time_granularity='daily',
    forecast_horizon=30,
    use_ensemble=True,  # 启用集成模型
    models=['auto_arima', 'prophet', 'xgboost']
)
```

### 只使用特定模型

```python
results = forecaster.forecast_by_dimension(
    dimension='store_id',
    time_granularity='daily',
    forecast_horizon=30,
    models=['prophet', 'xgboost']  # 只使用这两个模型
)
```

### 批量预测所有维度

```python
all_results = forecaster.forecast_all_dimensions(
    time_granularities=['daily', 'weekly', 'monthly']
)
```

## 7. 常见问题

### Q: 安装依赖时出错怎么办？

A: 某些包（如Prophet）可能需要编译。尝试：

```bash
# 使用conda（如果可用）
conda install -c conda-forge prophet

# 或升级pip
pip install --upgrade pip
pip install -r requirements.txt
```

### Q: 预测速度太慢？

A: 可以：
1. 不使用LSTM模型（最慢）
2. 使用 `use_ensemble=False`
3. 减少要评估的模型数量

```python
results = forecaster.forecast_by_dimension(
    dimension='store_id',
    time_granularity='daily',
    forecast_horizon=30,
    models=['auto_arima', 'xgboost']  # 只用快速模型
)
```

### Q: 数据量太少怎么办？

A: 系统会自动过滤样本数不足的时间序列。建议：
- 日预测：至少60-90天数据
- 周预测：至少12-24周数据
- 月预测：至少6-12个月数据

### Q: 如何评估预测准确性？

A: 系统会自动计算MAPE（平均绝对百分比误差）：
- MAPE < 10%: 非常好
- MAPE 10-20%: 好
- MAPE 20-50%: 可接受
- MAPE > 50%: 需要改进

## 8. 下一步

- 阅读 [USAGE.md](USAGE.md) 了解详细用法
- 查看 [examples/advanced_usage.py](examples/advanced_usage.py) 学习高级功能
- 根据业务需求调整 [src/config.py](src/config.py) 中的参数

## 需要帮助？

- 查看 [README.md](README.md) 了解系统架构
- 查看 [USAGE.md](USAGE.md) 了解详细文档
- 提交 Issue 报告问题

---

祝你使用愉快！

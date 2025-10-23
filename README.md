# SKU销售预测系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

智能SKU销售预测系统，支持**多算法自动选择**、**多维度聚合**、**多时间粒度**的销售预测。

## 📋 目录

- [功能特性](#功能特性)
- [支持的算法](#支持的算法)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [示例](#示例)
- [常见问题](#常见问题)

## ✨ 功能特性

### 🎯 核心功能

1. **多算法自动选择**
   - 自动评估多个预测算法（ARIMA、Prophet、LSTM、XGBoost等）
   - 根据每个产品的销售数据特性，自动选择最优算法
   - 支持集成模型（Ensemble），组合多个算法提高准确性

2. **多维度预测**
   - 按**店铺**（送货专卖店卡号）预测
   - 按**省份**预测
   - 按**城市**预测
   - 每个维度下按产品名称细分

3. **多时间粒度**
   - **日**级别预测
   - **周**级别预测
   - **月**级别预测

4. **智能数据处理**
   - 自动数据清洗和验证
   - 异常值检测和处理
   - 缺失日期填补
   - 自动特征工程

5. **完整的评估体系**
   - 多种评估指标（MAE、RMSE、MAPE、SMAPE）
   - 自动交叉验证
   - 详细的性能报告

## 🤖 支持的算法

| 算法 | 类型 | 适用场景 | 特点 |
|------|------|----------|------|
| **ARIMA** | 统计模型 | 平稳时间序列 | 经典、可解释性强 |
| **Auto-ARIMA** | 统计模型 | 自动参数选择 | 自动优化ARIMA参数 |
| **Prophet** | 统计模型 | 强季节性数据 | Facebook开源，处理节假日效应 |
| **LSTM** | 深度学习 | 复杂非线性模式 | 适合长期依赖关系 |
| **XGBoost** | 机器学习 | 多特征场景 | 梯度提升，高性能 |
| **LightGBM** | 机器学习 | 大规模数据 | 速度快，内存占用低 |

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 2. 安装依赖

```bash
# 克隆仓库
git clone <repository-url>
cd sku-sales-prediction

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 准备数据

数据应包含以下字段：

| 字段名 | 说明 | 示例 |
|--------|------|------|
| 下单时间 | 订单日期 | 2023-01-01 |
| 数量 | 销量 | 100 |
| 送货专卖店卡号 | 店铺ID | STORE001 |
| 货品名称 | 产品名称 | 产品A |
| 省 | 省份 | 广东省 |
| 市 | 城市 | 深圳市 |
| 货品代码 | SKU代码 | SKU001 |
| 送货地址 | 配送地址 | 深圳市南山区 |
| 配送方式 | 配送方式 | 快递 |
| 月份 | 月份 | 2023-01 |

### 4. 运行示例

```python
from src.forecaster import SalesForecaster

# 加载数据
forecaster = SalesForecaster(data_path="data/sales_data.csv")

# 按店铺预测每日销量
results = forecaster.forecast_by_dimension(
    dimension='store_id',        # 按店铺
    time_granularity='daily',    # 每日
    forecast_horizon=30,         # 预测30天
    use_ensemble=False           # 使用单一最优模型
)

# 查看结果
df_result = forecaster.get_forecast_dataframe('store_id', 'daily')
print(df_result.head())

# 导出结果
forecaster.export_all_results()
```

## 📖 使用指南

### 基础预测

```python
from src.forecaster import SalesForecaster

# 初始化预测器
forecaster = SalesForecaster(data_path="your_data.csv", auto_clean=True)

# 按店铺预测
results = forecaster.forecast_by_dimension(
    dimension='store_id',
    time_granularity='daily',
    forecast_horizon=30
)
```

### 多时间粒度预测

```python
# 按周预测
weekly_results = forecaster.forecast_by_dimension(
    dimension='province',
    time_granularity='weekly',
    forecast_horizon=12  # 预测12周
)

# 按月预测
monthly_results = forecaster.forecast_by_dimension(
    dimension='city',
    time_granularity='monthly',
    forecast_horizon=6  # 预测6个月
)
```

### 使用集成模型

```python
# 使用多个模型的加权平均，提高预测准确性
results = forecaster.forecast_by_dimension(
    dimension='store_id',
    time_granularity='daily',
    forecast_horizon=30,
    use_ensemble=True,  # 启用集成
    models=['auto_arima', 'prophet', 'xgboost']  # 指定模型
)
```

### 批量预测所有维度

```python
# 对所有维度和时间粒度进行预测
all_results = forecaster.forecast_all_dimensions(
    time_granularities=['daily', 'weekly', 'monthly'],
    use_ensemble=False
)

# 导出所有结果
forecaster.export_all_results()
```

### 自定义模型选择

```python
from src.model_selector import ModelSelector

# 手动比较模型
selector = ModelSelector(
    models=['auto_arima', 'prophet', 'xgboost'],
    metric='mape'  # 使用MAPE作为评估指标
)

# 选择最优模型
best_model, eval_results = selector.select_best_model(data)

# 查看评估摘要
summary = selector.get_evaluation_summary()
print(summary)

# 使用最优模型预测
predictions = selector.forecast_with_best_model(steps=30)
```

## 📁 项目结构

```
sku-sales-prediction/
├── src/                          # 源代码
│   ├── __init__.py
│   ├── config.py                 # 配置文件
│   ├── utils.py                  # 工具函数
│   ├── data_preprocessing.py     # 数据预处理
│   ├── forecaster.py             # 主预测器
│   ├── model_selector.py         # 模型选择器
│   └── models/                   # 预测模型
│       ├── __init__.py
│       ├── base_model.py         # 基础模型类
│       ├── arima_model.py        # ARIMA模型
│       ├── prophet_model.py      # Prophet模型
│       ├── lstm_model.py         # LSTM模型
│       └── xgboost_model.py      # XGBoost/LightGBM模型
├── examples/                     # 使用示例
│   ├── basic_usage.py            # 基础示例
│   └── advanced_usage.py         # 高级示例
├── data/                         # 数据目录
│   └── sample/                   # 示例数据
├── output/                       # 输出目录
│   └── models/                   # 模型缓存
├── tests/                        # 测试
├── requirements.txt              # 依赖清单
├── setup.py                      # 安装脚本
└── README.md                     # 说明文档
```

## ⚙️ 配置说明

主要配置在 `src/config.py` 中，可以自定义：

### 预测参数

```python
DEFAULT_FORECAST_CONFIG = {
    "test_size": 0.2,              # 测试集比例
    "validation_size": 0.1,        # 验证集比例
    "min_train_samples": 30,       # 最小训练样本数
    "forecast_horizon": {
        "daily": 30,               # 日预测步数
        "weekly": 12,              # 周预测步数
        "monthly": 6               # 月预测步数
    }
}
```

### 模型参数

可以在配置文件中调整各模型的超参数：

- `ARIMA_CONFIG` - ARIMA模型配置
- `PROPHET_CONFIG` - Prophet模型配置
- `LSTM_CONFIG` - LSTM模型配置
- `XGBOOST_CONFIG` - XGBoost模型配置
- `LIGHTGBM_CONFIG` - LightGBM模型配置

## 📊 示例

### 示例1：基础预测

详见 `examples/basic_usage.py`

### 示例2：高级功能

详见 `examples/advanced_usage.py`

包括：
- 自定义数据预处理
- 详细的模型比较
- 批量预测多个产品
- 带置信区间的预测
- 结果可视化

## ❓ 常见问题

### Q1: 如何选择合适的时间粒度？

- **日级别**：适合短期预测（7-30天），需要较多历史数据（建议至少90天）
- **周级别**：适合中期预测（4-12周），对数据量要求适中
- **月级别**：适合长期预测（3-12个月），对数据量要求较低

### Q2: 使用集成模型还是单一最优模型？

- **单一最优模型**：速度快，适合快速预测
- **集成模型**：准确性更高，但计算时间更长，适合对准确性要求高的场景

### Q3: 如何处理数据量不足的情况？

系统会自动过滤样本数不足的时间序列。建议：
- 日级别至少需要30-60天数据
- 周级别至少需要12-24周数据
- 月级别至少需要6-12个月数据

### Q4: 预测结果如何评估？

系统提供多个评估指标：
- **MAE**（平均绝对误差）：越小越好
- **RMSE**（均方根误差）：对大误差更敏感
- **MAPE**（平均绝对百分比误差）：推荐使用，易于理解
- **SMAPE**（对称MAPE）：处理小值时更稳定

### Q5: 如何提高预测准确性？

1. 确保数据质量（完整性、准确性）
2. 提供足够的历史数据
3. 使用集成模型
4. 根据业务特点调整模型参数
5. 定期更新模型

## 📝 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题，请提交Issue或联系开发团队。

---

**注意**: 这是一个销售预测系统，预测结果仅供参考，实际决策需结合业务知识和市场情况。

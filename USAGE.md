# 详细使用文档

本文档详细介绍SKU销售预测系统的使用方法和最佳实践。

## 目录

1. [数据准备](#数据准备)
2. [基础使用](#基础使用)
3. [高级功能](#高级功能)
4. [参数调优](#参数调优)
5. [性能优化](#性能优化)
6. [故障排除](#故障排除)

## 数据准备

### 数据格式要求

系统要求输入数据包含以下必需字段：

| 字段名 | 类型 | 是否必需 | 说明 |
|--------|------|----------|------|
| 下单时间 | datetime | 是 | 订单日期，支持多种格式 |
| 数量 | numeric | 是 | 销量（整数或浮点数） |
| 送货专卖店卡号 | string | 是 | 店铺唯一标识 |
| 货品名称 | string | 是 | 产品名称 |
| 省 | string | 是 | 省份 |
| 市 | string | 是 | 城市 |
| 货品代码 | string | 否 | SKU代码 |
| 送货地址 | string | 否 | 详细地址 |
| 配送方式 | string | 否 | 配送方式 |
| 月份 | string | 否 | 月份标识 |

### 数据质量要求

1. **完整性**
   - 必需字段不能为空
   - 建议至少有3个月的历史数据

2. **准确性**
   - 日期格式一致
   - 销量为非负数
   - 店铺和产品标识唯一且一致

3. **时间范围**
   - 日级别预测：建议至少90天数据
   - 周级别预测：建议至少24周数据
   - 月级别预测：建议至少12个月数据

### 数据示例

CSV格式：

```csv
下单时间,数量,送货专卖店卡号,货品名称,省,市,货品代码,送货地址,配送方式,月份
2023-01-01,100,STORE001,产品A,广东省,深圳市,SKU001,深圳市南山区,快递,2023-01
2023-01-02,120,STORE001,产品A,广东省,深圳市,SKU001,深圳市南山区,快递,2023-01
2023-01-03,95,STORE001,产品A,广东省,深圳市,SKU001,深圳市南山区,快递,2023-01
```

Excel格式同样支持，确保列名一致。

## 基础使用

### 1. 导入和初始化

```python
from src.forecaster import SalesForecaster
from loguru import logger

# 配置日志（可选）
logger.add("logs/forecast.log", rotation="10 MB")

# 从CSV文件加载
forecaster = SalesForecaster(
    data_path="data/sales_data.csv",
    auto_clean=True  # 自动清洗数据
)

# 或从DataFrame加载
import pandas as pd
df = pd.read_csv("data/sales_data.csv")
forecaster = SalesForecaster(df=df, auto_clean=True)
```

### 2. 查看数据摘要

```python
# 获取数据统计摘要
summary = forecaster.get_summary_statistics()
print(summary)

# 输出示例：
# 总记录数        10000
# 总销量          150000
# 平均销量        15.0
# 销量中位数      12.0
# 销量标准差      8.5
# 最小销量        0
# 最大销量        200
# 唯一产品数      50
# 唯一店铺数      20
# 时间跨度        2023-01-01 至 2023-12-31
```

### 3. 单维度预测

#### 按店铺预测

```python
# 按店铺预测每日销量
results = forecaster.forecast_by_dimension(
    dimension='store_id',      # 按店铺聚合
    time_granularity='daily',  # 每日粒度
    forecast_horizon=30,       # 预测30天
    use_ensemble=False,        # 使用单一最优模型
    save_results=True          # 保存到CSV
)

# 查看某个店铺+产品的预测结果
for series_key, result in results.items():
    if result.get('success'):
        print(f"\n{series_key}:")
        print(f"  最优模型: {result['best_model']}")
        print(f"  MAPE: {result['best_score']:.2f}%")
        print(f"  预测值: {result['predictions'][:5]}")
```

#### 按省份预测

```python
# 按省份预测每周销量
results = forecaster.forecast_by_dimension(
    dimension='province',
    time_granularity='weekly',
    forecast_horizon=12  # 预测12周
)
```

#### 按城市预测

```python
# 按城市预测每月销量
results = forecaster.forecast_by_dimension(
    dimension='city',
    time_granularity='monthly',
    forecast_horizon=6  # 预测6个月
)
```

### 4. 获取预测结果

```python
# 获取DataFrame格式的结果
df_result = forecaster.get_forecast_dataframe(
    dimension='store_id',
    time_granularity='daily'
)

print(df_result.head())
# 输出：
#         date  predicted_quantity  store_id  product    model time_granularity
# 0 2024-01-01               120.5  STORE001  产品A  Prophet            daily
# 1 2024-01-02               125.3  STORE001  产品A  Prophet            daily
# ...

# 获取特定时间序列的结果
df_specific = forecaster.get_forecast_dataframe(
    dimension='store_id',
    time_granularity='daily',
    series_key='STORE001_产品A'
)
```

### 5. 导出结果

```python
# 导出所有预测结果到CSV
forecaster.export_all_results()

# 导出到指定目录
forecaster.export_all_results(output_dir='output/my_forecast')
```

## 高级功能

### 1. 使用集成模型

集成模型通过组合多个模型的预测结果，通常能获得更高的准确性。

```python
results = forecaster.forecast_by_dimension(
    dimension='store_id',
    time_granularity='daily',
    forecast_horizon=30,
    use_ensemble=True,  # 启用集成模型
    models=['auto_arima', 'prophet', 'xgboost', 'lightgbm']
)
```

### 2. 自定义模型选择

```python
from src.model_selector import ModelSelector

# 手动控制模型选择过程
selector = ModelSelector(
    models=['auto_arima', 'prophet', 'xgboost'],  # 指定要评估的模型
    metric='mape'  # 评估指标
)

# 准备单个时间序列数据
ts_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=180, freq='D'),
    'quantity': [...]  # 你的销量数据
})

# 选择最优模型
best_model, eval_results = selector.select_best_model(
    ts_data,
    test_size=0.2,
    parallel=True  # 并行评估
)

# 查看评估摘要
summary = selector.get_evaluation_summary()
print(summary)
#     model  success      mae     rmse     mape    smape
# 0  prophet     True    8.45   10.23    12.34    11.89
# 1  xgboost     True    9.12   11.45    13.56    12.45
# 2  auto_arima  True   10.23   12.67    14.78    13.56

# 使用最优模型预测
forecast_result = selector.forecast_with_best_model(steps=30)
print(f"预测模型: {forecast_result['model_name']}")
print(f"预测结果: {forecast_result['predictions']}")
```

### 3. 批量预测所有维度

```python
# 一次性对所有维度和时间粒度进行预测
all_results = forecaster.forecast_all_dimensions(
    time_granularities=['daily', 'weekly', 'monthly'],
    use_ensemble=False,
    models=['auto_arima', 'prophet', 'xgboost']  # 可选：指定模型
)

# 查看所有结果
print(f"完成预测组合数: {len(all_results)}")
# 输出: 完成预测组合数: 9 (3个维度 × 3个时间粒度)

# 导出所有结果
forecaster.export_all_results()
```

### 4. 自定义数据预处理

```python
from src.data_preprocessing import DataPreprocessor

# 手动控制预处理流程
preprocessor = DataPreprocessor(data_path="data/sales_data.csv")

# 验证数据列
if not preprocessor.validate_columns():
    print("数据列验证失败")
    exit()

# 清洗数据
cleaned_data = preprocessor.clean_data(
    remove_outliers=True  # 是否移除异常值
)

# 准备特定维度的预测数据
forecast_data = preprocessor.prepare_forecast_data(
    dimension='store_id',
    time_granularity='daily',
    min_samples=60  # 自定义最小样本数要求
)

print(f"准备好 {len(forecast_data)} 个时间序列")
```

### 5. 使用特定模型

```python
from src.models import AutoARIMAModel, ProphetModel, XGBoostModel

# 创建并训练特定模型
model = ProphetModel()

# 准备数据
ts_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=180, freq='D'),
    'quantity': [...]
})

# 训练
model.fit(ts_data)

# 预测
predictions = model.predict(steps=30)
print(predictions)

# 对于Prophet模型，可以获取置信区间
if isinstance(model, ProphetModel):
    forecast_components = model.get_forecast_components(steps=30)
    print(forecast_components[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
```

## 参数调优

### 1. 调整预测时间范围

在 `src/config.py` 中修改：

```python
DEFAULT_FORECAST_CONFIG = {
    "forecast_horizon": {
        "daily": 60,    # 改为预测60天
        "weekly": 24,   # 改为预测24周
        "monthly": 12   # 改为预测12个月
    }
}
```

或在调用时指定：

```python
results = forecaster.forecast_by_dimension(
    dimension='store_id',
    time_granularity='daily',
    forecast_horizon=60  # 直接指定
)
```

### 2. 调整模型参数

修改 `src/config.py` 中的模型配置：

```python
# Prophet模型参数
PROPHET_CONFIG = {
    "changepoint_prior_scale": 0.1,  # 增加以捕捉更多趋势变化
    "seasonality_prior_scale": 15,   # 增加以捕捉更强的季节性
    "seasonality_mode": "multiplicative"  # 或 "additive"
}

# XGBoost模型参数
XGBOOST_CONFIG = {
    "n_estimators": 200,     # 增加树的数量
    "max_depth": 8,          # 增加树的深度
    "learning_rate": 0.05,   # 降低学习率
}
```

### 3. 调整数据质量阈值

```python
# 调整最小样本数要求
forecast_data = preprocessor.prepare_forecast_data(
    dimension='store_id',
    time_granularity='daily',
    min_samples=90  # 要求至少90天数据
)

# 调整测试集比例
selector = ModelSelector()
best_model, _ = selector.select_best_model(
    data,
    test_size=0.3  # 使用30%作为测试集
)
```

## 性能优化

### 1. 并行处理

```python
# 模型选择时启用并行
selector = ModelSelector()
best_model, _ = selector.select_best_model(
    data,
    parallel=True  # 启用并行评估
)
```

### 2. 限制模型数量

```python
# 只使用快速的模型
results = forecaster.forecast_by_dimension(
    dimension='store_id',
    time_granularity='daily',
    forecast_horizon=30,
    models=['auto_arima', 'xgboost']  # 排除LSTM等慢速模型
)
```

### 3. 批量处理策略

```python
# 分批处理大量时间序列
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(data_path="large_data.csv")
preprocessor.clean_data()

# 获取所有时间序列
all_series = preprocessor.prepare_forecast_data(
    dimension='store_id',
    time_granularity='daily'
)

# 分批处理
batch_size = 10
series_keys = list(all_series.keys())

for i in range(0, len(series_keys), batch_size):
    batch_keys = series_keys[i:i+batch_size]
    print(f"处理批次 {i//batch_size + 1}")

    for key in batch_keys:
        ts_data = all_series[key]
        # 处理单个时间序列
        selector = ModelSelector(models=['auto_arima', 'xgboost'])
        best_model, _ = selector.select_best_model(ts_data)
        predictions = best_model.predict(30)
        # 保存结果...
```

## 故障排除

### 常见问题及解决方案

#### 1. 数据加载失败

**问题**: `ValueError: Must provide either data_path or df`

**解决**: 确保传入了数据文件路径或DataFrame

```python
# 正确的用法
forecaster = SalesForecaster(data_path="data/sales.csv")
# 或
forecaster = SalesForecaster(df=my_dataframe)
```

#### 2. 列名不匹配

**问题**: 数据列验证失败

**解决**: 检查数据列名是否与配置匹配

```python
# 方法1：修改数据列名
df = df.rename(columns={
    'order_date': '下单时间',
    'qty': '数量',
    'store': '送货专卖店卡号',
    # ...
})

# 方法2：修改config.py中的DATA_COLUMNS配置
```

#### 3. 样本数不足

**问题**: 时间序列被过滤掉

**解决**: 降低最小样本数要求或提供更多数据

```python
forecast_data = preprocessor.prepare_forecast_data(
    dimension='store_id',
    time_granularity='daily',
    min_samples=20  # 降低要求
)
```

#### 4. 模型训练失败

**问题**: 所有模型评估都失败

**解决**:
1. 检查数据质量（是否有足够的样本）
2. 尝试只使用稳定的模型

```python
# 使用更稳定的模型
results = forecaster.forecast_by_dimension(
    dimension='store_id',
    time_granularity='daily',
    models=['auto_arima', 'prophet']  # 排除可能不稳定的模型
)
```

#### 5. 内存不足

**问题**: 处理大量数据时内存溢出

**解决**:
1. 分批处理
2. 减少并行数
3. 不使用LSTM等内存密集型模型

```python
# 限制并行数
import os
os.environ['OMP_NUM_THREADS'] = '2'

# 不使用LSTM
models = ['auto_arima', 'prophet', 'xgboost']  # 排除LSTM
```

#### 6. Prophet安装问题

**问题**: Prophet安装失败

**解决**:

```bash
# 先安装依赖
pip install pystan
pip install prophet

# 如果还是失败，尝试conda
conda install -c conda-forge prophet
```

## 最佳实践

1. **数据准备**
   - 确保数据质量和完整性
   - 定期更新历史数据
   - 处理异常值和缺失值

2. **模型选择**
   - 先用小数据集测试
   - 使用交叉验证评估
   - 根据业务特点选择合适的时间粒度

3. **预测监控**
   - 定期评估预测准确性
   - 记录模型性能指标
   - 根据实际情况调整模型

4. **结果应用**
   - 结合业务知识解读结果
   - 设置合理的置信区间
   - 预测结果作为决策参考而非唯一依据

---

更多问题请查看项目GitHub Issues或联系技术支持。

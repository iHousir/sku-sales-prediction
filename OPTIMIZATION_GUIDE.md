# 预测速度优化指南

如果预测运行时间太长，可以通过以下方法优化。

## 立即停止运行中的程序

在命令行窗口按：
```
Ctrl + C
```

## 快速预测脚本（推荐）

使用优化后的快速预测脚本：

```cmd
python run_forecast_fast.py
```

这个脚本已经预配置了最优的速度设置。

## 优化策略对比

### 速度提升倍数估算

| 优化方法 | 速度提升 | 说明 |
|---------|---------|------|
| 只用2个快速模型 | 2-3倍 | Prophet + XGBoost |
| 只预测1个维度 | 3倍 | 只预测店铺，不预测省市 |
| 只预测1个粒度 | 3倍 | 只预测每日，不预测周月 |
| 提高最小样本数 | 1.5-2倍 | 过滤掉数据少的序列 |
| 关闭LSTM | 5-10倍 | LSTM最慢 |
| **组合使用** | **10-30倍** | 推荐 |

## 具体优化方案

### 方案1：最快速度（推荐用于测试）

只预测最重要的：按店铺预测每日销量

修改 `run_forecast_fast.py`：

```python
FORECAST_CONFIG = {
    # 只预测店铺
    "forecast_by_store": True,
    "forecast_by_province": False,
    "forecast_by_city": False,

    # 只预测每日
    "forecast_daily": True,
    "forecast_weekly": False,
    "forecast_monthly": False,

    # 只预测7天（而不是30天）
    "daily_days": 7,

    # 只用最快的模型
    "models": ['xgboost'],  # 单一模型

    # 提高最小样本要求
    "min_samples": 90,  # 只预测数据充足的
}
```

**预期效果**：原来需要1小时，现在可能只需5-10分钟

### 方案2：平衡模式（推荐用于生产）

平衡速度和准确性

```python
FORECAST_CONFIG = {
    # 预测店铺和省份
    "forecast_by_store": True,
    "forecast_by_province": True,
    "forecast_by_city": False,

    # 预测每日和每周
    "forecast_daily": True,
    "forecast_weekly": True,
    "forecast_monthly": False,

    # 标准预测时长
    "daily_days": 30,
    "weekly_weeks": 12,

    # 用2个快速模型
    "models": ['prophet', 'xgboost'],

    # 标准样本要求
    "min_samples": 60,
}
```

**预期效果**：原来需要1小时，现在约20-30分钟

### 方案3：完整预测（最准确但最慢）

如果时间充足，需要最准确的结果

```python
FORECAST_CONFIG = {
    # 预测所有维度
    "forecast_by_store": True,
    "forecast_by_province": True,
    "forecast_by_city": True,

    # 预测所有粒度
    "forecast_daily": True,
    "forecast_weekly": True,
    "forecast_monthly": True,

    # 使用集成模型
    "use_ensemble": True,

    # 用多个模型
    "models": ['auto_arima', 'prophet', 'xgboost', 'lightgbm'],

    "min_samples": 30,
}
```

**预期效果**：可能需要2-4小时（取决于数据量）

## 模型速度对比

### 各模型速度排名（从快到慢）

1. **XGBoost** ⚡⚡⚡ - 最快
2. **LightGBM** ⚡⚡⚡ - 最快
3. **Prophet** ⚡⚡ - 快
4. **Auto-ARIMA** ⚡ - 中等
5. **ARIMA** ⚡ - 中等
6. **LSTM** 🐌 - 最慢（慢5-10倍）

### 推荐组合

**最快组合**：
```python
"models": ['xgboost']
```

**快速且准确**：
```python
"models": ['prophet', 'xgboost']
```

**平衡组合**：
```python
"models": ['prophet', 'xgboost', 'lightgbm']
```

**不推荐**（太慢）：
```python
"models": ['lstm', 'arima']  # 避免使用
```

## 数据层面优化

### 1. 减少预测的时间序列数量

如果你有很多店铺和产品组合，可以：

**方法A：提高最小样本数**
```python
"min_samples": 90,  # 只预测数据充足的产品
```

**方法B：预处理数据，只保留重要的**

在加载数据前先过滤：

```python
import pandas as pd

# 读取数据
df = pd.read_csv(DATA_PATH, encoding='gbk')

# 只保留销量前80%的产品
product_sales = df.groupby('货品代码')['数量'].sum()
top_products = product_sales.nlargest(int(len(product_sales) * 0.8)).index
df = df[df['货品代码'].isin(top_products)]

# 只保留主要店铺
main_stores = df['送货专卖店卡号'].value_counts().head(20).index
df = df[df['送货专卖店卡号'].isin(main_stores)]

# 然后再创建预测器
forecaster = SalesForecaster(df=df, auto_clean=True)
```

### 2. 缩短预测时长

```python
# 从预测30天改为7天
"daily_days": 7,

# 从预测12周改为4周
"weekly_weeks": 4,

# 从预测6个月改为3个月
"monthly_months": 3,
```

## 实际应用建议

### 第一次运行（测试）

使用**方案1**快速测试：
```cmd
# 修改 run_forecast_fast.py 为方案1的配置
python run_forecast_fast.py
```

预期5-10分钟完成，验证：
1. 数据能否正常加载
2. 结果格式是否正确
3. 评估大概需要多少时间

### 正式运行（生产）

根据测试结果选择：

- **如果测试很快（<10分钟）** → 使用方案3（完整预测）
- **如果测试较慢（10-30分钟）** → 使用方案2（平衡模式）
- **如果测试很慢（>30分钟）** → 继续使用方案1，或进一步优化

### 定期预测

如果需要每天/每周定期预测：

```python
# 只预测最近变化的数据
# 只预测未来7天（而不是30天）
"daily_days": 7,

# 只用最快的模型
"models": ['xgboost'],
```

## 进度监控

运行时查看进度：

```
3/4 执行预测

按店铺预测:
  [1/3] 每日预测（未来30天）... ✓ (45/50)  ← 当前进度
  [2/3] 每周预测（未来12周）...
  [3/3] 每月预测（未来6个月）...
```

显示：
- 当前任务进度 [1/3]
- 成功的时间序列数量 (45/50)

## 命令行快速切换

创建不同的配置文件：

```cmd
# 测试模式（最快）
python run_forecast_fast.py

# 标准模式
python run_forecast.py

# 自定义配置
# 修改配置文件后运行
```

## 常见问题

### Q: 如何估算预测需要多长时间？

**A**: 粗略估算：
```
时间 ≈ (店铺数 × 产品数) × (维度数 × 粒度数) × 模型数 × 5秒
```

例如：
- 20个店铺
- 50个产品
- 3个维度 × 3个粒度 = 9个任务
- 3个模型

```
时间 ≈ (20 × 50) × 9 × 3 × 5秒 = 135,000秒 ≈ 37.5小时
```

**优化后**（只预测1个维度、1个粒度、1个模型）：
```
时间 ≈ (20 × 50) × 1 × 1 × 5秒 = 5,000秒 ≈ 1.4小时
```

### Q: 能否并行运行加快速度？

**A**: 目前已经在模型评估阶段使用了并行（parallel=True）。如果要进一步加快：

1. 手动拆分数据，分批预测不同的店铺/产品
2. 在多台机器上并行运行

### Q: 预测结果准确性会降低吗？

**A**:
- **减少模型数量**：略微降低（5-10%）
- **只用快速模型**：基本不降低（XGBoost和Prophet已经很准确）
- **减少预测时长**：不影响准确性
- **提高最小样本数**：反而可能提高准确性（过滤掉不稳定的）

## 总结

### 推荐的快速配置

```python
FORECAST_CONFIG = {
    "forecast_by_store": True,    # 只预测店铺
    "forecast_by_province": False,
    "forecast_by_city": False,

    "forecast_daily": True,       # 只预测每日
    "forecast_weekly": False,
    "forecast_monthly": False,

    "daily_days": 30,
    "models": ['prophet', 'xgboost'],  # 2个快速模型
    "min_samples": 60,
}
```

**运行**：
```cmd
python run_forecast_fast.py
```

**预期**：比原始版本快10-20倍

---

更新日期：2024-10-23

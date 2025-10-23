# 如何使用 - 最简单的方式

本文档提供最简单直接的使用方法。

## Windows用户快速开始

### 第1步：准备数据

确保你的CSV文件包含以下字段：

```csv
下单时间,数量,送货专卖店卡号,货品代码,省,市
2023-01-01,100,STORE001,SKU001,广东省,深圳市
2023-01-02,120,STORE001,SKU001,广东省,深圳市
```

### 第2步：修改配置

打开 `run_forecast.py` 文件，修改第13行的数据路径：

```python
# 修改这里为你的CSV文件路径
DATA_PATH = r"E:\03 Python xm-1\嘉兴数据V1.csv"
```

**注意**：
- Windows路径前面加 `r`，例如：`r"E:\数据\文件.csv"`
- 或者使用双反斜杠：`"E:\\数据\\文件.csv"`

### 第3步：运行预测

打开命令提示符（CMD）或PowerShell：

```cmd
# 进入项目目录
cd /d E:\03 Python xm-1

# 激活虚拟环境（如果使用虚拟环境）
sku_forecast_venv\Scripts\activate

# 运行预测
python run_forecast.py
```

### 第4步：查看结果

预测完成后，结果保存在 `output/` 目录下：

```
output/
├── forecast_store_id_daily.csv      # 按店铺的每日预测
├── forecast_store_id_weekly.csv     # 按店铺的每周预测
├── forecast_store_id_monthly.csv    # 按店铺的每月预测
├── forecast_province_daily.csv      # 按省份的每日预测
├── forecast_province_weekly.csv     # 按省份的每周预测
├── forecast_province_monthly.csv    # 按省份的每月预测
├── forecast_city_daily.csv          # 按城市的每日预测
├── forecast_city_weekly.csv         # 按城市的每周预测
└── forecast_city_monthly.csv        # 按城市的每月预测
```

用Excel或其他工具打开这些CSV文件即可查看预测结果。

## 配置说明

### 基础配置

打开 `run_forecast.py`，在开头的配置区域可以修改：

```python
FORECAST_CONFIG = {
    # 预测哪些维度
    "forecast_by_store": True,      # 是否按店铺预测
    "forecast_by_province": True,   # 是否按省份预测
    "forecast_by_city": True,       # 是否按城市预测

    # 预测多长时间
    "store_daily_days": 30,         # 按店铺预测未来30天
    "store_weekly_weeks": 12,       # 按店铺预测未来12周
    "store_monthly_months": 6,      # 按店铺预测未来6个月

    # 模型配置
    "use_ensemble": False,          # False=更快, True=更准确
    "models": ['auto_arima', 'prophet', 'xgboost'],
}
```

### 只预测特定维度

如果你只需要某些预测，可以关闭不需要的：

```python
FORECAST_CONFIG = {
    "forecast_by_store": True,      # 只预测店铺
    "forecast_by_province": False,  # 不预测省份
    "forecast_by_city": False,      # 不预测城市
}
```

### 调整预测时长

```python
# 只预测未来7天、4周、3个月
"store_daily_days": 7,
"store_weekly_weeks": 4,
"store_monthly_months": 3,
```

### 使用集成模型（更准确但更慢）

```python
"use_ensemble": True,  # 启用集成模型
```

## 常见问题

### Q1: 命令行显示"不是内部或外部命令"

**原因**：你在CMD中直接输入了Python代码。

**解决**：Python代码不能在CMD中直接运行，需要：
1. 保存为 `.py` 文件
2. 用 `python 文件名.py` 运行

### Q2: 提示找不到数据文件

检查：
1. 文件路径是否正确（Windows要用 `r"路径"` 或 `"路径\\文件"`）
2. 文件是否存在
3. 是否在正确的目录运行命令

### Q3: 提示缺少模块

安装依赖：

```cmd
pip install -r requirements.txt
```

### Q4: CSV文件编码问题

如果中文显示乱码，尝试：

```python
# 在 run_forecast.py 中修改：
import pandas as pd
df = pd.read_csv(DATA_PATH, encoding='gbk')  # 或 'utf-8-sig'
forecaster = SalesForecaster(df=df, auto_clean=True)
```

### Q5: 数据验证失败

确保CSV包含必需的字段：
- 下单时间
- 数量
- 送货专卖店卡号
- 货品代码
- 省
- 市

### Q6: 预测速度慢

减少要评估的模型：

```python
"models": ['prophet', 'xgboost'],  # 只用2个快速模型
```

或关闭部分预测维度：

```python
"forecast_by_province": False,  # 不预测省份
"forecast_by_city": False,      # 不预测城市
```

## Linux/Mac 用户

```bash
# 进入项目目录
cd /path/to/sku-sales-prediction

# 激活虚拟环境（如果使用）
source venv/bin/activate

# 运行预测
python run_forecast.py
```

路径格式：

```python
DATA_PATH = "/home/user/data/sales.csv"  # Linux/Mac直接用正斜杠
```

## 预测结果说明

CSV输出文件包含以下列：

| 列名 | 说明 |
|------|------|
| date | 预测日期 |
| predicted_quantity | 预测销量 |
| store_id / province / city | 维度值（店铺/省/市） |
| product | 产品（货品代码） |
| model | 使用的最优模型 |
| time_granularity | 时间粒度（daily/weekly/monthly） |

示例：

```csv
date,predicted_quantity,store_id,product,model,time_granularity
2024-01-01,120.5,STORE001,SKU001,Prophet,daily
2024-01-02,125.3,STORE001,SKU001,Prophet,daily
```

## 进阶使用

如需更灵活的控制，参考：
- `examples/basic_usage.py` - 基础示例
- `examples/advanced_usage.py` - 高级示例
- `USAGE.md` - 详细使用文档

## 需要帮助？

- 查看 `README.md` - 完整文档
- 查看 `DATA_FORMAT_UPDATE.md` - 数据格式说明
- 提交 GitHub Issue

---

最后更新：2024-10-23

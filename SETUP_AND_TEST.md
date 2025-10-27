# 项目设置与测试指南

本文档提供针对您当前环境的详细设置和测试步骤。

## 您的环境信息

- **数据路径**: `E:\04 数据分析Python`
- **虚拟环境**: `E:\04 数据分析Python\全部门店20230101-20251020订单明细【清洗】\env_filter\Scripts`
- **项目路径**: `C:\Users\fang_hou\Documents\GitHub\sku-sales-prediction`

## 快速开始（3步完成）

### 第1步：激活虚拟环境

打开命令提示符（CMD）或PowerShell，运行：

```cmd
E:\04 数据分析Python\全部门店20230101-20251020订单明细【清洗】\env_filter\Scripts\activate
```

成功后，命令行前面会出现 `(env_filter)` 标识。

### 第2步：进入项目目录

```cmd
cd C:\Users\fang_hou\Documents\GitHub\sku-sales-prediction
```

### 第3步：运行预测

我已为您创建了专用的测试脚本 `run_test.py`，它已经配置好您的数据路径。

#### 方式A：快速测试（推荐先用这个）

```cmd
python run_test.py
```

这个脚本会：
- 自动使用您的数据路径 `E:\04 数据分析Python`
- 自动检测CSV文件（查找 `.csv` 后缀的文件）
- 按省份进行每月预测（最快的配置）
- 预测未来6个月

#### 方式B：使用原有的快速预测脚本

如果您想手动指定具体的CSV文件：

```cmd
python run_forecast_fast.py
```

**注意**：需要先修改 `run_forecast_fast.py` 第15行的数据路径为您的CSV文件完整路径。

## 数据文件说明

### 自动检测CSV文件

`run_test.py` 会在 `E:\04 数据分析Python` 目录下查找CSV文件：
- 如果只有1个CSV文件，自动使用它
- 如果有多个CSV文件，会列出所有文件让您选择
- 如果没有CSV文件，会提示错误

### 手动指定CSV文件

如果您想直接指定某个CSV文件，修改 `run_test.py` 第14行：

```python
# 修改为您的具体CSV文件路径
DATA_DIR = r"E:\04 数据分析Python\您的文件名.csv"
```

例如：
```python
DATA_DIR = r"E:\04 数据分析Python\全部门店20230101-20251020订单明细【清洗】.csv"
```

## 数据格式要求

确保您的CSV文件包含以下必需字段：

| 字段名 | 说明 |
|--------|------|
| 下单时间 | 订单日期 |
| 数量 | 销量（可以为负数表示退货） |
| 送货专卖店卡号 | 店铺ID |
| 货品代码 | SKU代码 |
| 省 | 省份 |
| 市 | 城市 |

其他字段可选，不影响预测。

## 安装依赖（如果需要）

如果运行时提示缺少模块，在虚拟环境中安装依赖：

```cmd
# 确保已激活虚拟环境
pip install -r requirements.txt
```

主要依赖包括：
- pandas
- numpy
- prophet
- xgboost
- scikit-learn
- loguru

## 预测输出

### 输出位置

预测结果保存在：
```
C:\Users\fang_hou\Documents\GitHub\sku-sales-prediction\output\
```

### 输出文件

当前配置（按省份每月预测）会生成：

```
output/
└── forecast_province_monthly.csv
```

### 输出内容

CSV文件包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| date | 预测月份 | 2024-11-01 |
| predicted_quantity | 预测销量 | 1250.5 |
| province | 省份 | 浙江省 |
| product | 产品代码 | SKU001 |
| model | 使用的模型 | Prophet |
| time_granularity | 时间粒度 | monthly |

### 查看结果

用Excel打开 `output/forecast_province_monthly.csv` 即可查看预测结果。

## 常见问题排查

### 问题1：虚拟环境激活失败

**症状**：提示"无法加载文件"或"禁止运行脚本"

**解决方案**：

方法1（推荐）- 使用activate.bat：
```cmd
E:\04 数据分析Python\全部门店20230101-20251020订单明细【清洗】\env_filter\Scripts\activate.bat
```

方法2 - 修改PowerShell执行策略（需要管理员权限）：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 问题2：找不到数据文件

**症状**：提示"找不到数据文件"

**解决方案**：
1. 检查 `E:\04 数据分析Python` 目录是否存在
2. 检查该目录下是否有 `.csv` 文件
3. 如果CSV在子目录，修改 `run_test.py` 的 `DATA_DIR` 路径

### 问题3：CSV编码问题

**症状**：中文显示乱码或无法读取

**解决方案**：
- `run_test.py` 已包含自动编码检测（GBK、UTF-8等）
- 如果仍有问题，用Excel打开CSV，另存为"UTF-8 CSV"格式

### 问题4：缺少依赖包

**症状**：提示"ModuleNotFoundError: No module named 'xxx'"

**解决方案**：
```cmd
# 激活虚拟环境后
pip install -r requirements.txt
```

### 问题5：预测速度慢

**症状**：程序运行时间超过30分钟

**解决方案**：

当前配置已经是最快的（只预测省份×月份），如果还是慢，可以：

1. 减少预测月数，修改 `run_test.py` 第27行：
```python
"monthly_months": 3,  # 从6个月改为3个月
```

2. 只使用一个模型，修改第30行：
```python
"models": ['xgboost'],  # 只用XGBoost
```

3. 提高最小样本要求，修改第33行：
```python
"min_samples": 90,  # 从60改为90，过滤数据少的产品
```

### 问题6：内存不足

**症状**：程序崩溃或提示内存错误

**解决方案**：

1. 预处理数据，只保留主要产品：

修改 `run_test.py`，在创建预测器之前添加：

```python
# 只保留销量前80%的产品
product_sales = df.groupby('货品代码')['数量'].sum()
top_products = product_sales.nlargest(int(len(product_sales) * 0.8)).index
df = df[df['货品代码'].isin(top_products)]

# 只保留主要省份（如果需要）
main_provinces = df['省'].value_counts().head(10).index
df = df[df['省'].isin(main_provinces)]
```

## 预测配置说明

### 当前配置（run_test.py）

```python
FORECAST_CONFIG = {
    # 预测维度：只预测省份
    "forecast_by_store": False,
    "forecast_by_province": True,    # ✓
    "forecast_by_city": False,

    # 时间粒度：只预测月度
    "forecast_daily": False,
    "forecast_weekly": False,
    "forecast_monthly": True,        # ✓

    # 预测时长
    "monthly_months": 6,             # 预测6个月

    # 模型配置
    "use_ensemble": False,           # 单模型（更快）
    "models": ['prophet', 'xgboost'], # 2个最快的模型
    "min_samples": 60,               # 最少60个历史数据点

    # 输出
    "output_dir": "output",
}
```

### 如何修改配置

如果您需要不同的预测：

#### 按店铺预测
```python
"forecast_by_store": True,
"forecast_by_province": False,
```

#### 按城市预测
```python
"forecast_by_city": True,
"forecast_by_province": False,
```

#### 每日预测
```python
"forecast_daily": True,
"forecast_monthly": False,
"daily_days": 30,  # 预测30天
```

#### 使用更多模型（更准确但更慢）
```python
"models": ['prophet', 'xgboost', 'lightgbm', 'auto_arima'],
```

#### 使用集成模型（最准确但最慢）
```python
"use_ensemble": True,
```

## 完整运行示例

```cmd
REM 1. 打开CMD

REM 2. 激活虚拟环境
E:\04 数据分析Python\全部门店20230101-20251020订单明细【清洗】\env_filter\Scripts\activate.bat

REM 3. 进入项目目录
cd C:\Users\fang_hou\Documents\GitHub\sku-sales-prediction

REM 4. 运行预测
python run_test.py

REM 5. 等待完成，查看结果
explorer output
```

## 预期运行时间

根据数据量大小：

| 数据规模 | 预期时间 |
|---------|---------|
| 小（<10万条记录） | 5-10分钟 |
| 中（10-50万条记录） | 10-30分钟 |
| 大（>50万条记录） | 30-60分钟 |

如果超过60分钟，参考"问题5：预测速度慢"进行优化。

## 进度监控

运行时会显示进度信息：

```
================================================================================
1/4 数据加载
================================================================================
尝试 gbk 编码... ✓ 成功
✓ 数据验证成功

================================================================================
2/4 数据摘要
================================================================================
  总记录数: 150000
  唯一产品数: 50
  时间跨度: 2023-01-01 至 2025-10-20

================================================================================
3/4 执行预测
================================================================================

按省份预测:
  [1/1] 每月预测（未来6月）... ✓ (45/50)

================================================================================
4/4 完成
================================================================================
  预测任务: 1
  时间序列: 50
  成功: 45
  成功率: 90.0%

✓ 结果已保存到 output/ 目录
```

## 停止运行

如果需要中断运行，在命令行窗口按：
```
Ctrl + C
```

## 更多帮助

- 详细使用文档：`HOW_TO_USE.md`
- 性能优化指南：`OPTIMIZATION_GUIDE.md`
- 数据格式说明：`DATA_FORMAT_UPDATE.md`
- 完整文档：`README.md`

## 技术支持

如遇问题，请提供以下信息：
1. 错误信息截图
2. 数据文件的列名（前几列即可）
3. 数据文件大小和记录数
4. Python版本（运行 `python --version`）

---

更新日期：2025-10-27

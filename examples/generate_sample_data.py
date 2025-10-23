"""
生成示例数据
用于快速测试销售预测系统
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(
    start_date='2023-01-01',
    days=365,
    num_stores=3,
    num_products=5,
    output_file='data/sample/sales_data.csv'
):
    """
    生成示例销售数据

    Args:
        start_date: 开始日期
        days: 天数
        num_stores: 店铺数量
        num_products: 产品数量
        output_file: 输出文件路径
    """
    print(f"生成示例数据: {days}天, {num_stores}个店铺, {num_products}个产品")

    # 生成日期范围
    dates = pd.date_range(start=start_date, periods=days, freq='D')

    # 店铺和产品配置
    stores = [
        {'id': 'STORE001', 'name': '旗舰店', 'province': '广东省', 'city': '深圳市', 'district': '南山区'},
        {'id': 'STORE002', 'name': '华南分店', 'province': '广东省', 'city': '广州市', 'district': '天河区'},
        {'id': 'STORE003', 'name': '华东分店', 'province': '上海市', 'city': '上海市', 'district': '浦东新区'},
    ][:num_stores]

    products = [
        {'code': 'SKU001', 'name': '经典款T恤', 'base_price': 100},
        {'code': 'SKU002', 'name': '时尚连衣裙', 'base_price': 200},
        {'code': 'SKU003', 'name': '运动鞋', 'base_price': 300},
        {'code': 'SKU004', 'name': '休闲裤', 'base_price': 150},
        {'code': 'SKU005', 'name': '针织衫', 'base_price': 180},
    ][:num_products]

    delivery_methods = ['快递', '自提', '同城配送']

    # 生成数据
    data_list = []

    for date in dates:
        day_of_week = date.dayofweek
        day_of_year = date.dayofyear
        is_weekend = day_of_week >= 5

        for store in stores:
            # 每个店铺的基础销量不同
            store_factor = hash(store['id']) % 50 + 50

            for product in products:
                # 产品基础销量
                product_base = product['base_price'] / 10

                # 趋势：整体上升
                trend = day_of_year * 0.05

                # 周季节性（周末销量更高）
                weekly_seasonal = 20 if is_weekend else 0

                # 月季节性（月初月末销量较高）
                day_of_month = date.day
                if day_of_month <= 5 or day_of_month >= 25:
                    monthly_seasonal = 15
                else:
                    monthly_seasonal = 0

                # 年季节性（模拟夏季和冬季销售高峰）
                yearly_seasonal = 30 * np.sin(2 * np.pi * day_of_year / 365)

                # 随机噪声
                noise = np.random.normal(0, 10)

                # 计算销量
                quantity = max(0, int(
                    store_factor +
                    product_base +
                    trend +
                    weekly_seasonal +
                    monthly_seasonal +
                    yearly_seasonal +
                    noise
                ))

                # 特殊事件（促销日）
                if day_of_month == 18:  # 每月18日促销
                    quantity = int(quantity * 1.5)

                # 生成记录
                record = {
                    '下单时间': date.strftime('%Y-%m-%d'),
                    '数量': quantity,
                    '送货专卖店卡号': store['id'],
                    '货品名称': product['name'],
                    '省': store['province'],
                    '市': store['city'],
                    '货品代码': product['code'],
                    '送货地址': f"{store['province']}{store['city']}{store['district']}",
                    '配送方式': np.random.choice(delivery_methods),
                    '月份': date.strftime('%Y-%m')
                }

                data_list.append(record)

    # 创建DataFrame
    df = pd.DataFrame(data_list)

    # 保存到文件
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n示例数据已生成:")
    print(f"  文件: {output_file}")
    print(f"  总记录数: {len(df):,}")
    print(f"  日期范围: {df['下单时间'].min()} 至 {df['下单时间'].max()}")
    print(f"  店铺数: {df['送货专卖店卡号'].nunique()}")
    print(f"  产品数: {df['货品名称'].nunique()}")
    print(f"  总销量: {df['数量'].sum():,}")
    print(f"\n数据预览:")
    print(df.head(10))

    return df


if __name__ == "__main__":
    # 生成示例数据
    df = generate_sample_data(
        start_date='2023-01-01',
        days=365,  # 一年的数据
        num_stores=3,
        num_products=5,
        output_file='data/sample/sales_data.csv'
    )

    print("\n" + "="*80)
    print("示例数据生成完成！")
    print("="*80)
    print("\n现在可以运行预测示例:")
    print("  python examples/basic_usage.py")
    print("或")
    print("  python examples/advanced_usage.py")

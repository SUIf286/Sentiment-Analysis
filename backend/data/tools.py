import pandas as pd

# 读取CSV文件
input_file = './test/test_clean.csv'  # 替换为您的CSV文件路径
output_file = 'test.csv'  # 指定保存修改后数据集的路径
target_cols=['Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find','Service#Queue', 'Service#Hospitality','Service#Parking','Service#Timely','Price#Level','Price#Cost_effective','Price#Discount','Ambience#Decoration','Ambience#Noise','Ambience#Space','Ambience#Sanitary','Food#Portion','Food#Taste','Food#Appearance','Food#Recommend'
]

# 读取数据
df = pd.read_csv(input_file)

# 检查并转换标签值
for col in target_cols:
    if col in df.columns:
        # 对标签列中的每个值加1
        df[col] = df[col] + 1
    else:
        print(f"Warning: Column {col} not found in the dataset.")

# 保存修改后的数据集到新的CSV文件
df.to_csv(output_file, index=False)

print(f"Data processing completed. Modified data saved to {output_file}")
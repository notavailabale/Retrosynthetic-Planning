import argparse
import pandas as pd
import numpy as np
import gzip
from sklearn.linear_model import LinearRegression
import tensorflow as tf

# 加载训练集数据
with gzip.open('train.pkl.gz', 'rb') as f:
    train_data = pd.read_pickle(f)

# 加载测试集数据
with gzip.open('test.pkl.gz', 'rb') as f:
    test_data = pd.read_pickle(f)

# 提取特征和目标值
X_train = np.array([np.unpackbits(fp) for fp in train_data['packed_fp']])
y_train = train_data['values']

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测单个分子的成本
def predict_cost(molecule_fp):
    molecule_fp = np.unpackbits(molecule_fp)
    cost = model.predict([molecule_fp])
    return cost[0]

# 预测多个分子的总成本
def predict_total_cost(molecule_fps):
    total_cost = 0
    for molecule_fp in molecule_fps:
        cost = predict_cost(molecule_fp)
        total_cost += cost
    return total_cost

# 解析命令行参数
parser = argparse.ArgumentParser(description='Predict cost for given molecules.')
parser.add_argument('-m', '--molecules', nargs='+', type=int, required=True, help='List of molecule IDs')
args = parser.parse_args()

# 获取需要预测的分子编号
molecule_ids = args.molecules

# 预测分子的指纹
molecule_fps = [test_data['packed_fp'][mol_id] for mol_id in molecule_ids]

# 预测多个分子的总成本
predicted_total_cost = predict_total_cost(molecule_fps)

# 将 TensorFlow 张量转换为 NumPy 数组
values = test_data['values'].numpy()

# 获取实际的成本值
actual_total_cost = sum(values[molecule_ids])

print("Predicted total cost for the molecules:", predicted_total_cost)
print("Actual total cost for the molecules:", actual_total_cost)
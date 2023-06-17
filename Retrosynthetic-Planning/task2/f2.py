import argparse
import pandas as pd
import numpy as np
import gzip
from sklearn.linear_model import LinearRegression

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

# 预测总成本
def predict_total_cost(molecule_fps):
    total_fp = np.zeros(2048)  # 初始化总特征向量为全零向量
    for molecule_fp in molecule_fps:
        molecule_fp = np.unpackbits(molecule_fp)
        total_fp += molecule_fp  # 将每个分子的特征向量加和
    cost = model.predict([total_fp])  # 使用模型预测总成本
    return cost[0]

# 解析命令行参数
parser = argparse.ArgumentParser(description='Predict cost for given molecules.')
parser.add_argument('-m', '--molecules', nargs='+', type=int, required=True, help='List of molecule IDs')
args = parser.parse_args()

# 获取需要预测的分子编号
molecule_ids = args.molecules

# 提取需要预测分子的特征
molecule_fps = [test_data['packed_fp'][mol_id] for mol_id in molecule_ids]

# 预测总成本
total_cost = predict_total_cost(molecule_fps)
print("Predicted total cost for the molecules:", total_cost)

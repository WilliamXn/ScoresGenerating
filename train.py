import os
import numpy as np
import librosa
import pickle
from sklearn.neural_network import MLPClassifier

def extract_features(file_path):
    audio, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features = np.mean(mfcc, axis=1)
    return features

def create_dataset(folder_path, label):
    dataset = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        features = extract_features(file_path)
        dataset.append((features, label))
    return dataset

# 定义文件夹路径和对应的标签
folder1_path = "D:\Downloads\扫弦"  # 第一个文件夹的路径
folder2_path = "D:\Downloads\拨弦"  # 第二个文件夹的路径
label1 = 0  # 第一个文件夹对应的标签
label2 = 1  # 第二个文件夹对应的标签

# 创建数据集
dataset = []
dataset += create_dataset(folder1_path, label1)
dataset += create_dataset(folder2_path, label2)

# 打乱数据集顺序
np.random.shuffle(dataset)

# 分离特征和标签
features, labels = zip(*dataset)
features = np.array(features)
labels = np.array(labels)

# 保存特征和标签到文件
np.save("features.npy", features)
np.save("labels.npy", labels)

# 实例化模型
model = MLPClassifier(hidden_layer_sizes=(100,))

# 训练模型
model.fit(features, labels)

# 保存模型到文件
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
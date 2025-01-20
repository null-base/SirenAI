import os

import librosa
import numpy as np
import pandas as pd

# 音声ファイルのフォルダ
DATASET_PATH = "./dataset"
OUTPUT_CSV = "features.csv"

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13次元のMFCCを抽出
    mfcc_mean = np.mean(mfcc, axis=1)  # 各次元の平均
    return mfcc_mean

def process_dataset():
    features = []
    labels = []
    for label in os.listdir(DATASET_PATH):  # フォルダ名をラベルとして扱う
        class_path = os.path.join(DATASET_PATH, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                if file.endswith(".wav"):
                    print(f"Processing {file_path}")
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(label)
    # データをCSVに保存
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Features saved to {OUTPUT_CSV}")

process_dataset()

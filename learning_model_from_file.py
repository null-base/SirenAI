import os
import pickle

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# サンプリング設定
RATE = 16000
N_MFCC = 13  # MFCCの係数数
FEATURE_NAMES = [f"mfcc_{i}" for i in range(N_MFCC)]  # 特徴量名

# 音声ファイルからMFCC特徴量を抽出する関数
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # 音声の長さをチェック
        if len(y) < 2048:
            print(f"音声ファイルが短すぎます（{len(y)} サンプル）：{file_path}")
            return None

        # n_fft を音声の長さ以下に設定
        n_fft = min(2048, len(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, n_mels=40)  # n_mels を減少
        mfcc_mean = mfcc.mean(axis=1)
        return mfcc_mean
    except Exception as e:
        print(f"{file_path} の処理中にエラーが発生しました: {e}")
        return None

# 音声ファイルが格納されたディレクトリからデータとラベルを作成する関数
def load_data_from_directories(siren_directory, non_siren_directory):
    features = []
    labels = []

    # サイレン音ファイルを処理
    for filename in os.listdir(siren_directory):
        if filename.endswith('.wav'):  # WAVファイルだけを対象
            file_path = os.path.join(siren_directory, filename)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append('siren')  # サイレン音のラベル

    # 非サイレン音ファイルを処理
    for filename in os.listdir(non_siren_directory):
        if filename.endswith('.wav'):  # WAVファイルだけを対象
            file_path = os.path.join(non_siren_directory, filename)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append('non_siren')  # 非サイレン音のラベル

    return np.array(features), np.array(labels)

# モデルを学習し、保存する関数
def train_and_save_model(features, labels, model_file='siren_model.pkl'):
    # ラベルを数値に変換
    labels_num = np.array([1 if label == 'siren' else 0 for label in labels])

    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(features, labels_num, test_size=0.2, random_state=42)

    # ランダムフォレストモデルを学習
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # モデルの精度を表示
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # モデルを保存
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_file}")

# メイン関数
def main():
    # サイレン音と非サイレン音の音声ファイルが格納されたディレクトリのパス
    siren_directory = './dataset/siren/'  # ここにサイレン音のフォルダパスを入力
    non_siren_directory = './dataset/non_siren/'  # ここに非サイレン音のフォルダパスを入力

    # 音声ファイルからデータをロード
    features, labels = load_data_from_directories(siren_directory, non_siren_directory)

    if len(features) > 0:
        # モデルの学習と保存
        train_and_save_model(features, labels)
    else:
        print("No valid audio files found in the directories.")

if __name__ == '__main__':
    main()

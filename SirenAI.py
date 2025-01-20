import librosa
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model

# モデルのロード
model_path = "/Users/null/Documents/Development/NITGC/model.keras"  # 保存したモデルのパス
model = load_model(model_path)

# モデルに合わせた設定
sample_rate = 16000  # サンプリングレート（学習時と一致させる）
duration = 2  # 秒単位の音声クリップ長（例: 2秒）
num_mfcc = 40  # MFCCの次元数（学習時と一致させる）

# クラスラベル（例: サイレン/非サイレン）
class_labels = ["Non-Siren", "Siren"]

# 音声を処理する関数
def process_audio(audio):
    # 音声データを特徴量に変換
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc)

    # モデルに合うように形状を変更（Conv2D用）
    mfccs = np.expand_dims(mfccs, axis=-1)  # チャネル次元を追加
    mfccs = np.expand_dims(mfccs, axis=0)  # バッチ次元を追加
    return mfccs

# 音声をリアルタイムで取得し、モデルで推論
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    print(f"Audio data received with shape: {indata.shape}")
    audio_data = indata[:, 0]  # モノラル化
    mfccs = process_audio(audio_data)

    try:
        # モデルで予測
        prediction = model.predict(mfccs)
        predicted_class = class_labels[np.argmax(prediction)]
        prediction_confidence = np.max(prediction)  # 予測の確信度
        print(f"Predicted Class: {predicted_class} with confidence {prediction_confidence:.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

# ストリームを開始
with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback, blocksize=int(sample_rate * duration)):
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopped.")

import datetime
import pickle

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaudio
from matplotlib.animation import FuncAnimation

# モデルをロード
with open("siren_model.pkl", "rb") as f:
    model = pickle.load(f)

# サンプリング設定
RATE = 16000
CHUNK = 4096
FEATURE_NAMES = [str(i) for i in range(13)]  # 特徴量名を数値の文字列に変更

# 波形をリアルタイムで描画するクラス
class RealtimePlot:
    def __init__(self, chunk):
        self.chunk = chunk
        self.fig, self.ax = plt.subplots()
        self.x = np.arange(0, chunk)  # サンプル数
        self.y = np.zeros(chunk)      # 波形データ
        self.line, = self.ax.plot(self.x, self.y)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, chunk)
        plt.title("Real-time Audio Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

    def update_plot(self, new_data):
        self.y = new_data
        self.line.set_ydata(self.y)
        return self.line,

def predict_siren(audio_data):
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)
        # print("MFCC mean:", mfcc_mean)


        # データフレームを作成し、特徴量名を一致させる
        mfcc_df = pd.DataFrame([mfcc_mean], columns=FEATURE_NAMES)

        # 予測
        prediction = model.predict(mfcc_df)
        return prediction[0]
    except ValueError as e:
        print(f"ValueError during prediction: {e}")
        return None
    except librosa.util.exceptions.ParameterError as e:
        print(f"Librosa ParameterError during prediction: {e}")
        return None

# リアルタイム検知とプロット
def listen_and_detect():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for sirens...")
    plot = RealtimePlot(CHUNK)

    def update(frame):
        try:
            # 音声データの取得
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
            plot.update_plot(data)

            # サイレン検知
            label = predict_siren(data)
            if label == "siren":
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("Siren detected!")
                print(f"Detected at {current_time}")
            return plot.line,
        except IOError as e:
            print(f"Buffer overflow: {e}")
            return plot.line,

    ani = FuncAnimation(plot.fig, update, interval=100, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    listen_and_detect()

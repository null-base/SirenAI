import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.signal import correlate

# 定数設定
SAMPLING_RATE = 44100  # サンプリングレート (Hz)
CHUNK_SIZE = 22050  # チャンクサイズ (0.5秒分のデータ)
MIC_DISTANCE = 0.2  # マイク間距離 (m)
SOUND_SPEED = 343  # 音速 (m/s)

# 音源定位の計算関数
def calculate_angle(signal1, signal2):
    # 相互相関を計算
    corr = correlate(signal1, signal2, mode='full')
    lag = np.argmax(corr) - len(signal1) + 1

    # 時間差を計算
    time_diff = lag / SAMPLING_RATE

    # 角度を計算
    try:
        angle_value = time_diff * SOUND_SPEED / MIC_DISTANCE
        angle_value = np.clip(angle_value, -1, 1)  # 値を[-1, 1]に制限
        angle = np.arcsin(angle_value) * 180 / np.pi
    except ValueError:
        angle = 90 if time_diff > 0 else -90

    return angle

# オーディオデータのコールバック関数（デバイス1用）
def audio_callback1(indata, frames, time, status):
    global buffer1
    if status:
        print(f"Device 1 Status: {status}")
    buffer1 = np.append(buffer1, indata[:, 0])

# オーディオデータのコールバック関数（デバイス2用）
def audio_callback2(indata, frames, time, status):
    global buffer2
    if status:
        print(f"Device 2 Status: {status}")
    buffer2 = np.append(buffer2, indata[:, 0])

# バッファの初期化
buffer1 = np.array([])
buffer2 = np.array([])

# デバイス設定
device1 = 3  # デバイス1のIDを指定
device2 = 5  # デバイス2のIDを指定

# プロットの初期設定
plt.ion()  # インタラクティブモードを有効化
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
line1, = ax[0].plot([], [], label="Mic 1")
line2, = ax[1].plot([], [], label="Mic 2")
ax[0].set_xlim(0, CHUNK_SIZE)
ax[1].set_xlim(0, CHUNK_SIZE)
ax[0].set_ylim(-1, 1)
ax[1].set_ylim(-1, 1)
ax[0].set_title("Mic 1 Signal")
ax[1].set_title("Mic 2 Signal")
ax[0].legend()
ax[1].legend()

# ストリーム開始
with sd.InputStream(device=device1, channels=1, samplerate=SAMPLING_RATE, callback=audio_callback1), \
     sd.InputStream(device=device2, channels=1, samplerate=SAMPLING_RATE, callback=audio_callback2):
    print("Recording... Press Ctrl+C to stop.")
    try:
        while True:
            if len(buffer1) >= CHUNK_SIZE and len(buffer2) >= CHUNK_SIZE:
                # 処理する部分を切り出し
                segment1 = buffer1[:CHUNK_SIZE]
                segment2 = buffer2[:CHUNK_SIZE]

                # バッファを更新
                buffer1 = buffer1[CHUNK_SIZE:]
                buffer2 = buffer2[CHUNK_SIZE:]

                # 角度を計算
                angle = calculate_angle(segment1, segment2)
                print(f"Estimated angle: {angle:.2f} degrees")

                # プロットを更新
                line1.set_data(range(len(segment1)), segment1)
                line2.set_data(range(len(segment2)), segment2)
                plt.pause(0.01)  # 描画の更新
    except KeyboardInterrupt:
        print("\nStopped.")

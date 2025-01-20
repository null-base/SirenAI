import numpy as np
import sounddevice as sd
from scipy.signal import butter, correlate, filtfilt

# 定数設定
SAMPLING_RATE = 44100
CHUNK_SIZE = 22050
MIC_DISTANCE = 0.2
SOUND_SPEED = 343

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def calculate_angle(signal1, signal2):
    # 信号のフィルタ処理
    signal1 = bandpass_filter(signal1, 300, 3000, SAMPLING_RATE)
    signal2 = bandpass_filter(signal2, 300, 3000, SAMPLING_RATE)

    # 相互相関を計算
    corr = correlate(signal1, signal2, mode='full')
    lag = np.argmax(corr) - len(signal1) + 1

    # 時間差を計算
    time_diff = lag / SAMPLING_RATE
    max_time_diff = MIC_DISTANCE / SOUND_SPEED
    time_diff = np.clip(time_diff, -max_time_diff, max_time_diff)

    # 角度を計算
    angle_value = time_diff * SOUND_SPEED / MIC_DISTANCE
    angle = np.arcsin(angle_value) * 180 / np.pi

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
            sd.sleep(100)  # 0.1秒待機
    except KeyboardInterrupt:
        print("\nStopped.")

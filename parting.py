from pydub import AudioSegment
import numpy as np
import librosa
from tqdm import tqdm
import joblib
import tempfile
import soundfile as sf
import os

# 加载模型
svm = joblib.load("model.pkl")

def extract_features(audio, sr):
    # 确保音频片段的长度至少为2048个样本
    if len(audio) < 2048:
        audio = np.pad(audio, (0, 2048 - len(audio)), constant_values=(0, 0))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features = np.mean(mfcc, axis=1)
    return features

# 1. 使用pydub库加载音频文件
audio = AudioSegment.from_file("D:\迅雷云盘\罗大佑-童年.wav")

# 2. 将音频数据转换为numpy数组
y = np.array(audio.get_array_of_samples())

# 3. 如果音频文件是立体声，需要将其转换为单声道
if audio.channels == 2:
    y = y[::2]

# 4. 将音频数据的采样率设置为librosa库所需的采样率
sr = audio.frame_rate

# 5. 将音频数据转换为浮点数类型
y = y / (2**15)

# 6. 计算音频的色度图
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# 7. 计算音频的节拍
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# 8. 将节拍帧转换为时间
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# 9. 假设每个小节有4个节拍
beats_per_bar = 4

# 10. 将节拍时间按照小节进行拆分
bar_times = [beat_times[i:i+beats_per_bar] for i in range(0, len(beat_times), beats_per_bar)]

# 11. 定义音阶和和弦的映射
scale_to_chord = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# 12. 对每个小节进行处理
for bar in tqdm(bar_times, desc="Processing bars"):
    # 计算小节的开始和结束帧
    start_frame = librosa.time_to_frames(bar[0], sr=sr)
    end_frame = librosa.time_to_frames(bar[-1], sr=sr)

    # 提取小节的音频数据
    bar_audio = y[start_frame:end_frame]

    # 创建一个临时文件
    fd, path = tempfile.mkstemp(suffix=".wav")

    try:
        # 保存小节的音频数据为临时文件
        sf.write(path, bar_audio, sr)

        # 提取特征
        extracted_features = extract_features(bar_audio, sr)

        # 进行分类预测
        prediction = svm.predict([extracted_features])

        # 提取小节的色度
        bar_chroma = chroma[:, start_frame:end_frame]

        # 计算小节的和弦
        chord = scale_to_chord[np.argmax(np.sum(bar_chroma, axis=1))]

        # 打印小节的开始和结束时间，和弦以及预测结果
        if prediction[0] == 0:
            print(f"小节开始时间：{bar[0]}, 小节结束时间：{bar[-1]}, 和弦：{chord}, 弹奏方法：扫弦")
        else:
            print(f"小节开始时间：{bar[0]}, 小节结束时间：{bar[-1]}, 和弦：{chord}, 弹奏方法：拨弦")
    finally:
        # 关闭文件描述符并删除临时文件
        os.close(fd)
        os.remove(path)
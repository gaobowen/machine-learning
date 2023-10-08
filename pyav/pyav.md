# python 使用 PyAV 进行 rtmp 直播推流

```py
import cv2
import time
import numpy as np

import av
import av.datasets
import librosa
from fractions import Fraction

# 创建一个 FLV 容器
container = av.open('rtmp://test-push.xxxxx.com/livetest/video1', mode='w', format='flv')
# container = av.open('video1.flv', mode='w')

# 添加视频流
w = 256
h = 256
stream = container.add_stream('h264', rate=25, gop_size=1)  # 使用 H264 编码器，fps
stream.width = w  # 设置视频宽度
stream.height = h  # 设置视频高度
stream.pix_fmt = 'yuv420p'  # 设置像素格式
stream.gop_size = 1


# 添加音频流
AUDIO_SAMPLE_RATE = 24000
audio_stream = container.add_stream("aac", rate=AUDIO_SAMPLE_RATE)  # 使用 AAC 编码器
audio_stream.layout = 'mono'  # 设置音频格式
audio_stream.channels = 1  # 设置音频通道


speech_array, sample = librosa.load('input.wav', sr=24000) # 获取PCM

speech_array = (speech_array*32767).astype(np.int16)  # 转为整型
mod = len(speech_array) % 24000

if mod > 0:
    cat_array = np.zeros(24000-mod, dtype=np.int16)
    speech_array = np.concatenate((speech_array, cat_array), axis=0)

print('speech_array', speech_array, len(speech_array))


audio_format="s16" #pyav的s16 就是 s16le

#准备图像数据
im = cv2.imread('53aa2b30a940.png')
im = cv2.resize(im, (w,h))
print(im.shape)

# 创建视频帧
for i in range(len(speech_array)//AUDIO_SAMPLE_RATE):  

    img_frame = av.VideoFrame.from_ndarray(im, format='bgr24')
    for f in range(25):
      img_frame.time_base = Fraction(1, 25000)
      img_frame.pts = (i*25 + f)*1000
      for packet in stream.encode(img_frame):
          container.mux(packet)


    audio_frame = av.AudioFrame(audio_format, layout='mono', samples=AUDIO_SAMPLE_RATE)
    audio_frame.rate = AUDIO_SAMPLE_RATE
    audio_bytes = speech_array[i*AUDIO_SAMPLE_RATE:(i+1)*AUDIO_SAMPLE_RATE] #b"\x00\x00" * frame_size
    audio_frame.planes[0].update(audio_bytes)
    audio_frame.time_base = Fraction(1, AUDIO_SAMPLE_RATE)
    audio_frame.pts = i * AUDIO_SAMPLE_RATE
    for packet in audio_stream.encode(audio_frame):
        container.mux(packet)


    time.sleep(1)
    print(f'index={i}')

# 关闭容器
container.close()
```
import cv2, json, textwrap, tempfile, torch, torchaudio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip, concatenate_audioclips, AudioFileClip, CompositeVideoClip
from pydub import AudioSegment

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

from ChatTTS import ChatTTS
chat = ChatTTS.Chat()
chat.load_models(compile=False)

# Create a silent audio clip of the same duration as the video
audio = AudioSegment.silent(duration=100 * 1000)

for timestamp, text in [(0, 'Fighting! Please fighting!')]:
    wavs = chat.infer([text])
    torchaudio.save("demo/rendering/temp.wav", torch.from_numpy(wavs[0]), 24000)
    start_time = timestamp * 1000  # Convert to milliseconds
    audio_clip = AudioSegment.from_wav("demo/rendering/temp.wav")
    audio = audio.overlay(audio_clip, position=start_time)

audio.export('demo/rendering/temp.mp3', format="mp3")
import cv2, json, textwrap, tempfile, torch, torchaudio
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
from ChatTTS import ChatTTS
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip, concatenate_audioclips, AudioFileClip, CompositeVideoClip
from pydub import AudioSegment

chat = ChatTTS.Chat()
chat.load_models(compile=False)
torch.manual_seed(19260817)
assistant_spk = chat.sample_random_speaker()
torch.manual_seed(114514)
user_spk = chat.sample_random_speaker()
assistant_infer_code = {
  'spk_emb': assistant_spk, 
  'temperature': 1.0, # using custom temperature
  'top_P': 1.0, # top P decode
  'top_K': 1, # top K decode
}
user_infer_code = {
  'spk_emb': user_spk, 
  'temperature': 1.0, # using custom temperature
  'top_P': 1.0, # top P decode
  'top_K': 1, # top K decode
}

def create_board(height, width, font_path, font_size, interval, initial_text='', user_logo = None, assistant_logo = None):
    image = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    max_chars = width // draw.textbbox((0, 0), 'A', font=font)[2]
    if initial_text:
        draw.text((0, interval), initial_text, fill=(255, 255, 255), font=font)
        last = np.array([0, interval + font_size])
    else:
        last = np.array([0, 0])
    board = {'image': image, 'draw': draw, 'max_chars': max_chars, 'font': font, 'font_size': font_size, 'last': last, 'width': width, 'height': height, 'interval': interval}
    if assistant_logo:
        assistant_logo_ratio = assistant_logo.width / assistant_logo.height
        assistant_logo_height = font_size  
        assistant_logo_width = int(assistant_logo_height * assistant_logo_ratio)
        resized_assistant_logo = assistant_logo.resize((assistant_logo_width, assistant_logo_height))
        board['assistant_logo'] = resized_assistant_logo    
    if user_logo:
        user_logo_ratio = user_logo.width / user_logo.height
        user_logo_height = font_size
        user_logo_width = int(user_logo_height * user_logo_ratio)
        resized_user_logo = user_logo.resize((user_logo_width, user_logo_height))
        board['user_logo'] = resized_user_logo
    return board

def append_text(board: dict, text: str, with_logo: str = None):
    wrapped_lines = textwrap.wrap(text, width=board['max_chars'], break_long_words=True, replace_whitespace=True)
    for i, line in enumerate(wrapped_lines):
        board['last'] += np.array([0, board['interval']])
        if with_logo and i == 0:
            board['image'].paste(board[with_logo], board['last'].tolist(), board[with_logo])
            board['draw'].text((board['last'][0] + board[with_logo].width, board['last'][1]), line, fill=(255, 255, 255), font=board['font'])
        else:
            board['draw'].text(board['last'], line, fill=(255, 255, 255), font=board['font'])
        board['last'] += np.array([0, board['font_size']])

def update_text(board: dict, text: str):
    board['draw'].rectangle([0, 0, board['width'], board['height']], fill=(0, 0, 0))
    board['last'] = np.array([0, 0])
    append_text(board, text)

def create_live_boards(height, user_logo, assistant_logo, font_path, width=500):
    state_board = create_board(height=30, width=width, font_path=font_path, font_size=20, interval=10, initial_text='Waiting...')
    conversation_board = create_board(user_logo=user_logo, assistant_logo=assistant_logo, height=height-30, width=width, font_path=font_path, font_size=20, interval=5, initial_text='')
    return state_board, conversation_board

def main(conversation, video_path, output_path, user_logo_path, assistant_logo_path, font_path, board_width=750):
    user_logo = Image.open(user_logo_path)
    assistant_logo = Image.open(assistant_logo_path)
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Video properties
    height = 540
    width = int(height * cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo/rendering/temp.mp4', fourcc, video_fps, (width + board_width, height))

    # Process each frame
    message_idx, inference_fps = 0, 0
    streaming_texts = []
    for frame_idx in range(19260817):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        # Convert the frame to PIL Image
        video_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Create dialogue board
        if frame_idx == 0:
            state_board, conversation_board = create_live_boards(height, width=board_width, user_logo=user_logo, assistant_logo=assistant_logo, font_path=font_path)
        video_time = frame_idx / video_fps
        update_text(state_board, f"Video Time = {video_time:.1f}s, Average Processing FPS = {inference_fps:.1f}, GPU: RTX 3090")
        while len(conversation) > message_idx and video_time >= conversation[message_idx]['time']:
            message = conversation[message_idx]
            if 'fps' in message and message['fps'] != inference_fps:
                inference_fps = message['fps']
                update_text(state_board, f"Video Time = {video_time:.1f}s, Average Processing FPS = {inference_fps:.1f}, GPU: RTX 3090")
            if 'role' in message:
                role, time = message['role'], message['time']
                if role == 'user':
                    append_text(conversation_board, message['content'], with_logo='user_logo')
                    streaming_texts.append((video_time, message['en'], user_infer_code, 24000))
                elif role == 'assistant':
                    append_text(conversation_board, message['content'], with_logo='assistant_logo')
                    streaming_texts.append((video_time, message['en'], assistant_infer_code, 28000))
            message_idx += 1
        # Combine the video frame and the dialogue board
        combined = Image.new('RGB', (width + board_width, height))
        combined.paste(video_frame, (0, 0))
        combined.paste(state_board['image'], (width, 0))
        combined.paste(conversation_board['image'], (width, state_board['height']))

        # Convert the combined image back to OpenCV format and write to video
        combined_frame = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
        out.write(combined_frame)

        if message_idx >= len(conversation):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    video = VideoFileClip('demo/rendering/temp.mp4')
    # Create a silent audio clip of the same duration as the video
    audio = AudioSegment.silent(duration=video.duration * 1000)
    
    # wavs = chat.infer([text[text.index(':')+1:] for timestamp, text in streaming_texts], params_infer_code=params_infer_code)
    for i, (timestamp, text, role_infer_code, frequency) in enumerate(streaming_texts):
        wav = chat.infer([text], params_infer_code=role_infer_code)[0]
        torchaudio.save(f"demo/rendering/temp{i}.wav", torch.from_numpy(wav), frequency)
        start_time = timestamp * 1000  # Convert to milliseconds
        audio_clip = AudioSegment.from_wav(f"demo/rendering/temp{i}.wav")
        audio = audio.overlay(audio_clip, position=start_time)
    
    audio.export('demo/rendering/temp.wav', format="wav")
    combined_audio = AudioFileClip('demo/rendering/temp.wav')
    final_video = video.set_audio(combined_audio)
    final_video.write_videofile(output_path, codec="libx264", audio_codec='aac')

if __name__ == "__main__":
    user_logo_path = 'demo/user_rectangle.png'
    assistant_logo_path = 'demo/assistant_rectangle.png'
    font_path = 'demo/rendering/Consolas.ttf'
    results = json.load(open('demo/assets/cooking.json'))
    main(results['conversation'], results['video_path'], 'demo/rendering/cooking.mp4', user_logo_path, assistant_logo_path, font_path)
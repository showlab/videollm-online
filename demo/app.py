import os, torchvision, transformers
torchvision.set_video_backend('video_reader')
from functools import partial
import gradio as gr

from data.utils import ffmpeg_once

from .inference import LiveInfer
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.app --resume_from_checkpoint ... 

liveinfer = LiveInfer()

css = """
    #gr_title {text-align: center;}
    #gr_video {max-height: 480px;}
    #gr_chatbot {max-height: 480px;}
"""

get_gr_video_current_time = """async (video, _) => {
  const videoEl = document.querySelector("#gr_video video");
  return [video, videoEl.currentTime];
}"""

with gr.Blocks(title="VideoLLM-online", css=css) as demo:
    gr.Markdown("# VideoLLM-online: Online Video Large Language Model for Streaming Video", elem_id='gr_title')
    with gr.Row():
        with gr.Column():
            gr_video = gr.Video(label="video stream", elem_id="gr_video", visible=True, sources=['upload'], autoplay=True)
            gr_examples = gr.Examples(
                examples=[["demo/assets/cooking.mp4"], ["demo/assets/bicycle.mp4"], ["demo/assets/egoexo4d.mp4"]],
                inputs=gr_video,
                outputs=gr_video,
                label="Examples"
            )
            gr.Markdown("## Tips:")
            gr.Markdown("- When you upload/click a video, the model starts processing the video stream. You can input a query before or after that, at any point during the video as you like.")
            gr.Markdown("- **Gradio refreshes the chatbot box to update the answer, which will delay the program. If you want to enjoy faster demo as we show in teaser video, please use https://github.com/showlab/videollm-online/blob/main/demo/cli.py.**")
            gr.Markdown("- This work is primarily done at a university, and our resources are limited. Our model is trained with limited data, so it may not solve very complicated questions. However, we have seen the potential of 'learning in streaming'. We are working on new data method to scale streaming dialogue data to our next model.")
        
        with gr.Column():
            gr_chat_interface = gr.ChatInterface(
                fn=liveinfer.input_query_stream,
                chatbot=gr.Chatbot(
                    elem_id="gr_chatbot",
                    label='chatbot',
                    avatar_images=('demo/user_avatar.png', 'demo/assistant_avatar.png'),
                    render=False
                ),
                examples=['Please narrate the video in real time.', 'Please describe what I am doing.', 'Could you summarize what have been done?', 'Hi, guide me the next step.'],
            )
            
            def gr_frame_token_interval_threshold_change(frame_token_interval_threshold):
                liveinfer.frame_token_interval_threshold = frame_token_interval_threshold
            gr_frame_token_interval_threshold = gr.Slider(minimum=0, maximum=1, step=0.05, value=liveinfer.frame_token_interval_threshold, interactive=True, label="Streaming Threshold")
            gr_frame_token_interval_threshold.change(gr_frame_token_interval_threshold_change, inputs=[gr_frame_token_interval_threshold])

        gr_video_time = gr.Number(value=0, visible=False)
        gr_liveinfer_queue_refresher = gr.Number(value=False, visible=False)

        def gr_video_change(src_video_path, history, video_time, gate):
            name, ext = os.path.splitext(src_video_path)
            ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{liveinfer.frame_fps}fps_{liveinfer.frame_resolution}' + ext)
            if not os.path.exists(ffmpeg_video_path):
                os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
                ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=liveinfer.frame_resolution)
                logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolution')
            liveinfer.load_video(ffmpeg_video_path)
            liveinfer.input_video_stream(0)
            query, response = liveinfer()
            if query or response:
                history.append((query, response))
            return history, video_time + 1 / liveinfer.frame_fps, not gate
        gr_video.change(
            gr_video_change, inputs=[gr_video, gr_chat_interface.chatbot, gr_video_time, gr_liveinfer_queue_refresher], 
            outputs=[gr_chat_interface.chatbot, gr_video_time, gr_liveinfer_queue_refresher]
        )
        
        def gr_video_time_change(_, video_time):
            liveinfer.input_video_stream(video_time)
            return video_time
        gr_video_time.change(gr_video_time_change, [gr_video, gr_video_time], [gr_video_time], js=get_gr_video_current_time)

        def gr_liveinfer_queue_refresher_change(history):
            while True:
                query, response = liveinfer()
                if query or response:
                    history[-1][1] += f'\n{response}'
                yield history
        gr_liveinfer_queue_refresher.change(gr_liveinfer_queue_refresher_change, inputs=[gr_chat_interface.chatbot], outputs=[gr_chat_interface.chatbot])
    
    demo.queue()
    demo.launch(share=False)

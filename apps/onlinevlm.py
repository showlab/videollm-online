import os, functools, anyio, torch, json, random
from dataclasses import dataclass, asdict
from torchvision.io import read_video
import gradio as gr
from transformers import HfArgumentParser, GenerationConfig, CLIPModel, CLIPImageProcessor
from transformers.models.llama import tokenization_llama
from data import build_eval_dataset_dict
from data.template_qa import SYSTEM_PROMPT, USER, AI, FRAME_PLACEHOLDER
from modeling import build_model_for_inference, build_tokenizer
from data.utils import inverse_preprocess_to_pil_images

MODEL = 'ExpertVCLM'
TITLE = f'{MODEL} Demo'
AVATAR = 'demo/egoexo4d.png'
EXAMPLES = [
    ['EgoExo4D']
    # ['Guide me step-by-step to {task}.'],
    # ['To {task}, what is the following one step?'],
    # ['Tell me the next to {task} based on now.'],
    # ['Step-by-step instruct me the following things.'],
    # ['Could you streamingly output the following step?'],
    # ['Guide me for next plan progressively.'],
    # ['Narrate the video in real-time.']
]

class OnlineLMMApp:
    generation_config = GenerationConfig(max_new_tokens=100, num_beams=1, num_return_sequences=1, do_sample=False, top_p=1.0, temperature=1.0)
    def __init__(self, resume_from_checkpoint: str = None, llama_pretrained: str = None, clip_pretrained: str = None, **kwargs) -> None:
        self.tokenizer = build_tokenizer(llama_pretrained='meta-llama/Llama-2-7b-chat-hf')
        self.clip_pretrained = 'openai/clip-vit-large-patch14-336'
        self.clip_key = self.clip_pretrained.replace('/', '-')
        self.clip_preprocessor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        self.annos = json.load(open('datasets/egoexo4d/annotations/expert_commentary_ego.json'))
        self.results = json.load(open('outputs/ExpertVCLM-openai-clip-vit-large-patch14-336-llava-hf-llava-1.5-7b-hf/egoexo4d_expert_commentary_ego_train/bs64_lr0.0001_3e_lora128x256_4c1f1w1p_ranking0.1/predictions.json'))
        self.annos = [anno for anno in self.annos if anno['take_dir'] in self.results]
        self.commentary_dirs = [anno['commentary_dir'] for anno in self.annos]

    def generate_and_choose(self, index):
        result = self.results[self.anno['take_dir']]
        t_ppls = []
        for i in range(1926):
            if str(i) in result:
                ppls = result[str(i)]['ppls']
                t_ppls.append(ppls)
            else:
                break
        print(torch.tensor(t_ppls))
        if str(index) in result:
            ppls = result[str(index)]['ppls']
            probs = torch.tensor(ppls).log()[None].softmax(dim=-1)[0].tolist()
            predictions = result[str(index)]['predictions']
        else:
            predictions = 'This take is not inferenced'
        candidates = ['[GROUND-TRUTH] ' + c if i == index else c for i, (t, c) in enumerate(self.anno['commentaries'])]
        return predictions, {c:p for c, p in zip(candidates, probs)}

    def get_embeds(self, start, end):
        if self.embeds is not None:
            return self.embeds[start:end+1]
        if self.frames is not None:
            frames = self.frames[start:end+1]
            if frames.size(0) > 0:
                return self.clip_model.get_image_features(frames.cuda())
            else:
                return frames

online_lmm_app = OnlineLMMApp()

def online_lmm_chat(message, history, old_reply: str = '', video_metadata=None):
    reply = '' if not old_reply else old_reply
    if video_metadata:
        # the first line of reply is current time
        reply = reply.split('\n')
        reply, time_indicator = reply[:-1], reply[-1]
        reply = '\n'.join(reply)
        time_indicator = f"(Current video time: {video_metadata['currentTime']}s)"
        online_lmm_replies = online_lmm_app.online_lmm_reply(message, history, video_metadata['currentTime'])
        if online_lmm_replies:
            online_lmm_reply = '\n'.join([f'{r} (Time={t}s)' for t, q, r in online_lmm_replies])
            reply += '\n' + online_lmm_reply
        reply += '\n' + time_indicator
    return reply

css = """
    #title_markdown {text-align: center;}
    #gr_video {max-height: 480px;}
    #gr_image {max-height: 480px;}
    #gr_chatbot {max-height: 480px;}
"""
with gr.Blocks(title=TITLE, css=css) as demo:
    with gr.Row():
        gr.Markdown(f'# {TITLE}', elem_id='title_markdown')
        gr_candidate_dropdown = gr.Dropdown(choices=['Same Video, All Candidates', 'Same Activities, 15+1 Candidates', 'Random 15+1 Candidates'],
                                            label='Candidate Set', value='Same Video, All Candidates')
        gr_expert_dropdown = gr.Dropdown(choices=online_lmm_app.commentary_dirs, label='Expert ID')
    gr_candidate_dropdown.input(fn=lambda x:print(x), inputs=[gr_candidate_dropdown])
    with gr.Row():
        with gr.Column():
            video_box = gr.Video(label="Video Stream", elem_id="gr_video", visible=True, sources=['upload'])
            gallery_box = gr.Gallery(label="Key Frames", elem_id="gr_gallery", visible=True, object_fit="contain", height="auto")
            upload_video_btn = gr.UploadButton("📁 Load Video", file_types=["video"])
        with gr.Column():
            chat_interface = gr.ChatInterface(
                fn=online_lmm_chat,
                chatbot=gr.Chatbot(
                    elem_id="gr_chatbot",
                    label=MODEL,
                    avatar_images=(None, AVATAR),
                    render=False,
                ),
                examples=EXAMPLES,
            )
            gr_label = gr.Label(elem_id="gr_label", label='Candidates')

    def show_video_and_key_frames(commentary_dir, history):
        anno = online_lmm_app.anno = online_lmm_app.annos[online_lmm_app.commentary_dirs.index(commentary_dir)]
        video_path = anno['video_path']
        key_frames = torch.load(anno['key_336x336']['path'])
        key_frames = inverse_preprocess_to_pil_images(key_frames, online_lmm_app.clip_preprocessor.image_mean, online_lmm_app.clip_preprocessor.image_std)
        online_lmm_app.frame_embeds = torch.load(anno[f'1fps_1_{online_lmm_app.clip_key}']['path'])
        online_lmm_app.key_frame_embeds = torch.load(anno[f'key_576_{online_lmm_app.clip_key}']['path'])
        history = [[None, 'Loaded successfully! Please select one key frame for commentary.']]
        return video_path, history, history, key_frames

    gr_expert_dropdown.input(
        fn=show_video_and_key_frames,
        inputs=[gr_expert_dropdown, chat_interface.chatbot_state],
        outputs=[video_box, chat_interface.chatbot, chat_interface.chatbot_state, gallery_box]
    )

    def select_key_frame(evt: gr.SelectData):
        time, commentary = online_lmm_app.anno['commentaries'][evt.index]
        generation, candidate_scores = online_lmm_app.generate_and_choose(evt.index)
        # inputs = online_lmm_app.prompt_template()
        # inputs['frames'] = torch.cat([
        #     online_lmm_app.frame_embeds.flatten(0,1),
        #     online_lmm_app.frame_embeds[:int(time)+1].flatten(0,1),
        #     online_lmm_app.key_frame_embeds.flatten(0,1),
        # ])
        history = [
            [None, 'Loaded successfully! Please select one key frame for commentary.'],
            [None, f'Generated commentary in {time}s:\n{generation}']
        ]
        return history, history, candidate_scores

    gallery_box.select(fn=select_key_frame, inputs=None, outputs=[chat_interface.chatbot_state, chat_interface.chatbot, gr_label])

    demo.queue()
    demo.launch(share=True)

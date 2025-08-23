import torchvision, torch, tqdm, json
torchvision.set_video_backend('video_reader')
from transformers import CLIPVisionModel
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision.transforms.functional import normalize

from modeling import build_online_lmm, get_online_lmm_args_class, build_tokenizer
from data import Ego4DNarration

def generate(model, next_token_id, past_key_values):
    output_ids = []
    while next_token_id.item() != 2:
        output_ids.append(next_token_id)
        outputs = model.forward(input_ids=next_token_id, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = outputs.logits.argmax(dim=-1)
    output_ids = torch.cat(output_ids, dim=-1)
    return output_ids, past_key_values

if __name__ == "__main__":
    # 1. load model and tokenizer
    resume_from_checkpoint = 'outputs/online_lmm-openai--clip-vit-large-patch14-336-lmsys--vicuna-7b-v1.5-16k/ego4d_refined_narration_train/bs64_lr0.0003_2e_^model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$_stream1.0'
    vision_pretrained = 'openai/clip-vit-large-patch14-336'
    llama_pretrained = 'lmsys/vicuna-7b-v1.5-16k'

    num_vision_tokens_per_frame = 5
    spatial_size = (2,2)

    model, tokenizer = build_online_lmm(
        resume_from_checkpoint=resume_from_checkpoint,
        is_training=False,
        llama_pretrained=llama_pretrained,
        vision_pretrained=vision_pretrained,
        load_vision_embeds='1+2x2',
        attn_implementation='flash_attention_2',
    )
    pooler = torch.nn.AdaptiveAvgPool2d(spatial_size)
    vision_model = CLIPVisionModel.from_pretrained(vision_pretrained).vision_model
    model.eval()
    vision_model.eval()
    model.to(torch.bfloat16)
    vision_model.to('cuda')
    model.to('cuda')

    # 2. prepare chat template
    system_prompt = get_online_lmm_args_class().system_prompt
    frame_placeholder = get_online_lmm_args_class().frame_placeholder
    chat = [
        {'role': 'system', 'content': system_prompt},
        {"role": "user", "content": "You will receive a video in streaming mode. Please narrate the video in real time, and feel free to interrupt me to add narration whenever you think it's necessary."},
        {"role": "assistant", "content": "Sure! Please continuously provide video frames and I will interrupt you to add narration when I think it's necessary."},
    ]
    chat.append({'role': 'user', 'content': ''})
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.cuda()

    # 3. decode video
    # video_path = 'assets/datasets/egoexo4d/takes_2fps_336x336/nus_cooking_10_1/frame_aligned_videos/aria01_214-1.mp4'
    # video_path = 'assets/ecb608ac-6443-46d4-9025-1300c456b8c8_2fps_336x336.mp4'
    # video_path = 'datasets/ego4d/v2/full_scale_2fps_336x336/3fc60e72-91ad-4320-bd07-1cf753f4a5f1.mp4'
    video_path = 'assets/datasets/egoexo4d/takes_2fps_336x336/nus_cooking_10_4/frame_aligned_videos/aria01_214-1.mp4'
    frames = torchvision.io.read_video(video_path, output_format='TCHW', pts_unit='sec')[0]
    frames = normalize(frames / 255, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)

    # 4. prepare inputs to our model
    past_key_values = model.forward(input_ids=input_ids, use_cache=True).past_key_values
    frame_placeholder_id = 30816
    query_prefix = torch.tensor([[3148, 1001, 29901, 29871]], device='cuda') # 'USER: '
    answer_prefix = torch.tensor([[319, 1799, 9047, 13566, 29901]], device='cuda') # 'ASSISTANT:'
    answer_eos_prefix = torch.tensor([[319, 1799, 9047, 13566, 29901, 2]], device='cuda') # 'ASSISTANT: </s>'
    frame_placeholder_idt = torch.tensor([[frame_placeholder_id] * num_vision_tokens_per_frame], device='cuda')
    frame_eos_threshold = 0.65
    chat = [

    ]
    pbar = tqdm.tqdm(total=len(frames))
    for i, frame in enumerate(frames):
        if i >= 400:
            break
        frame = frame.cuda()
        input_ids = torch.cat([input_ids, frame_placeholder_idt], dim=-1)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                last_hidden_state = vision_model(frame[None]).last_hidden_state
                cls_token = last_hidden_state[:, :1]
                spatial_tokens = pooler(last_hidden_state[:, 1:].permute(0,2,1).unflatten(-1, (24, 24))).flatten(2,3).permute(0,2,1)
                frames = torch.cat([cls_token, spatial_tokens], dim=1).bfloat16()

            outputs = model.forward(
                input_ids = input_ids[:, past_key_values[0][0].size(2):],
                frames = frames,
                frame_placeholder_id = frame_placeholder_id,
                past_key_values = past_key_values,
                use_cache = True
            )
            past_key_values = outputs.past_key_values
            score = outputs.logits[0, -1].softmax(dim=-1)
            if score[tokenizer.eos_token_id] < frame_eos_threshold:
                score[tokenizer.eos_token_id] = 0
            if score.argmax() == tokenizer.eos_token_id:
                pbar.update(1)
                chat.append({'time': i / 2, 'text': '', 'fps': pbar.format_dict['rate']})
                continue
            input_ids = torch.cat([input_ids, answer_prefix], dim=-1)
            outputs = model.generate(
                input_ids = input_ids,
                past_key_values = past_key_values,
                use_cache = True,
                max_new_tokens = 50,
                do_sample = False, top_p = 1.0, temperature=1.0,
                return_dict_in_generate = True,
            )
            past_key_values = outputs.past_key_values
            response = tokenizer.decode(outputs.sequences[0, input_ids.size(1):])
            print(i / 2, response)
            input_ids = outputs.sequences
            input_ids = torch.cat([input_ids, query_prefix], dim=-1)
        chat.append({'time': i / 2, 'text': response, 'fps': pbar.format_dict['rate']})
        pbar.update(1)
    json.dump(chat, open('assets/predictions2.json', 'w'), indent=4)
    print(torch.cuda.max_memory_allocated() / 1024**3, 'GB')

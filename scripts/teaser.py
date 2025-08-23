import torchvision, torch, os, tqdm, json
from dataclasses import asdict
torchvision.set_video_backend('video_reader')
from torchvision.io import read_video
from transformers import logging

from models import build_model_and_tokenizer, parse_args
from data.utils import ffmpeg_once

logger = logging.get_logger(__name__)

@torch.no_grad()
def generate(model, input_ids, frames = None, past_key_values = None, frame_token_interval_id: torch.Tensor = None, eos_token_id: torch.Tensor = None, frame_token_interval_threshold: float = 0):
    outputs = model(input_ids=input_ids, frames=frames, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    score = outputs.logits[:,-1].softmax(dim=-1)
    # 1. determine if this frame needs generation
    if frames is not None:
        break_token_id = frame_token_interval_id if frame_token_interval_id else eos_token_id
        if score[:, break_token_id].le(frame_token_interval_threshold):
            score[:, break_token_id] = 0
        next_token_id = score.argmax(dim=-1)
        if next_token_id == break_token_id:
            return (break_token_id[None], past_key_values) if frame_token_interval_id else (None, past_key_values)
    else:
        next_token_id = score.argmax(dim=-1)

    output_ids = [next_token_id]
    # 2. this frame needs generation
    while next_token_id.item() != eos_token_id:
        outputs = model(input_ids=next_token_id[None], past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = outputs.logits[:,-1].argmax(dim=-1)
        output_ids.append(next_token_id)
    output_ids = torch.stack(output_ids, dim=1)
    return output_ids, past_key_values

def load_streaming_frames(src_path: str, args, start_time, end_time, brightness):
    dst_path = os.path.join('assets', 'cache', f'{args.frame_fps}fps_{args.frame_resolution}_{start_time}s-{end_time}s_brightness{brightness}', src_path)
    if not os.path.exists(dst_path):
        ffmpeg_once(src_path, dst_path, fps=args.frame_fps, resolution=args.frame_resolution, brightness=brightness, start_time=start_time, end_time=end_time)
    frames = read_video(dst_path, output_format='TCHW', pts_unit='sec')[0].cuda()
    return frames

def add_sys_to_context(model, args):
    conversation = [{'role': 'system', 'content': args.system_prompt}]
    return model(tokenizer.apply_chat_template(conversation, return_tensors='pt').cuda(), use_cache=True).past_key_values

# python cli.py --live_version live1+ --resume_from_checkpoint outputs/demo/fast1+ --vision_drop_strategy mod_0.2 --frame_token_interval_threshold 0.9
# datasets/egoexo4d/v2/takes/nus_cooking_10_2/frame_aligned_videos/downscaled/448/aria01_214-1.mp4
# (for live1) Please narrate the video in real time. Use the tag 'C' to denote the camera wearer, and other letter tags, such as 'X', to denote other individuals in the scene.

candidates = [
    ('datasets/ego4d/v2/full_scale/e25318c3-1bb2-4d60-9476-98b177af30be.mp4', 3, 200, 0),
    ('datasets/ego4d/goalstep/v2/full_scale/grp-694bb5c8-233d-4bdd-825f-763db7429fbd.mp4', 10, 610, 0),
]

if __name__ == "__main__":
    args = parse_args()
    logger.warning_once('\nWelcome to VideoLLM-online CLI!\n')
    logger.warning_once('\nBuild model and tokenizer...\n')
    model, tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, **asdict(args))
    print(model.config)
    model.to('cuda')
    frame_num_tokens = int(model.config.frame_token_cls)
    if model.config.frame_token_pooled:
        frame_num_tokens += model.config.frame_token_pooled[0] * model.config.frame_token_pooled[1]
    frame_token_interval_id = torch.tensor([model.config.frame_token_interval_id], device='cuda') if model.config.frame_token_interval_id else None
    eos_token_id = torch.tensor([model.config.eos_token_id], device='cuda')

    for video_path, start_time, end_time, brightness in candidates:
        logger.warning(f'\nLoad streaming video frames from {video_path}...\n')
        frames = load_streaming_frames(video_path, args, start_time, end_time, brightness)
        past_key_values = add_sys_to_context(model, args)
        input_ids = torch.tensor([[]], dtype=torch.long).cuda()
        frame_placeholder_ids = torch.tensor(model.config.v_placeholder_id, dtype=torch.long, device='cuda').repeat(frame_num_tokens).view(1, -1)
        pbar = tqdm.tqdm(total=frames.shape[0], smoothing=1.0)
        for i, frame in enumerate(frames):
            query = input('\nPlease input your query: ')
            if query:
                conversation = [ {'role': 'user', 'content': query} ]
                query_ids = tokenizer.apply_chat_template(conversation, return_tensors='pt', add_generation_prompt=True).cuda()
                output_ids, past_key_values = generate(model, input_ids=query_ids, frames=None, past_key_values=past_key_values, 
                    frame_token_interval_id=frame_token_interval_id, eos_token_id=eos_token_id, frame_token_interval_threshold=args.frame_token_interval_threshold)
                logger.warning(f'Assistant: {tokenizer.decode(output_ids[0])}')
            output_ids, past_key_values = generate(model, torch.cat([input_ids, frame_placeholder_ids], dim=-1), frame[None], past_key_values, 
                frame_token_interval_id=frame_token_interval_id, eos_token_id=eos_token_id, frame_token_interval_threshold=args.frame_token_interval_threshold)
            if output_ids is not None:
                logger.warning(f'{i / args.frame_fps}s: {tokenizer.decode(output_ids[0])}')
                input_ids = output_ids[:, -1:]
            pbar.update(1)
            fps = pbar.format_dict['rate']
            # query = input("\nYour query: ")
            # if query:
            #     conversation = [ {'role': 'user', 'content': query} ]
            #     prefix_ids = tokenizer.apply_chat_template(conversation, return_tensors='pt', add_generation_prompt=True).cuda()
            #     output_ids, past_key_values = generate(model, tokenizer, input_ids=prefix_ids,  past_key_values=past_key_values)
            #     input_ids = output_ids[:, -1:]
            #     print('Assistant: ' + tokenizer.decode(output_ids[0]))
                
        json.dump(results, open(os.path.join(args.resume_from_checkpoint, 'cli_inference.json'), 'w'))
        print(f'Saved inference results to {args.resume_from_checkpoint}/cli_inference.json')
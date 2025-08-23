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

def add_sys_and_query_to_context(model, args, query):
    conversation = [{'role': 'system', 'content': args.system_prompt}, {'role': 'user', 'content': query}]
    return model(tokenizer.apply_chat_template(conversation, return_tensors='pt').cuda(), use_cache=True).past_key_values

candidates = [
    # ('datasets/ego4d/v2/full_scale/36420847-b741-4b86-9a31-3a5bb4e296bc.mp4', 0, 100, 0), # live1+
    ('datasets/ego4d/v2/full_scale/5b8cb6f7-d2d6-471b-a710-6b2e27fa93b1.mp4', 0, 100, 0), # fast1+
    ('datasets/ego4d/v2/full_scale/e25318c3-1bb2-4d60-9476-98b177af30be.mp4', 0, 100, 0),
    ('datasets/ego4d/v2/full_scale/f6277269-1c87-439c-b5be-d4a02343018a.mp4', 0, 100, 0),
    ('datasets/ego4d/v2/full_scale/4d89d233-dbcf-4ea2-a90f-9a2c3b42d1d5.mp4', 0, 100, 0), # live1
]

# python -m scripts.narrate --live_version live1+ --resume_from_checkpoint outputs/demo/fast1+ --vision_drop_strategy mod_0.2 --frame_token_interval_threshold 0.85
# python -m scripts.narrate --live_version live1 --resume_from_checkpoint outputs/demo/live1 --frame_token_interval_threshold 0.7
# python -m scripts.narrate --live_version live1+ --resume_from_checkpoint outputs/demo/live1+_4k --frame_token_interval_threshold 0.7

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
    frame_placeholder_ids = torch.tensor(model.config.v_placeholder_id, dtype=torch.long, device='cuda').repeat(frame_num_tokens).view(1, -1)
    
    for video_path, start_time, end_time, brightness in candidates:
        logger.warning(f'\nLoad streaming video frames from {video_path}...\n')
        frames = load_streaming_frames(video_path, args, start_time, end_time, brightness)
        sys_query_key_values = add_sys_and_query_to_context(model, args, query='Please narrate the video in real time.')
        while True:
            threshold = input('press n to exit, otherwise input a frame token interval threshold, if not, just use the default args.frame_token_interval_threshold: ')
            if threshold == 'n':
                break
            if threshold == '':
                threshold = args.frame_token_interval_threshold
            else:
                threshold = float(threshold)
            past_key_values = sys_query_key_values
            input_ids = torch.tensor([[]], dtype=torch.long, device='cuda')
            pbar = tqdm.tqdm(total=frames.shape[0], smoothing=1.0)
            for i, frame in enumerate(frames):
                output_ids, past_key_values = generate(model, torch.cat([input_ids, frame_placeholder_ids], dim=-1), frame[None], past_key_values, 
                    frame_token_interval_id=frame_token_interval_id, eos_token_id=eos_token_id, frame_token_interval_threshold=threshold)
                if output_ids is not None:
                    logger.warning(f'{i / args.frame_fps}s: {tokenizer.decode(output_ids[0])}')
                    input_ids = output_ids[:, -1:]
                pbar.update(1)
        # json.dump(results, open(os.path.join(args.resume_from_checkpoint, 'cli_inference.json'), 'w'))
        # print(f'Saved inference results to {args.resume_from_checkpoint}/cli_inference.json')
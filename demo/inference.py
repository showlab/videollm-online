import torch, torchvision, transformers, collections
torchvision.set_video_backend('video_reader')
from dataclasses import asdict
from torchvision.io import read_video

from models import build_model_and_tokenizer, parse_args, fast_greedy_generate

logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

class LiveInfer:
    def __init__(self, ) -> None:
        args = parse_args()
        self.model, self.tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, **asdict(args))
        self.model.to('cuda')
        
        # visual
        self.hidden_size = self.model.config.hidden_size
        self.frame_fps = args.frame_fps
        self.frame_interval = 1 / self.frame_fps
        self.frame_resolution = self.model.config.frame_resolution
        self.frame_num_tokens = self.model.config.frame_num_tokens
        self.frame_v_placeholder = self.model.config.v_placeholder * self.frame_num_tokens
        self.frame_token_interval_id = self.model.config.frame_token_interval_id
        self.frame_placeholder_ids = torch.tensor(self.model.config.v_placeholder_id).repeat(self.model.config.frame_num_tokens).reshape(1,-1)
        
        # generation
        self.system_prompt = args.system_prompt
        self.inplace_output_ids = torch.zeros(1, 100, device='cuda', dtype=torch.long)
        self.frame_token_interval_threshold = 0.725
        self.eos_token_id = self.model.config.eos_token_id
        self._start_ids = self.tokenizer.apply_chat_template([{'role': 'system', 'content': self.system_prompt}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        self._added_stream_prompt_ids = self.tokenizer.apply_chat_template([{}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        self._added_stream_generation_ids = self.tokenizer.apply_chat_template([{}], add_stream_generation_prompt=True, return_tensors='pt').to('cuda')
        
        # app
        self.reset()

    def _call_for_response(self, video_time, query):
        if query is not None:
            self.last_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': query}], add_stream_query_prompt=True, add_generation_prompt=True, return_tensors='pt').to('cuda')
        else:
            assert self.last_ids == 933, f'{self.last_ids} != 933' # HACK, 933 = ]\n
            self.last_ids = self._added_stream_generation_ids
        inputs_embeds = self.model.get_input_embeddings()(self.last_ids)
        output_ids, self.past_key_values = fast_greedy_generate(model=self.model, inputs_embeds=inputs_embeds, past_key_values=self.past_key_values, eos_token_id=self.eos_token_id, inplace_output_ids=self.inplace_output_ids)
        self.last_ids = output_ids[:, -1:]
        if query:
            query = f'(Video Time = {video_time}s) User: {query}'
        response = f'(Video Time = {video_time}s) Assistant:{self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)}'
        return query, response
    
    def _call_for_streaming(self, ):
        while self.frame_embeds_queue:
            # 1. if query is before next frame, response
            if self.query_queue and self.frame_embeds_queue[0][0] > self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            video_time, frame_embeds = self.frame_embeds_queue.popleft()
            if not self.past_key_values:
                self.last_ids = self._start_ids
            elif self.last_ids == self.eos_token_id:
                self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)
            inputs_embeds = torch.cat([
                self.model.get_input_embeddings()(self.last_ids).view(1, -1, self.hidden_size),
                frame_embeds.view(1, -1, self.hidden_size),
            ], dim=1)
            outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.past_key_values)
            self.past_key_values = outputs.past_key_values
            # 2. if the same time, response after frame at that time
            if self.query_queue and video_time >= self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            # 3. if the next is frame but next is not interval, then response
            next_score = outputs.logits[:,-1:].softmax(dim=-1)
            if next_score[:,:,self.frame_token_interval_id] < self.frame_token_interval_threshold:
                next_score[:,:,self.frame_token_interval_id].zero_()
            self.last_ids = next_score.argmax(dim=-1)
            if self.last_ids != self.frame_token_interval_id: 
                return video_time, None
        return None, None
    
    def reset(self, ):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.last_ids = torch.tensor([[]], device='cuda', dtype=torch.long)
        self.past_key_values = None

    def input_query_stream(self, query, history=None, video_time=None):
        if video_time is None:
            self.query_queue.append((self.video_time, query))
        else:
            self.query_queue.append((video_time, query))
        if not self.past_key_values:
            return f'(NOTE: No video stream here. Please select or upload a video. Then the assistant will answer "{query} (at {self.video_time}s)" in the video stream)'
        return f'(NOTE: Received "{query}" (at {self.video_time}s). Please wait until previous frames have been processed)'
    
    def input_video_stream(self, video_time):
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            frames_embeds = self.model.visual_embed(self.video_tensor[ranger]).split(self.frame_num_tokens)
            self.frame_embeds_queue.extend([(r / self.frame_fps, frame_embeds) for r, frame_embeds in zip(ranger, frames_embeds)])
        self.last_frame_idx = frame_idx
        self.video_time = video_time
    
    def load_video(self, video_path):
        self.video_tensor = read_video(video_path, pts_unit='sec', output_format='TCHW')[0].to('cuda')
        self.num_video_frames = self.video_tensor.size(0)
        self.video_duration = self.video_tensor.size(0) / self.frame_fps
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS')

    def __call__(self, ):
        while not self.frame_embeds_queue:
            continue
        video_time, query = self._call_for_streaming()
        response = None
        if video_time is not None:
            query, response = self._call_for_response(video_time, query)
        return query, response
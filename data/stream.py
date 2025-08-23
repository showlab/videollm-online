import torch, torchvision, random
torchvision.set_video_backend("video_reader")
from transformers import PreTrainedTokenizer

from .utils import rand_bool

class StreamMixIn(torch.utils.data.Dataset):
    def __init__(self, is_training: bool, system_prompt: str, augmentation: bool, max_num_frames: int, tokenizer: PreTrainedTokenizer, **kwargs):
        super().__init__()
        self.is_training = is_training
        self.system_prompt = system_prompt
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.max_num_frames = max_num_frames
        assert system_prompt is not None, 'Please add a system prompt'

    # NOTE: algorithm:
    # if augmentation, randomly choose a assistant response r, make the following augmentation:
    # 1. 50% prob, randomly shift left random x frames or randomly shift right random x frames
    # 2. 50% prob, randomly replace r to any responses in the video
    # 3. make the augmented segment not learn. other parts are still learned.
    def augment(self, conversation): # user, stream, assistant, stream, assistant
        assistant_messages = [(i, message) for i, message in enumerate(conversation) if message['role'] == 'assistant']
        if self.augmentation and rand_bool() and self.is_training and len(assistant_messages) >= 2: # 2 round
            i, correct_assistant = random.choice(assistant_messages[:-1])
            content_to_replace = random.choice(list(set([message['content'] for message in conversation if 'assistant' == message['role']])) + [''])
            last_num_frames = conversation[i-1]['num_frames'] if conversation[i - 1]['role'] == 'stream' else 0
            next_num_frames = conversation[i+1]['num_frames'] if conversation[i + 1]['role'] == 'stream' else 0
            if last_num_frames > 1 and next_num_frames > 1:
                num_frames_to_shift = random.randint(-(last_num_frames - 1), (next_num_frames - 1))
            else:
                num_frames_to_shift = 0
            if num_frames_to_shift == 0 and content_to_replace == correct_assistant['content']:
                return conversation
            if content_to_replace == '':
                return conversation[:i] + conversation[i+1:]
            augmented = []
            if num_frames_to_shift != 0:
                augmented.append({'role': 'stream', 'learn': False, 'num_frames': conversation[i-1]['num_frames'] + num_frames_to_shift})
            else:
                augmented.append(conversation[i-1])
            augmented.append({'role': 'assistant', 'learn': False, 'content': content_to_replace})
            if num_frames_to_shift != 0:
                augmented.append({'role': 'stream', 'learn': False, 'num_frames': conversation[i+1]['num_frames'] - num_frames_to_shift})
            else:
                augmented.append(conversation[i+1])
            conversation = conversation[:i-1] + augmented + conversation[i+2:]
        return conversation

    def max_frames_clip(self, conversation: list[dict], load_ranges: dict[str, range], max_num_frames: int):
        cum_num_frames = 0
        for i, message in enumerate(conversation):
            if message['role'] == 'stream':
                if cum_num_frames + message['num_frames'] > max_num_frames:
                    conversation = conversation[:i]
                    load_ranges = {path: range(ranger.start, ranger.start + cum_num_frames) for path, ranger in load_ranges.items()}
                    break
                cum_num_frames += message['num_frames']
        return conversation, load_ranges

    def __getitem__(self, *, conversation: list[dict], load_ranges: dict[str, range], add_generation_prompt=False, **kwargs):
        # 1. load visual encoding
        conversation, load_ranges = self.max_frames_clip(conversation, load_ranges, self.max_num_frames)
        frames = torch.cat([torch.load(path)[ranger] for path, ranger in load_ranges.items()])
        # 2. prepare texts
        if self.augmentation:
            conversation = self.augment(conversation)
        conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=add_generation_prompt)
        # 3. learn ranges
        learn_ranges = self.tokenizer.get_learn_ranges(conversation) if not add_generation_prompt else []
        return text, frames, learn_ranges

import json, torch, tqdm, random

from .ego4d import Ego4D
from ..stream import StreamMixIn
from ..utils import ceil_time_by_fps, floor_time_by_fps, rand_bool, DictWithTo

class Ego4DGoalStepLiveChat(Ego4D, StreamMixIn):
    anno_path = 'datasets/ego4d/v2/annotations/goalstep_livechat_trainval_filtered_21k.json'
    evaluation_kwargs = DictWithTo(evaluator='generate')

    def __init__(self, *, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps

        annos = json.load(open(self.anno_path))
        self.annos = []
        for anno in tqdm.tqdm(annos):
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            if not anno['conversation']:
                continue
            role = anno['conversation'][0]['role']
            time = anno['conversation'][0]['time']
            content = anno['conversation'][0]['content']
            if not (role == 'user' and time > 0 and time <= duration and content):
                continue
            # 1. add random frames before the user
            fps_time = floor_time_by_fps(time, frame_fps, 0, duration)
            waiting_frames = random.randint(0, min(20, int(fps_time * frame_fps)))
            conversation = []
            if waiting_frames:
                conversation.append({'role': 'stream', 'num_frames': waiting_frames, 'learn': waiting_frames - 1})
            conversation.append({'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time})
            start_fps_time = fps_time - (waiting_frames - 1) / frame_fps
            # 2. for loop to add message
            for message in anno['conversation'][1:]:
                role, content, time = message['role'], message['content'], message['time']
                if time > duration:
                    break
                if time < conversation[-1]['time']:
                    break
                if time == conversation[-1]['time']:
                    if role == 'user':
                        break
                    else:
                        if conversation[-1]['role'] == 'user':
                            conversation.append({'role': 'assistant', 'content': content, 'time': time, 'fps_time': conversation[-1]['fps_time'], 'learn': True})
                        else:
                            conversation[-1]['content'] = content
                        continue
                if role == 'user':
                    fps_time = floor_time_by_fps(time, frame_fps, conversation[-1]['fps_time'], duration)
                    if fps_time > duration:
                        break
                    if fps_time > conversation[-1]['fps_time']:
                        conversation.append({'role': 'stream', 'num_frames': int((fps_time - conversation[-1]['fps_time']) * frame_fps), 'learn': True})
                    conversation.append({'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time})
                else:
                    fps_time = ceil_time_by_fps(time, frame_fps, conversation[-1]['fps_time'], duration)
                    if fps_time > duration:
                        break
                    if fps_time > conversation[-1]['fps_time']:
                        conversation.append({'role': 'stream', 'num_frames': int((fps_time - conversation[-1]['fps_time']) * frame_fps), 'learn': True})
                        conversation.append({'role': 'assistant', 'content': content, 'time': time, 'fps_time': fps_time, 'learn': True})
            if not conversation:
                continue
            self.annos.append({
                'conversation': conversation,
                'load_ranges': {self.metadata[video_uid]['path']: range(int(start_fps_time*frame_fps), int(conversation[-1]['fps_time']*frame_fps)+1)}
            })

    def preprocess_conversation(self, conversation):
        if self.augmentation and self.is_training and len(conversation) >= 4: # 2 round
            i = random.randint(0, len(conversation) - 1) # stream, assistant, stream, ...
            if i > len(conversation) - 3:
                return [random.choice(self.user_instructions)] + conversation
            if conversation[i]['role'] == 'stream':
                i += 1 # assistant
            assert conversation[i]['role'] == 'assistant'
            correct_assistant = conversation[i]
            wrong_texts = set([turn['content'] for turn in conversation if 'assistant' == turn['role']]) - set(correct_assistant['content'])
            wrong_texts = list(wrong_texts) + ['']
            wrong_assistant = {'role': 'assistant', 'content': random.choice(wrong_texts)}
            augmented = [wrong_assistant]
            num_next_frames = conversation[i+1]['intervals'].numel()
            if num_next_frames > 1:
                if rand_bool(): # promptly fix behavior
                    frame_placeholder_with_interval = self.v_placeholders_per_frame + self.frame_interval
                    next_stream_placeholder = frame_placeholder_with_interval * (num_next_frames - 1)
                    next_intervals = torch.arange(len(frame_placeholder_with_interval), len(next_stream_placeholder)+1, len(frame_placeholder_with_interval)) - len(self.frame_interval)
                    if self.frame_interval: # last frame does not have frame interval
                        next_stream_placeholder = next_stream_placeholder[:-len(self.frame_interval)]
                    augmented += [
                        {'role': 'stream', 'content': self.v_placeholders_per_frame, 'intervals': torch.tensor([len(self.v_placeholders_per_frame)])},
                        correct_assistant,
                        {'role': 'stream', 'content': next_stream_placeholder, 'intervals': next_intervals}
                    ]
                else: # condition on video behavior
                    augmented += [
                        {'role': 'stream', 'content': conversation[i+1]['content']}
                    ]
            else:
                augmented += [conversation[i+1]]
            conversation = conversation[:i] + augmented + conversation[i+2:]
        return [random.choice(self.user_instructions)] + conversation

    def __getitem__(self, index):
        anno = self.annos[index]
        return *super().__getitem__(
            conversation=anno['conversation'],
            load_ranges=anno['load_ranges'],
        ), index, self.evaluation_kwargs

def build_ego4d_goalstep_livechat_trainval(**kwargs):
    return Ego4DGoalStepLiveChat(**kwargs)

if __name__ == '__main__':
    build_ego4d_goalstep_livechat_trainval(
        is_training=True, augmentation=False, embed_mark='2fps_384_1+3x3', system_prompt='', tokenizer=None,
        frame_fps=2, vision_pretrained='google/siglip-large-patch16-384'
    )

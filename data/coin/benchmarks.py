import Levenshtein
import numpy as np
from transformers import PreTrainedTokenizer, EvalPrediction

from .coin import COIN
from ..stream import StreamMixIn
from ..utils import ceil_time_by_fps, DictWithTo

class COINBenchmark(COIN, StreamMixIn):
    evaluation_kwargs = DictWithTo(evaluator='generate_after_embed', max_new_tokens=512, do_sample=False, use_cache=True, temperature=1.0, top_p=1.0)

    @staticmethod
    def fuzzy_match(text, choices):
        return min([(Levenshtein.distance(text, choice), choice) for choice in choices])[1]

    def compute_metrics(self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs):
        batch_pred_tensor, sample_idxs = eval_predictions.predictions, eval_predictions.label_ids
        batch_pred_tensor[batch_pred_tensor < 0] = tokenizer.bos_token_id # not use clamp(min=0), since 0 is ! in Llama-3 tokenizer and may affect matching
        predictions = tokenizer.batch_decode(batch_pred_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        for prediction, label in zip(predictions, self.labels[sample_idxs]): # should be self.labels[sample_idx] to get the correct order
            prediction = prediction.lower().rstrip('.')
            if prediction == label or self.fuzzy_match(prediction, self.categories) == label:
                correct += 1
        return dict(accuracy=correct / len(predictions) * 100) # * 100

    def __getitem__(self, index):
        anno = self.annos[index]
        conversation = anno['conversation'] if self.is_training else anno['conversation'][:-1] # if not training, do not include the assistant message
        return *super().__getitem__(conversation=conversation, load_ranges=anno['load_ranges'], add_generation_prompt=not self.is_training), index, self.evaluation_kwargs

class COINStep(COINBenchmark):
    user_message = {
        "role": "user",
        "content": 'What is the action in the video? Format your answer concisely. No extra text output.'
    }
    def __init__(self, *, split: str, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.annos, self.labels = [], []
        for anno in self._annos:
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            steps = anno['steps']
            for i in range(len(steps)):
                response = steps[i]['text'].capitalize() + '.'
                self.labels.append(steps[i]['text'].lower())
                start_time = ceil_time_by_fps(steps[i]['start'], frame_fps, min_time=0, max_time=duration)
                end_time = ceil_time_by_fps(steps[i]['end'], frame_fps, min_time=0, max_time=duration)
                start_frame = int(start_time * frame_fps)
                end_frame = int(end_time * frame_fps) + 1
                conversation = [
                    COINStep.user_message,
                    {"role": "stream", 'num_frames': end_frame - start_frame, 'learn': True},
                    {"role": "assistant", "content": response, 'learn': True}
                ]
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(start_frame, end_frame)}
                })
        self.labels = np.array(self.labels) # for fast indexing
        self.categories = self.step_categories

def build_coin_step_train(**kwargs):
    return COINStep(split='train', **kwargs)

def build_coin_step_test(**kwargs):
    return COINStep(split='test', **kwargs)

class COINNext(COINBenchmark):
    user_message = {
        "role": "user",
        "content": 'What is the next action for the video? Format your answer concisely. No extra text output.'
    }
    def __init__(self, *, split: str, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.annos, self.labels = [], []
        for anno in self._annos:
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            steps = anno['steps']
            for i in range(len(steps) - 1):
                response = steps[i+1]['text'].capitalize() + '.'
                self.labels.append(steps[i+1]['text'].lower())
                start_time = ceil_time_by_fps(steps[i]['start'], frame_fps, min_time=0, max_time=duration)
                end_time = ceil_time_by_fps(steps[i]['end'], frame_fps, min_time=0, max_time=duration)
                start_frame = int(start_time * frame_fps)
                end_frame = int(end_time * frame_fps) + 1
                conversation = [
                    COINNext.user_message,
                    {"role": "stream", 'num_frames': end_frame - start_frame, 'learn': True},
                    {"role": "assistant", "content": response, 'learn': True}
                ]
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(start_frame, end_frame)}
                })
        self.labels = np.array(self.labels) # for fast indexing
        self.categories = self.step_categories

def build_coin_next_train(**kwargs):
    return COINNext(split='train', **kwargs)

def build_coin_next_test(**kwargs):
    return COINNext(split='test', **kwargs)

class COINTask(COINBenchmark):
    user_message = {
        "role": "user",
        "content": 'What is the overall activity in the video? Format your answer concisely. No extra text output.'
    }
    def __init__(self, *, split: str, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.annos, self.labels = [], []
        for anno in self._annos:
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            response = anno['task'].capitalize() + '.'
            self.labels.append(anno['task'].lower())
            start_time = ceil_time_by_fps(anno['start'], frame_fps, min_time=0, max_time=duration)
            end_time = ceil_time_by_fps(anno['end'], frame_fps, min_time=0, max_time=duration)
            start_frame = int(start_time * frame_fps)
            end_frame = int(end_time * frame_fps) + 1
            conversation = [
                COINTask.user_message,
                {"role": "stream", 'num_frames': end_frame - start_frame, 'learn': True},
                {"role": "assistant", "content": response, 'learn': True}
            ]
            self.annos.append({
                'conversation': conversation,
                'load_ranges': {self.metadata[video_uid]['path']: range(start_frame, end_frame)}
            })
        self.labels = np.array(self.labels) # for fast indexing
        self.categories = self.task_categories

def build_coin_task_train(**kwargs):
    return COINTask(split='train', **kwargs)

def build_coin_task_test(**kwargs):
    return COINTask(split='test', **kwargs)

class COINProcedure(COINBenchmark):
    max_num_steps = 5
    user_message = lambda num_steps: {
        "role": "user",
        "content": f'What is the next {num_steps} actions for the video? Format your answer concisely, listing each action on a new line with a number prefix. No extra text output.'
    }
    def __init__(self, *, split: str, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.annos, self.labels = [], []
        for anno in self._annos:
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            steps = anno['steps']
            for i in range(len(steps) - 1):
                start_time = ceil_time_by_fps(steps[i]['start'], frame_fps, min_time=0, max_time=duration)
                end_time = ceil_time_by_fps(steps[i]['end'], frame_fps, min_time=0, max_time=duration)
                start_frame = int(start_time * frame_fps)
                end_frame = int(end_time * frame_fps) + 1
                next_steps = steps[i+1:i+self.max_num_steps+1]
                num_next_steps = len(next_steps)
                if num_next_steps == 1:
                    conversation = [
                        COINNext.user_message,
                        {"role": "stream", 'num_frames': end_frame - start_frame, 'learn': True}
                    ]
                    response = next_steps[0]['text'].capitalize() + '.'
                    self.labels.append(np.array([next_steps[0]['text'].lower()]))
                else:
                    conversation = [
                        COINProcedure.user_message(num_next_steps),
                        {"role": "stream", 'num_frames': end_frame - start_frame, 'learn': True}
                    ]
                    response = '\n'.join(f"{i+1}. {s['text'].capitalize()}." for i, s in enumerate(next_steps))
                    self.labels.append(np.array([s['text'].lower() for s in next_steps]))
                conversation.append({"role": "assistant", "content": response, 'learn': True})
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(start_frame, end_frame)}
                })
        self.categories = self.step_categories

    def compute_metrics(self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs):
        batch_pred_tensor, sample_idxs = eval_predictions.predictions, eval_predictions.label_ids
        batch_pred_tensor[batch_pred_tensor < 0] = tokenizer.bos_token_id
        predictions = tokenizer.batch_decode(batch_pred_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct, total = 0, 0
        labels = [self.labels[i] for i in sample_idxs]
        for prediction_steps, label_steps in zip(predictions, labels):
            for prediction_step, label_step in zip(prediction_steps.split('\n'), label_steps):
                prediction_step = prediction_step.split('. ')[-1]
                if prediction_step == label_step or self.fuzzy_match(prediction_step, self.categories) == label_step:
                    correct += 1
                total += 1
        return {'accuracy': correct / total * 100}

def build_coin_procedure_train(**kwargs):
    return COINProcedure(split='train', **kwargs)

def build_coin_procedure_test(**kwargs):
    return COINProcedure(split='test', **kwargs)

class COINTaskProcedure(COINBenchmark):
    max_num_steps = 5
    get_query_single = lambda task: {
        "role": "user",
        "content": f'To {task}, what is the next action for the video? Format your answer concisely. No extra text output.'
    }
    get_query_multi = lambda task, num_steps: {
        "role": "user",
        "content": f'To {task}, what is the next {num_steps} actions for the video? Format your answer concisely, listing each action on a new line with a number prefix. No extra text output.'
    }
    def __init__(self, *, split: str, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.annos, self.labels = [], []
        for anno in self._annos:
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            steps = anno['steps']
            for i in range(len(steps) - 1):
                start_time = ceil_time_by_fps(steps[i]['start'], frame_fps, min_time=0, max_time=duration)
                end_time = ceil_time_by_fps(steps[i]['end'], frame_fps, min_time=0, max_time=duration)
                start_frame = int(start_time * frame_fps)
                end_frame = int(end_time * frame_fps) + 1
                next_steps = steps[i+1:i+self.max_num_steps+1]
                num_next_steps = len(next_steps)
                if num_next_steps == 1:
                    conversation = [
                        COINTaskProcedure.get_query_single(anno['task']),
                        {"role": "stream", 'num_frames': end_frame - start_frame, 'learn': True}
                    ]
                    response = next_steps[0]['text'].capitalize() + '.'
                    self.labels.append([next_steps[0]['text'].lower()])
                else:
                    conversation = [
                        COINTaskProcedure.get_query_multi(anno['task'], num_next_steps),
                        {"role": "stream", 'num_frames': end_frame - start_frame, 'learn': True}
                    ]
                    response = '\n'.join(f"{i+1}. {s['text'].capitalize()}." for i, s in enumerate(next_steps))
                    self.labels.append([s['text'].lower() for s in next_steps])
                conversation.append({"role": "assistant", "content": response, 'learn': True})
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(start_frame, end_frame)}
                })
        self.categories = self.step_categories

    def compute_metrics(self, *args, **kwargs):
        return COINProcedure.compute_metrics(self, *args, **kwargs)

def build_coin_taskprocedure_train(**kwargs):
    return COINTaskProcedure(split='train', **kwargs)

def build_coin_taskprocedure_test(**kwargs):
    return COINTaskProcedure(split='test', **kwargs)

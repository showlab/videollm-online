import Levenshtein as lev
import numpy as np
from transformers import PreTrainedTokenizer, EvalPrediction

from .coin import COIN
from ..stream import StreamMixIn
from ..utils import ceil_time_by_fps, DictWithTo

class COINBenchmark(COIN, StreamMixIn):
    evaluation_kwargs = DictWithTo(evaluator='generate', max_new_tokens=512, do_sample=False, use_cache=True, temperature=1.0, top_p=1.0)

    @staticmethod
    def fuzzy_match(text, choices):
        scores = [-lev.distance(text, choice) for choice in choices]
        return scores.index(max(scores))

    def compute_metrics(self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs):
        batch_pred_tensor, sample_idxs = eval_predictions.predictions, eval_predictions.label_ids
        batch_pred_tensor = batch_pred_tensor.clip(min=0)
        predictions = tokenizer.batch_decode(batch_pred_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        predictions = np.array([self.fuzzy_match(text, self.mapping_categories) for text in predictions])
        accuracy = (predictions == np.array(self.answers)).mean()
        return dict(accuracy=accuracy)

    def __getitem__(self, index):
        anno = self.annos[index]
        return *super().__getitem__(conversation=anno['conversation'], load_ranges=anno['load_ranges']), index, self.evaluation_kwargs

class COINStep(COINBenchmark):
    user_message = {
        "role": "user",
        "content": 'What is the action in the video? Format your answer concisely. No extra text output.'
    }
    def __init__(self, *, split: str, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.annos = []
        self.answers, self.mapping_categories = [], self.steps_categories
        for anno in self._annos:
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            steps = anno['steps']
            for i in range(len(steps)):
                response = steps[i]['text'].capitalize() + '.'
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
                self.answers.append(self.mapping_categories.index(response))

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
        self.annos = []
        self.answers, self.mapping_categories = [], self.steps_categories
        for anno in self._annos:
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            steps = anno['steps']
            for i in range(len(steps) - 1):
                response = steps[i+1]['text'].capitalize() + '.'
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
                self.answers.append(self.mapping_categories.index(response))

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
        self.annos = []
        self.answers, self.mapping_categories = [], self.tasks_categories
        for anno in self._annos:
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            response = anno['task'].capitalize() + '.'
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
            self.answers.append(self.mapping_categories.index(response))

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
        self.annos = []
        self.answers, self.mapping_categories = [], self.steps_categories
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
                else:
                    conversation = [
                        COINProcedure.user_message(num_next_steps),
                        {"role": "stream", 'num_frames': end_frame - start_frame, 'learn': True}
                    ]
                    response = '\n'.join(f"{i+1}. {s['text'].capitalize()}." for i, s in enumerate(next_steps))
                conversation.append({"role": "assistant", "content": response, 'learn': True})
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(start_frame, end_frame)}
                })
                self.answers.append([self.mapping_categories.index(step['text'].capitalize() + '.') for step in next_steps])

    def compute_metrics(self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs):
        batch_pred_tensor, sample_idxs = eval_predictions.predictions, eval_predictions.label_ids
        batch_pred_tensor = batch_pred_tensor.clip(min=0)
        batch_pred_text = tokenizer.batch_decode(batch_pred_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        predictions = []
        for pred_text in batch_pred_text:
            pred_steps = pred_text.split('\n')
            predictions.append([self.fuzzy_match(step, self.mapping_categories) for step in pred_steps])
        total_num_steps = len(sum(self.answers, []))
        correct_num_steps = sum([sum(1 for p, a in zip(prediction, answer) if p == a) for prediction, answer in zip(predictions, self.answers)])
        return {'accuracy': correct_num_steps / total_num_steps}

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
        self.annos = []
        self.answers, self.mapping_categories = [], self.steps_categories
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
                else:
                    conversation = [
                        COINTaskProcedure.get_query_multi(anno['task'], num_next_steps),
                        {"role": "stream", 'num_frames': end_frame - start_frame, 'learn': True}
                    ]
                    response = '\n'.join(f"{i+1}. {s['text'].capitalize()}." for i, s in enumerate(next_steps))
                conversation.append({"role": "assistant", "content": response, 'learn': True})
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(start_frame, end_frame)}
                })
                self.answers.append([self.mapping_categories.index(step['text'].capitalize() + '.') for step in next_steps])

    def compute_metrics(self, *args, **kwargs):
        return COINProcedure.compute_metrics(self, *args, **kwargs)

def build_coin_taskprocedure_train(**kwargs):
    return COINTaskProcedure(split='train', **kwargs)

def build_coin_taskprocedure_test(**kwargs):
    return COINTaskProcedure(split='test', **kwargs)
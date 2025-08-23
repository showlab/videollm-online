import torch
import numpy as np
from transformers import PreTrainedTokenizer, EvalPrediction, GenerationConfig

from .coin import COIN
from ..stream import StreamMixIn
from ..utils import round_time_by_fps

class COINBenchmark(COIN, StreamMixIn):
    evaluation_kwargs = {
        'evaluator': 'generate',
        'generation_config': GenerationConfig(max_new_tokens=200, do_sample=False, top_p=1.0, temperature=1.0)
    }

    def compute_metrics(self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs):
        batch_pred_tensor, sample_idxs = eval_predictions.predictions, eval_predictions.label_ids
        batch_pred_tensor = batch_pred_tensor.clip(min=0)
        predictions = np.array(tokenizer.batch_decode(batch_pred_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        answers = np.array([self.annos[sample_idx]['label'] for sample_idx in sample_idxs])
        accuracy = (answers[sample_idxs] == predictions).sum() / len(predictions)
        return dict(accuracy=accuracy)

    def __getitem__(self, index):
        return *super().__getitem__(**self.annos[index]), index, self.evaluation_kwargs if not self.is_training else {}

class COINStep(COINBenchmark):
    query = f'What is the action in the video? Format your answer concisely. No extra text output.'
    def __init__(self, fps: int, load_vision_embeds: bool, is_training: bool, frame_placeholder: str, **kwargs):
        super().__init__(fps=fps, load_vision_embeds=load_vision_embeds, **kwargs)
        self.is_training = is_training
        self.annos = []
        for anno in self._annos:
            video_uid = anno['video_uid']
            steps = anno['steps']
            for i in range(len(steps)):
                start_time = round_time_by_fps(steps[i]['start'], fps, min_time=0, max_time=self.metadata[video_uid]['duration'])
                end_time = round_time_by_fps(steps[i]['end'], fps, min_time=start_time, max_time=self.metadata[video_uid]['duration'])
                frame_placeholders = int((end_time - start_time) * fps + 1) * frame_placeholder
                response = steps[i]['text'].capitalize() + '.'
                chat = [
                    {"role": "user", "content": frame_placeholders + f'\n{self.query}'},
                    {"role": "assistant", "content": response, 'lm': [0, len(response)]} if self.is_training else {"role": "assistant", "content": ''}
                ]
                self.annos.append({
                    'chat': chat,
                    'load_ranges': [[
                        self.metadata[video_uid]['path'],
                        dict(
                            start_time=start_time,
                            end_time=end_time,
                        ) if not load_vision_embeds else dict(
                            start_frame=int(start_time*fps),
                            end_frame=int(end_time*fps)
                        )
                    ]],
                    'label': response,
                })

def build_coin_step_train(**kwargs):
    return COINStep(split='train', **kwargs)

def build_coin_step_test(**kwargs):
    return COINStep(split='test', **kwargs)

class COINNext(COINBenchmark):
    query = f'What is the next action for the video? Format your answer concisely. No extra text output.'
    def __init__(self, fps: int, load_vision_embeds: bool, is_training: bool, frame_placeholder: str, **kwargs):
        super().__init__(fps=fps, load_vision_embeds=load_vision_embeds, **kwargs)
        self.is_training = is_training
        self.annos = []
        for anno in self._annos:
            video_uid = anno['video_uid']
            steps = anno['steps']
            for i in range(len(steps) - 1):
                start_time = round_time_by_fps(steps[i]['start'], fps, min_time=0, max_time=self.metadata[video_uid]['duration'])
                end_time = round_time_by_fps(steps[i]['end'], fps, min_time=start_time, max_time=self.metadata[video_uid]['duration'])
                frame_placeholders = int((end_time - start_time) * fps + 1) * frame_placeholder
                response = steps[i+1]['text'].capitalize() + '.'
                chat = [
                    {"role": "user", "content": frame_placeholders + f'\n{self.query}'},
                    {"role": "assistant", "content": response, 'lm': [0, len(response)]} if self.is_training else {"role": "assistant", "content": ''}
                ]
                self.annos.append({
                    'chat': chat,
                    'load_ranges': [[
                        self.metadata[video_uid]['path'],
                        dict(
                            start_time=start_time,
                            end_time=end_time,
                        ) if not load_vision_embeds else dict(
                            start_frame=int(start_time*fps),
                            end_frame=int(end_time*fps)
                        )
                    ]],
                    'label': response,
                })

def build_coin_next_train(**kwargs):
    return COINNext(split='train', **kwargs)

def build_coin_next_test(**kwargs):
    return COINNext(split='test', **kwargs)

class COINTask(COINBenchmark):
    query = f'What is the overall activity in the video? Format your answer concisely. No extra text output.'
    def __init__(self, fps: int, load_vision_embeds: bool, is_training: bool, frame_placeholder: str, **kwargs):
        super().__init__(fps=fps, load_vision_embeds=load_vision_embeds, **kwargs)
        self.is_training = is_training
        self.annos = []
        for anno in self._annos:
            video_uid = anno['video_uid']
            start_time = round_time_by_fps(anno['start'], fps, min_time=0, max_time=self.metadata[video_uid]['duration'])
            end_time = round_time_by_fps(anno['end'], fps, min_time=start_time, max_time=self.metadata[video_uid]['duration'])
            frame_placeholders = int((end_time - start_time) * fps + 1) * frame_placeholder
            response = anno['task'].capitalize() + '.'
            chat = [
                {"role": "user", "content": frame_placeholders + f'\n{self.query}'},
                {"role": "assistant", "content": response, 'lm': [0, len(response)]} if self.is_training else {"role": "assistant", "content": ''}
            ]
            self.annos.append({
                'chat': chat,
                'load_ranges': [[
                    self.metadata[video_uid]['path'],
                    dict(
                        start_time=start_time,
                        end_time=end_time,
                    ) if not load_vision_embeds else dict(
                        start_frame=int(start_time*fps),
                        end_frame=int(end_time*fps)
                    )
                ]],
                'label': response,
            })

def build_coin_task_train(**kwargs):
    return COINTask(split='train', **kwargs)

def build_coin_task_test(**kwargs):
    return COINTask(split='test', **kwargs)

class COINProcedure(COINBenchmark):
    max_num_steps = 5
    query_single = COINNext.query
    query_multi = f'What is the next {max_num_steps} actions for the video? Format your answer concisely, listing each action on a new line with a number prefix. No extra text output.'
    def __init__(self, fps: int, load_vision_embeds: bool, is_training: bool, frame_placeholder: str, **kwargs):
        super().__init__(fps=fps, load_vision_embeds=load_vision_embeds, **kwargs)
        self.is_training = is_training
        self.annos = []
        for anno in self._annos:
            video_uid = anno['video_uid']
            steps = anno['steps']
            for i in range(len(steps) - 1):
                start_time = round_time_by_fps(steps[i]['start'], fps, min_time=0, max_time=self.metadata[video_uid]['duration'])
                end_time = round_time_by_fps(steps[i]['end'], fps, min_time=start_time, max_time=self.metadata[video_uid]['duration'])
                frame_placeholders = int((end_time - start_time) * fps + 1) * frame_placeholder
                next_steps = steps[i+1:i+self.max_num_steps+1]
                num_next_steps = len(next_steps)
                if num_next_steps == 1:
                    query = self.query_single
                    response = next_steps[0]['text'].capitalize() + '.'
                else:
                    query = self.query_multi.replace('5', str(num_next_steps))
                    response = '\n'.join(f"{i+1}. {s['text'].capitalize()}." for i, s in enumerate(next_steps))
                chat = [
                    {"role": "user", "content": frame_placeholders + f'\n{query}'},
                    {"role": "assistant", "content": response, 'lm': [0, len(response)]} if self.is_training else {"role": "assistant", "content": ''}
                ]
                self.annos.append({
                    'chat': chat,
                    'load_ranges': [[
                        self.metadata[video_uid]['path'],
                        dict(
                            start_time=start_time,
                            end_time=end_time,
                        ) if not load_vision_embeds else dict(
                            start_frame=int(start_time*fps),
                            end_frame=int(end_time*fps)
                        )
                    ]],
                    'label': response,
                })

    def compute_metrics(self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs):
        batch_pred_tensor, sample_idxs = eval_predictions.predictions, eval_predictions.label_ids
        batch_pred_tensor = batch_pred_tensor.clip(min=0)
        batch_pred_text = tokenizer.batch_decode(batch_pred_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        batch_label_text = [self.annos[sample_idx]['label'] for sample_idx in sample_idxs]
        correct_num_steps, total_num_steps = 0, 0
        for pred_text, label_text in zip(batch_pred_text, batch_label_text):
            pred_steps = pred_text.split('\n')
            label_steps = label_text.split('\n')
            for i in range(len(label_steps)):
                if i < len(pred_steps) and pred_steps[i] == label_steps[i]:
                    correct_num_steps += 1
                total_num_steps += 1
        return {'accuracy': correct_num_steps / total_num_steps}

def build_coin_procedure_train(**kwargs):
    return COINProcedure(split='train', **kwargs)

def build_coin_procedure_test(**kwargs):
    return COINProcedure(split='test', **kwargs)

class COINTaskProcedure(COINBenchmark):
    max_num_steps = 5
    task_placeholder = 'task'
    query_single = f'To {task_placeholder}, what is the next action for the video? Format your answer concisely. No extra text output.'
    query_multi = f'To {task_placeholder}, what is the next {max_num_steps} actions for the video? Format your answer concisely, listing each action on a new line with a number prefix. No extra text output.'
    def __init__(self, fps: int, load_vision_embeds: bool, is_training: bool, frame_placeholder: str, **kwargs):
        super().__init__(fps=fps, load_vision_embeds=load_vision_embeds, **kwargs)
        self.is_training = is_training
        self.annos = []
        for anno in self._annos:
            video_uid = anno['video_uid']
            steps = anno['steps']
            for i in range(len(steps) - 1):
                start_time = round_time_by_fps(steps[i]['start'], fps, min_time=0, max_time=self.metadata[video_uid]['duration'])
                end_time = round_time_by_fps(steps[i]['end'], fps, min_time=start_time, max_time=self.metadata[video_uid]['duration'])
                frame_placeholders = int((end_time - start_time) * fps + 1) * frame_placeholder
                next_steps = steps[i+1:i+self.max_num_steps+1]
                num_next_steps = len(next_steps)
                if num_next_steps == 1:
                    query = self.query_single.replace(self.task_placeholder, anno['task'])
                    response = next_steps[0]['text'].capitalize() + '.'
                else:
                    query = self.query_multi.replace('5', str(num_next_steps)).replace(self.task_placeholder, anno['task'])
                    response = '\n'.join(f"{i+1}. {s['text'].capitalize()}." for i, s in enumerate(next_steps))
                chat = [
                    {"role": "user", "content": frame_placeholders + f'\n{query}'},
                    {"role": "assistant", "content": response, 'lm': [0, len(response)]} if self.is_training else {"role": "assistant", "content": ''}
                ]
                self.annos.append({
                    'chat': chat,
                    'load_ranges': [[
                        self.metadata[video_uid]['path'],
                        dict(
                            start_time=start_time,
                            end_time=end_time,
                        ) if not load_vision_embeds else dict(
                            start_frame=int(start_time*fps),
                            end_frame=int(end_time*fps)
                        )
                    ]],
                    'label': response,
                })

    def compute_metrics(self, *args, **kwargs):
        return COINProcedure.compute_metrics(self, *args, **kwargs)

def build_coin_taskprocedure_train(**kwargs):
    return COINTaskProcedure(split='train', **kwargs)

def build_coin_taskprocedure_test(**kwargs):
    return COINTaskProcedure(split='test', **kwargs)

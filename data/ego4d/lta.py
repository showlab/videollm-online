import os, collections, json, re, torch, itertools, editdistance, Levenshtein
from transformers import AutoTokenizer, EvalPrediction
import numpy as np

from .ego4d import Ego4D
from ..stream import StreamMixIn
from ..utils import round_time_by_fps, DictWithTo

class Ego4DLTA(Ego4D, StreamMixIn):
    num_input_actions = 8
    num_future_actions = 20 # Z
    num_beams = 5 # K
    evaluation_kwargs = DictWithTo(evaluator='generate', max_new_tokens=512, num_beams=num_beams, num_return_sequences=num_beams, do_sample=False, use_cache=True, temperature=1.0, top_p=1.0)
    get_user_message = lambda num_frames: {
        "role": "user",
        "content": f"After {num_frames} video frames, anticipate the next {Ego4DLTA.num_future_actions} actions. Format your answer concisely, listing each action on a new line with a number prefix. No extra text output."
    }

    def __init__(self, split: str, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(frame_fps=frame_fps, is_training=is_training, split=split, **kwargs)
        self.split = split
        self.is_training = is_training
        # 1. build taxonomy
        taxonomy = json.load(open(os.path.join(self.root, 'annotations', 'fho_lta_taxonomy.json')))
        self.verbs = [Ego4DLTA.get_no_overlap_word(verb) for verb in taxonomy['verbs']]
        self.nouns = [Ego4DLTA.get_no_overlap_word(noun) for noun in taxonomy['nouns']]
        self.action_to_verb_label, self.action_to_noun_label = {}, {}
        action_counter = collections.defaultdict(int)
        for (i, verb), (j, noun) in itertools.product(enumerate(self.verbs), enumerate(self.nouns)):
            action = f'{verb} {noun}'
            self.action_to_verb_label[action] = i
            self.action_to_noun_label[action] = j
            action_counter[action] += 1
        self.most_common_action = max(action_counter, key=action_counter.get)

        # 2. make clip2anno annotations
        anno_path = os.path.join(self.root, 'annotations', f'fho_lta_{split}.json')
        annos = json.load(open(anno_path))['clips']
        clip2anno = collections.defaultdict(list)
        for anno in annos:
            clip2anno[anno['clip_uid']].append({
                'video_uid': anno['video_uid'],
                'start': anno['clip_parent_start_sec'] + anno['action_clip_start_sec'],
                'end': anno['clip_parent_start_sec'] + anno['action_clip_end_sec'],
                'action_idx': anno['action_idx'],
                'verb_label': anno.get('verb_label'), 'noun_label': anno.get('noun_label', None),
                'clip_idx': anno.get('clip_idx', None),
                'clip_uid': anno['clip_uid'],
            })
        clip2anno = {
            clip:sorted(anno, key=lambda x:x['action_idx']) \
            for clip, anno in clip2anno.items() \
            if len(anno) >= self.num_future_actions + self.num_input_actions
        }
        self.clip2anno = clip2anno

        # 3. make flatten annotations
        self.annos = []
        for clip_uid, anno in clip2anno.items():
            for i in range(len(anno) - self.num_future_actions - self.num_input_actions + 1):
                video_uid = anno[i]['video_uid']
                j, k = i + self.num_input_actions, i + self.num_future_actions + self.num_input_actions
                if 'test_unannotated' in split:
                    verb_labels, noun_labels = None, None
                else:
                    verb_noun_labels = [(a['verb_label'], a['noun_label']) for a in anno[j:k]]
                    response = self.verb_noun_labels_to_text(verb_noun_labels)
                    verb_labels, noun_labels = zip(*verb_noun_labels)
                start_time = round_time_by_fps(anno[i]['start'], frame_fps, min_time=0, max_time=self.metadata[video_uid]['duration'])
                end_time = round_time_by_fps(anno[j-1]['end'], frame_fps, min_time=0, max_time=self.metadata[video_uid]['duration'])
                start_frame = int(start_time * frame_fps)
                stop_frame = int(end_time * frame_fps) + 1
                conversation = [
                    Ego4DLTA.get_user_message(stop_frame - start_frame),
                    {"role": "stream", 'num_frames': stop_frame - start_frame},
                ]
                if is_training:
                    conversation[-1]['learn'] = True
                    conversation.append({"role": "assistant", "content": response, 'learn': True})
                self.annos.append({
                    'conversation': conversation,
                    'add_generation_prompt': not is_training,
                    'load_ranges': {self.metadata[video_uid]['path']: range(start_frame, stop_frame)},
                    'verb_labels': verb_labels,
                    'noun_labels': noun_labels,
                    'clip_uid': clip_uid,
                    'last_visible_action_idx': anno[j-1]['action_idx'],
                })

        # 4. fast label access
        self.annos_verb_labels = np.array([anno['verb_labels'] for anno in self.annos])
        self.annos_noun_labels = np.array([anno['noun_labels'] for anno in self.annos])

    @staticmethod
    def get_no_overlap_word(row):
        replace_dict = {
            'pot_(planter)': 'flowerpot',
            'bat_(sports)': 'sport bat',
            'bat_(tool)': 'bat',
            'nut_(food)': 'nuts',
            'nut_(tool)': 'nut',
            'chip_(food)': 'snack',
            "chip_(wood'_metal),": 'chips',
            'chip_(wood,_metal)': 'chip'
        }
        return replace_dict.get(row, Ego4DLTA.split_row_to_words(row)[0])

    @staticmethod
    def split_row_to_words(row):
        if '(' in row:
            words = [re.sub(r'_$', '', row.split('(')[0]).replace('_', ' ')]
            strings = re.sub(r'[)]', '', row.split('(')[1]).split(',')
            strings = [s.lstrip('_').replace('_', ' ') for s in strings]
            words.extend(s for string in strings for s in string.split('/'))
            return words
        else:
            return [row.replace('_', ' ')]

    def get_labels(self, indices):
        return self.annos_verb_labels[indices], self.annos_noun_labels[indices]

    def verb_noun_labels_to_text(self, verb_noun_labels: list[tuple[str]]):
        return '\n'.join([f'{i+1}. {self.verbs[v].capitalize()} {self.nouns[n]}.' for i, (v, n) in enumerate(verb_noun_labels)])

    def map_action_to_verb_label(self, action: str):
        if action not in self.action_to_verb_label:
            action = min([(Levenshtein.distance(action, key), key) for key in self.action_to_verb_label.keys()])[1]
        return self.action_to_verb_label[action]

    def map_action_to_noun_label(self, action: str):
        if action not in self.action_to_noun_label:
            action = min([(Levenshtein.distance(action, key), key) for key in self.action_to_noun_label.keys()])[1]
        return self.action_to_noun_label[action]

    def text_to_verb_noun_ids(self, text: str, num_actions: int):
        actions = []
        text = text.strip(' \n')
        for line in text.split('\n'):
            match = re.search(r'(?:\d+\.|[^\s]+\s\d+\.)\s*(.*)', line)
            if match:
                actions.append(match.group(1).lower().rstrip('.'))
        verb_noun_ids = [(self.map_action_to_verb_label(action), self.map_action_to_noun_label(action)) for action in actions]
        verb_noun_ids = verb_noun_ids[:num_actions]
        if len(verb_noun_ids) < num_actions:
            if verb_noun_ids:
                verb_noun_ids = verb_noun_ids + [verb_noun_ids[-1]] * (num_actions - len(verb_noun_ids))
            else:
                verb_noun_ids = [(
                    self.map_action_to_verb_label(self.most_common_action),
                    self.map_action_to_noun_label(self.most_common_action)
                )] * num_actions
        return verb_noun_ids

    def __getitem__(self, index):
        return *super().__getitem__(**self.annos[index]), index, self.evaluation_kwargs

    # simplified from https://github.com/EGO4D/forecasting/blob/main/ego4d_forecasting/evaluation/lta_metrics.py
    @staticmethod
    def edit_distance(preds: np.ndarray, labels: np.ndarray):
        N, K, Z = preds.shape
        dists = []
        for n in range(N):
            dist = min([editdistance.eval(preds[n, k, :], labels[n])/Z for k in range(K)])
            dists.append(dist)
        return np.mean(dists)

    @staticmethod
    def AUED(preds, labels):
        ED = np.vstack(
            [Ego4DLTA.edit_distance(preds[:, :, :z], labels[:, :z]) for z in range(1, Ego4DLTA.num_future_actions + 1)]
        )
        AUED = np.trapz(y=ED, axis=0) / (Ego4DLTA.num_future_actions - 1)
        return AUED.item()

    def compute_metrics(self, eval_predictions: EvalPrediction, *, tokenizer: AutoTokenizer, output_dir: str = './', **kwargs):
        batch_beam_pred_tensor, sample_idxs = eval_predictions.predictions, eval_predictions.label_ids
        batch_beam_pred_verb_ids, batch_beam_pred_noun_ids = [], []
        for beam_pred_tensor in batch_beam_pred_tensor:
            beam_pred_tensor = beam_pred_tensor[beam_pred_tensor != -100].reshape(self.num_beams, -1)
            beam_pred_string = tokenizer.batch_decode(beam_pred_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            beam_verb_noun_ids = np.array([self.text_to_verb_noun_ids(pred_string, self.num_future_actions) for pred_string in beam_pred_string]) # 5 x 20 x 2
            beam_pred_verb_ids, beam_pred_noun_ids = beam_verb_noun_ids[:,:,0], beam_verb_noun_ids[:,:,1]
            batch_beam_pred_verb_ids.append(beam_pred_verb_ids)
            batch_beam_pred_noun_ids.append(beam_pred_noun_ids)
        batch_beam_pred_verb_ids, batch_beam_pred_noun_ids = np.stack(batch_beam_pred_verb_ids), np.stack(batch_beam_pred_noun_ids)
        if 'test_unannotated' not in self.split:
            batch_gt_verb_ids, batch_gt_noun_ids = self.get_labels(sample_idxs)
            return {
                'verb_AUED': Ego4DLTA.AUED(batch_beam_pred_verb_ids, batch_gt_verb_ids),
                'noun_AUED': Ego4DLTA.AUED(batch_beam_pred_noun_ids, batch_gt_noun_ids)
            }
        else:
            predictions = {}
            for beam_pred_verb_ids, beam_pred_noun_ids, sample_idx in zip(batch_beam_pred_verb_ids, batch_beam_pred_noun_ids, sample_idxs):
                clip_uid = self.annos[sample_idx]['clip_uid']
                last_visible_action_idx = self.annos[sample_idx]['last_visible_action_idx']
                key = f'{clip_uid}_{last_visible_action_idx}'
                predictions[key] = dict(verb=beam_pred_verb_ids.tolist(), noun=beam_pred_noun_ids.tolist())
            if torch.cuda.current_device() == 0:
                json.dump(predictions, open(os.path.join(output_dir, f'{self.split}_predictions.json'), 'w'))
            return {}

def build_ego4d_lta_train(**kwargs):
    return Ego4DLTA(split='train', **kwargs)

def build_ego4d_lta_val(**kwargs):
    return Ego4DLTA(split='val', **kwargs)

def build_ego4d_lta_test_unannotated(**kwargs):
    return Ego4DLTA(split='test_unannotated', **kwargs)

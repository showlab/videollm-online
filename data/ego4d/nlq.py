
import os, json, collections, torch, tqdm, random

from ..ego4d import Ego4D
from ..stream import StreamMixIn
from ..utils import temporal_iou, DictWithTo, ceil_time_by_fps

class Ego4DNLQ(Ego4D):
    def __init__(self, split: str, **kwargs):
        assert split in ['train', 'val', 'test']
        super().__init__(split=split, **kwargs)
        anno_path = os.path.join(self.root, 'annotations', f'nlq_{split}.json')
        annos = json.load(open(anno_path))

        self.annos = collections.defaultdict(list)
        for video_annos in annos['videos']:
            assert video_annos['split'] == split
            video_id = video_annos['video_uid']
            for clip_annos in video_annos['clips']:
                video_start_sec, video_end_sec = clip_annos['video_start_sec'], clip_annos['video_end_sec']
                for _annos in clip_annos['annotations']:
                    for query_anno in _annos['language_queries']:
                        if 'query' in query_anno and query_anno['query']:
                            query = query_anno['query'].lower()
                        elif 'slot_x' in query_anno:
                            query = query_anno['slot_x'].lower()
                        else:
                            continue
                        sample_id = (video_id, video_start_sec, video_end_sec, query)
                        query_start_time, query_end_time = query_anno['video_start_sec'], query_anno['video_end_sec']
                        regions = self.annos[sample_id]
                        merged = False
                        for region in regions:
                            if temporal_iou(region, [query_start_time, query_end_time]) > 0:
                                region = [min(region[0], query_start_time), max(region[1], query_end_time)]
                                merged = True
                                break
                        if not merged:
                            regions.append([query_start_time, query_end_time])
        self.annos = {k: sorted(query_region_times, key=lambda x:x[0]) for k, query_region_times in self.annos.items()}

class Ego4DStreamNLQ(Ego4DNLQ, StreamMixIn):
    query_prompt_templates = [
        "Locate video clips related to the query \"QUERY\".",
        "Remind me when the query \"QUERY\".",
        "When query \"QUERY\" starts and ends, remind me.",
        "Do temporal grounding to query \"QUERY\".",
        "Can you locate query \"QUERY\" in the video?",
        "Record when query \"QUERY\".",
        "Please find the period of query \"QUERY\".",
        "Retrieve query \"QUERY\".",
        "Identify the start and end times of query \"QUERY\" in the video.",
        "Show me the video segment where query \"QUERY\" takes place.",
    ]
    evaluation_kwargs = DictWithTo(evaluator='stream_evaluate')
    def __init__(self, split: str, frame_fps: int, **kwargs):
        assert split in ['train', 'val', 'test']
        super().__init__(split=split, frame_fps=frame_fps, **kwargs)
        annos = []
        for (video_uid, video_start_time, video_end_time, query), query_region_times in tqdm.tqdm(self.annos.items()):
            duration = self.metadata[video_uid]['duration']
            conversation = []
            if video_start_time > duration or video_end_time > duration:
                continue
            video_end_time = ceil_time_by_fps(video_end_time, frame_fps, 0, duration)
            video_start_time = ceil_time_by_fps(video_start_time, frame_fps, 0, video_end_time)
            assert video_end_time > video_start_time
            last_time = video_start_time - 1 / frame_fps
            for query_start_time, query_end_time in query_region_times:
                query_start_time = ceil_time_by_fps(query_start_time, frame_fps, last_time + 1 / frame_fps, video_end_time)
                query_end_time = ceil_time_by_fps(query_end_time, frame_fps, query_start_time, video_end_time)
                if int((query_start_time - last_time) * frame_fps) <= 0:
                    break
                if int((query_end_time - query_start_time) * frame_fps) <= 0:
                    break
                conversation.extend([
                    {'role': 'stream', 'num_frames': int((query_start_time - last_time) * frame_fps), 'learn': True},
                    {'role': 'assistant', 'content': f"The video related to the query \"{query}\" starts.", 'learn': True},
                    {'role': 'stream', 'num_frames': int((query_end_time - query_start_time) * frame_fps), 'learn': True},
                    {'role': 'assistant', 'content': f"The video related to the query \"{query}\" ends.", 'learn': True},
                ])
                last_time = query_end_time
            if not conversation:
                continue
            annos.append({
                'query': query,
                'conversation': conversation,
                'load_ranges': {self.metadata[video_uid]['path']: range(int(video_start_time*frame_fps), int(last_time*frame_fps)+1)}
            })
        self.annos = annos

    def preprocess_conversation(self, conversation, query):
        query_prompt = random.choice(self.query_prompt_templates).replace('QUERY', query)
        return [{'role': 'user', 'content': query_prompt}] + conversation

    def __getitem__(self, index):
        anno = self.annos[index]
        return *super().__getitem__(
            conversation=self.preprocess_conversation(anno['conversation'], anno['query']),
            load_ranges=anno['load_ranges'],
        ), index, self.evaluation_kwargs

def build_ego4d_nlq_stream_train(**kwargs):
    return Ego4DStreamNLQ(split='train', **kwargs)

def build_ego4d_nlq_stream_val(**kwargs):
    return Ego4DStreamNLQ(split='val',**kwargs)

def build_ego4d_nlq_stream_test_unannotated(**kwargs):
    return Ego4DStreamNLQ(split='test', **kwargs)

if __name__ == '__main__':
    build_ego4d_nlq_stream_train(
        frame_fps=2, is_training=True, augmentation=True,
        system_prompt='', tokenizer=None,
        vision_pretrained='google/siglip-large-patch16-384',
        embed_mark='2fps_384_1+3x3'
    )
    build_ego4d_nlq_stream_val(
        frame_fps=2, is_training=True, augmentation=True,
        system_prompt='', tokenizer=None,
        vision_pretrained='google/siglip-large-patch16-384',
        embed_mark='2fps_384_1+3x3'
    )

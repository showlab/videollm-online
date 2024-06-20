import os, json, tqdm, torch
from transformers import CLIPImageProcessor

class COIN:
    root = 'datasets/coin'
    video_root = os.path.join(root, 'videos')
    def __init__(self, split: str, model_config: dict, fps: int, load_vision_embeds: bool, **kwargs):
        super().__init__(load_vision_embeds=load_vision_embeds, **kwargs)
        # 1. prepare load path
        vision_pretrained = model_config.vision_pretrained if not isinstance(model_config, dict) else model_config['vision_pretrained']
        frame_processor = CLIPImageProcessor.from_pretrained(vision_pretrained)
        crop_size = frame_processor.crop_size
        self.frames_dir = f"{self.video_root}_{fps}fps_{crop_size['height']}x{crop_size['width']}"
        if load_vision_embeds:
            self.frames_dir += '_' + vision_pretrained.replace('/', '--')

        # 2. prepare annos for all downstream benchmarks
        self.metadata = get_metadata(self.frames_dir, load_vision_embeds, fps)
        annos = json.load(open(os.path.join(self.root, 'coin.json')))['database']
        assert split in ['train', 'test']
        self._annos = [{
            'video_uid': video_uid,
            'task': COIN._clean_task(anno['class']),
            'start': anno['start'],
            'end': anno['end'],
            'steps': [dict(
                start=step['segment'][0],
                end=step['segment'][1],
                text=COIN._clean_step(step['label']),
            ) for step in anno['annotation']],
        } for video_uid, anno in annos.items() if (split in anno['subset'].lower()) and (video_uid in self.metadata)]
        self.annos: list[dict]

    @staticmethod
    def _clean_step(step):
        replaces = {
            'process (crop, fold) paper': 'crop and fold paper',
            'try to press gun head, spray residual old grease': 'try to press gun head to spray residual old grease'
        }
        return replaces.get(step, step)

    # PutOnHair -> put on hair
    @staticmethod
    def _clean_task(text):
        result = ''
        for char in text:
            if char.isupper():
                result += ' '  + char.lower()
            else:
                result += char
        result = result.replace(' t v', ' TV')
        result = result.replace(' c d', ' CD')
        result = result.replace('s i m', 'SIM')
        result = result.replace('n b a', 'NBA')
        result = result.replace('s s d','SSD')
        result = result.replace('r j45', 'RJ45')
        return result.strip()

    def __len__(self):
        return len(self.annos)

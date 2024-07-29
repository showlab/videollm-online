import os, json, tqdm, torch

class COIN:
    root = 'datasets/coin'
    video_root = os.path.join(root, 'videos')
    anno_root = os.path.join(root, 'annotations')
    def __init__(self, split: str, vision_pretrained: str, embed_mark: str, frame_fps: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dir = f"{self.video_root}_{embed_mark}_{vision_pretrained.replace('/', '--')}"
        self.frame_fps = frame_fps
        self.metadata = self.get_metadata()
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
        self.tasks_categories = list(set([v['task'].capitalize() + '.' for v in self._annos]))
        self.steps_categories = list(set([step['text'].capitalize() + '.' for steps in self._annos for step in steps['steps']]))
        self.annos: list[dict]

    def __len__(self):
        return len(self.annos)

    def get_metadata(self, ):
        metadata_path = f'{self.embed_dir}_metadata.json'
        if os.path.exists(metadata_path):
            print(f'load {metadata_path}...')
            metadata = json.load(open(metadata_path))
        else:
            metadata = {}
            for file in tqdm.tqdm(os.listdir(self.embed_dir), desc=f'prepare {metadata_path}...'):
                path = os.path.join(self.embed_dir, file)
                duration = (len(torch.load(path)) - 1) / self.frame_fps
                key = os.path.splitext(os.path.basename(path))[0]
                metadata[key] = {'duration': duration, 'path': path}
            json.dump(metadata, open(metadata_path, 'w'), indent=4)
        return metadata
    
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

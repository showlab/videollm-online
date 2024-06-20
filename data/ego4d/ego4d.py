import torch, os, json, tqdm

class Ego4D:
    root = 'datasets/ego4d/v2'
    video_root = os.path.join(root, 'full_scale')
    anno_root = os.path.join(root, 'annotations')
    def __init__(self, vision_pretrained: str, embed_mark: str, frame_fps: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dir = f"{self.video_root}_{embed_mark}_{vision_pretrained.replace('/', '--')}"
        self.frame_fps = frame_fps
        self.metadata = self.get_metadata()
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

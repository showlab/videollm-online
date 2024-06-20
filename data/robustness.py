import torch
from .stream import StreamMixIn

class Robustness(StreamMixIn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.annos = [ 
            {
                'conversation': [
                    {"role": "user", 'content': 'Please help me to narrate the video in real time.'},
                    {"role": "assistant", "content": 'Sure! Please provide me the video stream, and I will help you.', 'learn': True},
                ],
                'load_ranges': None,
            },
            {
                'conversation': [
                    {"role": "user", 'content': 'Concisely output what you have seen.'},
                    {"role": "assistant", "content": 'Sure! Please provide me the video inputs.', 'learn': True},
                ],
                'load_ranges': None,
            },
            {
                'conversation': [
                    {"role": "user", 'content': '...'},
                    {"role": "assistant", "content": 'Hi! How can I help you?', 'learn': True},
                ],
                'load_ranges': None,
            },
            {
                'conversation': [
                    {"role": "stream", 'num_frames': 10, 'learn': False},
                    {"role": "user", 'content': 'What is happening now?', 'learn': False},
                    {"role": "assistant", 'content': 'Sorry, it seems the input video frames are noise. Could you have a check?', 'learn': True},
                ],
                'load_ranges': torch.rand(1, 1024),
            },
            {
                'conversation': [
                    {"role": "stream", 'num_frames': 1, 'learn': False},
                    {"role": "user", 'content': 'What is the next step?', 'learn': False},
                    {"role": "assistant", 'content': 'Sorry, I cannot see anything in the input video frame. Could you check it?', 'learn': True},
                ],
                'load_ranges': torch.zeros(1, 1024),
            },
        ]

    def __getitem__(self, index):
        anno = self.annos[index]
        return *super().__getitem__(
            conversation=anno['conversation'],
            load_ranges=anno['load_ranges']
        ), index, {}

    def __len__(self):
        return len(self.annos)

def build_robustness(**kwargs):
    return Robustness(**kwargs)
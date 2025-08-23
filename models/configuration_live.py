
from transformers import PretrainedConfig

class LiveConfigMixin(PretrainedConfig):
    def __init__(self, *, vision_pretrained: str = None,
        frame_resolution: int = None, frame_token_cls: bool = None, frame_token_pooled: list[int] = None,
        v_placeholder: str = '<v>', frame_token_interval: str = None, v_placeholder_id: int = None, frame_token_interval_id: int = None, stream_loss_weight: float = 1.0,
        vision_drop_strategy: str = None, is_mod_weighted: bool = True, mod_warmup_steps: int = 0, is_return_vision_weights: bool = True,
        vision_hidden_size=1024, **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_pretrained = vision_pretrained
        self.frame_resolution = frame_resolution
        self.frame_token_cls = frame_token_cls
        self.frame_token_pooled = frame_token_pooled
        self.vision_hidden_size = vision_hidden_size
        self.stream_loss_weight = stream_loss_weight
        self.v_placeholder = v_placeholder
        self.frame_token_interval = frame_token_interval
        self.v_placeholder_id = v_placeholder_id
        self.frame_token_interval_id = frame_token_interval_id
        self.vision_drop_strategy = vision_drop_strategy
        self.is_mod_weighted = is_mod_weighted
        self.mod_warmup_steps = mod_warmup_steps
        self.is_return_vision_weights = is_return_vision_weights

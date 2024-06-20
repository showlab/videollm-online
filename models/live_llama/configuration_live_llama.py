
from transformers import LlamaConfig

from ..configuration_live import LiveConfigMixin

class LiveLlamaConfig(LlamaConfig, LiveConfigMixin):
    pass
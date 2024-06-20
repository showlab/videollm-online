import torch
from torch import nn
from transformers import LlamaForCausalLM, Cache
from transformers.activations import GELUActivation
from transformers.utils import logging

from .configuration_live_llama import LiveLlamaConfig
from ..modeling_live import build_live, LiveMixin

logger = logging.get_logger(__name__)

class LiveLlamaForCausalLM(LlamaForCausalLM, LiveMixin):
    config_class = LiveLlamaConfig
    _keys_to_ignore_on_load_missing = ['vision_encoder', 'connector']

    def __init__(self, config: LiveLlamaConfig):
        super().__init__(config)
        self.connector = torch.nn.Sequential(
            torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
            GELUActivation(config.hidden_size),
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames)
        outputs = super().forward(
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            # labels
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            cache_position=cache_position,
        )

        loss = None
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
            weight = v_mask * self.config.stream_loss_weight + ~v_mask
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
            loss = loss.sum() / (labels >= 0).sum()

        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs

        outputs.loss = loss
        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):  
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, past_length :] # NOTE

        # NOTE
        if inputs_embeds is not None and past_length < inputs_embeds.size(1):
            model_inputs = {"inputs_embeds": inputs_embeds[:, past_length:]}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids, # 长度为新的inputs，从past开始
                "cache_position": cache_position, # 没有被cache的区域
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask, # cache + input的长度
            }
        )
        return model_inputs
    
def build_live_llama(**kwargs):
    return build_live(config_class=LiveLlamaConfig, model_class=LiveLlamaForCausalLM, **kwargs)

if __name__ == '__main__':
    from ..arguments_live import LiveOnePlusTrainingArguments
    print(LiveOnePlusTrainingArguments().to_dict())
    model, tokenizer = build_live_llama(is_training=True, **LiveOnePlusTrainingArguments().to_dict())
    print(model.config, tokenizer)

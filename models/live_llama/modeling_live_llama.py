import math, time
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.activations import GELUActivation
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .configuration_live_llama import LiveLlamaConfig
from ..modeling_live import build_live, LiveMixin

logger = logging.get_logger(__name__)

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time} seconds")
        return result
    return wrapper

@dataclass
class BaseModelOutputWithPast(BaseModelOutputWithPast):
    vision_weights: Optional[list] = None

@dataclass
class CausalLMOutputWithPast(CausalLMOutputWithPast):
    vision_weights: Optional[list] = None

class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.layer_idx = layer_idx
        self.is_return_vision_weights = config.is_return_vision_weights
        self.frame_num_tokens = 1 + config.frame_token_pooled[0] * config.frame_token_pooled[1] if config.frame_token_pooled else 1
        if config.vision_drop_strategy:
            strategy, arg = config.vision_drop_strategy.split('_')
            if strategy == 'mod' and layer_idx % 2 != 0:
                self.router = nn.Linear(config.hidden_size, 1)
                self.vision_capacity = float(arg)
                self.is_mod_weighted = config.is_mod_weighted
                self.mod_warmup_steps = config.mod_warmup_steps
                self.training_step = 0
            if strategy == 'modall':
                self.router = nn.Linear(config.hidden_size, 1)
                self.vision_capacity = float(arg)
                self.is_mod_weighted = config.is_mod_weighted
                self.mod_warmup_steps = config.mod_warmup_steps
                self.training_step = 0
            if strategy == 'moddeep' and layer_idx % 2 != 0 and layer_idx > 1: # we denote deep layers as after layer2
                self.router = nn.Linear(config.hidden_size, 1)
                self.vision_capacity = float(arg)
                self.is_mod_weighted = config.is_mod_weighted
                self.mod_warmup_steps = config.mod_warmup_steps
                self.training_step = 0
            if strategy == 'moddeepall' and layer_idx > 1:
                self.router = nn.Linear(config.hidden_size, 1)
                self.vision_capacity = float(arg)
                self.is_mod_weighted = config.is_mod_weighted
                self.mod_warmup_steps = config.mod_warmup_steps
                self.training_step = 0
            if strategy in ['modrandom', 'moduniform'] and layer_idx % 2 != 0:
                self.vision_capacity = float(arg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        v_mask: Optional[torch.Tensor] = None,
        frame_interval_mask: Optional[torch.Tensor] = None,
        layer_type: str = None,
        all_vision_weights: Optional[list] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        return getattr(self, layer_type+'_forward')(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            v_mask=v_mask,
            frame_interval_mask=frame_interval_mask,
            all_vision_weights=all_vision_weights,
        )
    
    def full_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        return super().forward(
            hidden_states=hidden_states, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_value=past_key_value, 
            output_attentions=output_attentions, 
            use_cache=use_cache, 
            cache_position=cache_position, 
            **kwargs,
        )
    
    def mod_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        v_mask: Optional[torch.Tensor] = None,
        all_vision_weights: Optional[list] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # only support bs=1
        selected_mask = ~v_mask
        beam_num = hidden_states.shape[0]

        # for sequence has vision tokens
        if v_mask.any():
            # mod warmup for stable training
            # if self.router.training:
            #     vision_capacity = max(self.vision_capacity, 1- (1- self.vision_capacity) * (self.training_step / (self.mod_warmup_steps+1e-8)))
            #     self.training_step += 1
            # else:
            # vision_capacity = self.vision_capacity

            # Compute vision weights for each frame
            # NOTE: support beam search, but do not support batch_size > 1, since vision token is not aligned across batch but beam search does
            vision_hidden_states = hidden_states[0][v_mask[0]]
            vision_weights = self.router(vision_hidden_states.view(-1, self.frame_num_tokens, vision_hidden_states.shape[-1])).flatten(1, 2).sigmoid()
            # select topk vision tokens in each frame
            k = math.ceil(self.vision_capacity * self.frame_num_tokens)
            topk_indices = vision_weights.topk(k, dim=1, sorted=False).indices
            selected_weights_mask = torch.zeros_like(vision_weights, dtype=torch.bool).scatter_(1, topk_indices, True).flatten(0, 1).repeat(beam_num)
            selected_mask[v_mask] = selected_weights_mask

        processed_hidden_states = hidden_states.clone()
        selected_hidden_states = hidden_states[selected_mask].view(beam_num, -1, self.hidden_size)
        selected_position_ids = position_ids[selected_mask].view(beam_num, -1)
        selected_cache_position = cache_position[selected_mask[0]]

        block_outputs = self.full_forward(
            hidden_states=selected_hidden_states,
            attention_mask=attention_mask,
            position_ids=selected_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=selected_cache_position,
        )
        if use_cache:
            processed_hidden_states[selected_mask], present_key_value = block_outputs[0].flatten(0, 1), block_outputs[1]
        else:
            processed_hidden_states[selected_mask] = block_outputs[0]

        # multiply topk weights on selected vision tokens
        if self.is_mod_weighted and v_mask.any():
            selected_vision_mask = selected_mask ^ ~v_mask
            processed_hidden_states[selected_vision_mask] = processed_hidden_states[selected_vision_mask] * vision_weights.flatten(0, 1).repeat(beam_num)[selected_weights_mask].unsqueeze(-1)

        outputs = (processed_hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        if v_mask.any() and self.is_return_vision_weights and not self.training:
            all_vision_weights.append(vision_weights)

        return outputs
    
    def dropall_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        v_mask: Optional[torch.Tensor] = None,
        frame_interval_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # only support batch_size=1, beam search
        beam_num = hidden_states.shape[0]
        selected_mask = ~(v_mask | frame_interval_mask)
        if (~selected_mask).all():
            outputs = (hidden_states,)
            if use_cache:
                outputs += (None,)
            return outputs
        
        # selected_hidden_states, selected_position_ids, seleted_cache_position = hidden_states[selected_mask][None], position_ids[selected_mask][None], cache_position[selected_mask.squeeze()]
        selected_hidden_states = hidden_states[selected_mask].view(beam_num, -1, self.hidden_size)
        selected_position_ids = position_ids[selected_mask].view(beam_num, -1)
        selected_cache_position = cache_position[selected_mask[0]]


        processed_hidden_states = hidden_states.clone()

        block_outputs = self.full_forward(
            hidden_states=selected_hidden_states,
            attention_mask=attention_mask,
            position_ids=selected_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=selected_cache_position,
        )
        if use_cache:
            # processed_hidden_states[selected_mask], present_key_value = block_outputs
            processed_hidden_states[selected_mask], present_key_value = block_outputs[0].flatten(0, 1), block_outputs[1]
        else:
            processed_hidden_states[selected_mask] = block_outputs[0]

        outputs = (processed_hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    def mod_random_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        v_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # only support bs=1
        selected_mask = ~v_mask
        beam_num = hidden_states.shape[0]

        # for sequence has vision tokens
        if v_mask.any():
            # Compute vision weights for each frame
            # NOTE: support beam search, but do not support batch_size > 1, since vision token is not aligned across batch but beam search does
            # select topk vision tokens in each frame
            k = math.ceil(self.vision_capacity * self.frame_num_tokens)
            topk_indices = torch.randint(0, self.frame_num_tokens, (k,), device=selected_mask.device)[None]
            selected_weights_mask = torch.zeros((beam_num, v_mask[0].sum()), dtype=torch.bool, device=selected_mask.device).scatter_(1, topk_indices, True).flatten(0, 1).repeat(beam_num)
            selected_mask[v_mask] = selected_weights_mask

        processed_hidden_states = hidden_states.clone()
        selected_hidden_states = hidden_states[selected_mask].view(beam_num, -1, self.hidden_size)
        selected_position_ids = position_ids[selected_mask].view(beam_num, -1)
        selected_cache_position = cache_position[selected_mask[0]]

        block_outputs = self.full_forward(
            hidden_states=selected_hidden_states,
            attention_mask=attention_mask,
            position_ids=selected_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=selected_cache_position,
        )
        if use_cache:
            processed_hidden_states[selected_mask], present_key_value = block_outputs[0].flatten(0, 1), block_outputs[1]
        else:
            processed_hidden_states[selected_mask] = block_outputs[0]

        outputs = (processed_hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def mod_uniform_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        v_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # only support bs=1
        selected_mask = ~v_mask
        beam_num = hidden_states.shape[0]

        # for sequence has vision tokens
        if v_mask.any():
            # Compute vision weights for each frame
            # NOTE: support beam search, but do not support batch_size > 1, since vision token is not aligned across batch but beam search does
            # select topk vision tokens in each frame
            k = math.ceil(self.vision_capacity * self.frame_num_tokens)
            stride = self.frame_num_tokens // k
            topk_indices = torch.arange(0, stride * k, stride, device=selected_mask.device)[None]
            selected_weights_mask = torch.zeros((beam_num, v_mask[0].sum()), dtype=torch.bool, device=selected_mask.device).scatter_(1, topk_indices, True).flatten(0, 1).repeat(beam_num)
            selected_mask[v_mask] = selected_weights_mask

        processed_hidden_states = hidden_states.clone()
        selected_hidden_states = hidden_states[selected_mask].view(beam_num, -1, self.hidden_size)
        selected_position_ids = position_ids[selected_mask].view(beam_num, -1)
        selected_cache_position = cache_position[selected_mask[0]]

        block_outputs = self.full_forward(
            hidden_states=selected_hidden_states,
            attention_mask=attention_mask,
            position_ids=selected_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=selected_cache_position,
        )
        if use_cache:
            processed_hidden_states[selected_mask], present_key_value = block_outputs[0].flatten(0, 1), block_outputs[1]
        else:
            processed_hidden_states[selected_mask] = block_outputs[0]

        outputs = (processed_hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class LlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.vision_drop_strategy:
            strategy, arg = self.config.vision_drop_strategy.split('_')
            if strategy == 'mod':
                self.layers_type = ['full', 'mod'] * (config.num_hidden_layers // 2)
            elif strategy == 'modall':
                self.layers_type = ['mod'] * config.num_hidden_layers
            elif strategy == 'moddeep':
                self.layers_type = ['full'] * 2 + ['full', 'mod'] * (config.num_hidden_layers // 2 - 1)
            elif strategy == 'moddeepall': # default best setting
                self.layers_type = ['full'] * 2 + ['mod'] * (config.num_hidden_layers - 2)
            elif strategy == 'modrandom':
                self.layers_type = ['full', 'mod_random'] * (config.num_hidden_layers // 2)
            elif strategy == 'moduniform':
                self.layers_type = ['full', 'mod_uniform'] * (config.num_hidden_layers // 2)
            elif strategy == 'earlyexit':
                self.layers_type = ['full'] * config.num_hidden_layers
                for i in range(int(arg), config.num_hidden_layers):
                    self.layers_type[i] = 'dropall'
            elif strategy == 'layerskip':
                self.layers_type = ['full', 'dropall'] * (config.num_hidden_layers // 2)
            else:
                raise NotImplementedError
        else: #support original live
            self.layers_type = ['full'] * config.num_hidden_layers

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        v_mask: Optional[torch.Tensor] = None,
        frame_interval_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # for expert visualization
        all_vision_weights = []
        for decoder_layer, layer_type in zip(self.layers, self.layers_type):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    v_mask,
                    frame_interval_mask,
                    layer_type,
                    all_vision_weights,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    v_mask=v_mask,
                    frame_interval_mask=frame_interval_mask,
                    layer_type=layer_type,
                    all_vision_weights=all_vision_weights,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            vision_weights=all_vision_weights,
        )

class LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        v_mask: Optional[torch.Tensor] = None,
        frame_interval_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            v_mask=v_mask,
            frame_interval_mask=frame_interval_mask,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            vision_weights=outputs.vision_weights,
        )

class LiveLlamaForCausalLM(LlamaForCausalLM, LiveMixin):
    config_class = LiveLlamaConfig
    _keys_to_ignore_on_load_missing = ['vision_encoder', 'connector', 'router']

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
        v_mask: Optional[torch.Tensor] = None,
        frame_interval_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            v_mask = input_ids == self.config.v_placeholder_id
            frame_interval_mask = input_ids == self.config.frame_token_interval_id if self.config.frame_token_interval_id is not None else torch.zeros_like(input_ids, dtype=torch.bool)
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
            # fast args
            v_mask=v_mask,
            frame_interval_mask=frame_interval_mask,
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
        self, input_ids, v_mask, frame_interval_mask, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds, cache_position, **kwargs)
        model_inputs.update({"v_mask": v_mask, "frame_interval_mask": frame_interval_mask})
        return model_inputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor = None,
        frames: torch.Tensor = None,
        v_mask: Optional[torch.Tensor] = None,
        frame_interval_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        v_mask = input_ids == self.config.v_placeholder_id
        frame_interval_mask = input_ids == self.config.frame_token_interval_id if self.config.frame_token_interval_id is not None else torch.zeros_like(input_ids, dtype=torch.bool)
        inputs_embeds = self.joint_embed(input_ids, frames)
        output_ids = super().generate(
            inputs_embeds=inputs_embeds,
            v_mask=v_mask,
            frame_interval_mask=frame_interval_mask,
            **kwargs,
        )
        return output_ids

def build_live_llama(**kwargs):
    return build_live(config_class=LiveLlamaConfig, model_class=LiveLlamaForCausalLM, **kwargs)

if __name__ == '__main__':
    from ..arguments_live import LiveOnePlusTrainingArguments
    print(LiveOnePlusTrainingArguments().to_dict())
    model, tokenizer = build_live_llama(is_training=True, **LiveOnePlusTrainingArguments().to_dict())
    print(model.config, tokenizer)

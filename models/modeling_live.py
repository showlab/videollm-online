import torch, os
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, Cache
from transformers.utils import logging

from .tokenization_live import build_live_tokenizer_and_update_config
from .vision_live import build_live_vision

logger = logging.get_logger(__name__)

class LiveMixin(AutoModelForCausalLM):
    def set_vision_inside(self):
        logger.warning_once("!!! Set vision encoder in the model, only recommended for on in-the-wild inference. "
            "Please dont call this for efficient training & evaluation. Instead, do visual feature pre-extraction.")
        self.vision_encoder, self.vision_encode = build_live_vision(self.config)

    def unset_vision_inside(self):
        del self.vision_encoder
        del self.vision_encode

    def visual_embed(self, frames: torch.Tensor):
        if hasattr(self, 'vision_encode'):
            with torch.cuda.amp.autocast():
                frames = self.vision_encode(self.vision_encoder, frames)
            frames = frames.to(self.dtype)
        frames = self.connector(frames)
        return frames.view(-1, frames.shape[-1])

    def joint_embed(
        self,
        input_ids: torch.Tensor = None,
        frames: torch.Tensor = None,
    ):
        if frames is None:
            return self.get_input_embeddings()(input_ids)
        if input_ids is None:
            return self.visual_embed(frames)
        inputs_embeds = self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size-1))
        v_mask = input_ids == self.config.v_placeholder_id
        if v_mask.any():
            inputs_embeds[v_mask] = self.visual_embed(frames)
        return inputs_embeds

    @torch.no_grad()
    def stream_evaluate(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        frames: torch.ByteTensor,
        ignore_token_id: int = -100,
        frame_token_interval_threshold: float = 0.0,
        **kwargs
    ):
        # 0. evaluation only supports batch_size = 1
        assert input_ids.size(0) == labels.size(0) == 1
        input_id, label = input_ids[0], labels[0]
        device = input_id.device
        zero = torch.tensor(0, dtype=torch.int, device=device)
        one = torch.tensor(1, dtype=torch.int, device=device)

        # 1. prepare multi-turn start and stop
        turn_stops = ((input_id == self.config.eos_token_id).nonzero() + 1)[:,0].tolist()
        turn_starts = [0] + turn_stops[:-1]
        num_turns = len(turn_starts)

        # 2. forward the full input_ids and labels, get tokenwise logits and losses
        outputs = self.forward(input_ids=input_ids, frames=frames, return_dict=True, use_cache=True)
        logit, past_key_values = outputs.logits[0], outputs.past_key_values

        # 3. compute metrics for each turn
        v_placeholder_id = self.config.v_placeholder_id
        use_interval = self.config.frame_token_interval_id is not None
        frame_token_interval_id = self.config.frame_token_interval_id if use_interval else self.config.eos_token_id
        frame_num_tokens = self.config.frame_token_cls
        if self.config.frame_token_pooled:
            frame_num_tokens += self.config.frame_token_pooled[0] * self.config.frame_token_pooled[1]
        past_num_frames = 0
        lm_ppls, frame_diffs, fluencies, lm_correctness = [], [], [], []
        for r, (turn_start, turn_stop) in enumerate(zip(turn_starts, turn_stops)):
            ## 3.1. we only have two losses: stream loss on frame tokens, and lm loss. prepare corresponding mask according two losses
            turn_label = label[turn_start:turn_stop]
            turn_learn_mask = turn_label != ignore_token_id
            if not turn_learn_mask.any():
                continue
            turn_logit = logit[turn_start:turn_stop]
            turn_input_id = input_id[turn_start:turn_stop]
            turn_v_mask = turn_input_id == v_placeholder_id
            turn_num_frames = turn_v_mask.sum() // frame_num_tokens
            turn_stream_mask = turn_v_mask & turn_learn_mask
            turn_lm_mask = turn_learn_mask & ~turn_stream_mask

            ## 3.2 ppl, offline metric
            if turn_lm_mask.any():
                turn_lm_masked_logit, turn_lm_masked_label = turn_logit[turn_lm_mask], turn_label[turn_lm_mask]
                lm_ppl = torch.nn.functional.cross_entropy(turn_lm_masked_logit, turn_lm_masked_label).exp()
                lm_ppls.append(lm_ppl)
                turn_lm_masked_wrong_mask = turn_lm_masked_logit.argmax(dim=-1) != turn_lm_masked_label
                if turn_lm_masked_wrong_mask.any():
                    num_lm_correct_tokens = turn_lm_masked_wrong_mask.nonzero()[0,0]
                else:
                    num_lm_correct_tokens = (~turn_lm_masked_wrong_mask).sum()
                lm_correctness.append(num_lm_correct_tokens / turn_lm_masked_label.numel())

            ## 3.3. frame_diff (will be casted to time_diff in compute_metrics)
            if turn_stream_mask.any():
                ## 3.3.1: reply before (at) turn_num_frames
                turn_score = turn_logit.softmax(dim=-1)
                turn_stream_masked_score = turn_score[turn_stream_mask]
                if frame_token_interval_threshold > 0:
                    lower_threshold_mask = turn_stream_masked_score[:, frame_token_interval_id] < frame_token_interval_threshold
                    turn_stream_masked_score[lower_threshold_mask] = 0
                turn_stream_masked_pred_mask = turn_stream_masked_score.argmax(dim=-1) != frame_token_interval_id
                if turn_stream_masked_pred_mask.any():
                    frame_diff = turn_stream_mask.sum() - turn_stream_masked_pred_mask.nonzero()[0,0] - 1
                else:
                    ## 3.3.2: the most complex part,reply after turn_num_frames. we assume the 'assistant: ...' not exists
                    turn_last_stream_idx = turn_stream_mask.nonzero()[-1,0]
                    past_key_values_before_assistant = self.trim_past_key_values(past_key_values, 0, turn_start + turn_last_stream_idx + 1)
                    if r == num_turns - 1: # no future frame. we assume the model should receive a signal when streaming ends (e.g. close button).
                        frame_diff = zero
                    else:
                        next_turn_num_frames = (input_id[turn_starts[r+1]:turn_stops[r+1]] == v_placeholder_id).sum() // frame_num_tokens
                        to_append_num_frames = min(next_turn_num_frames, turn_num_frames - 1) # avoid bias. current as center, two equal left/right side
                        if to_append_num_frames == 0:
                            frame_diff = zero
                        else:
                            to_append_frames = frames[past_num_frames+turn_num_frames:past_num_frames+turn_num_frames+to_append_num_frames]
                            frame_placeholder = [v_placeholder_id] * frame_num_tokens
                            if use_interval:
                                frame_placeholder = [frame_token_interval_id] + frame_placeholder
                            to_append_input_id = torch.tensor(frame_placeholder * to_append_num_frames, dtype=torch.long, device=device)
                            to_append_logit = self.forward(
                                input_ids=to_append_input_id[None],
                                past_key_values=past_key_values_before_assistant,
                                frames=to_append_frames,
                                return_dict=True, use_cache=True
                            ).logits[0]
                            # we only use the last idx of each frame
                            idxs = torch.arange(len(frame_placeholder)-1, len(to_append_input_id), len(frame_placeholder), device=device)
                            to_append_score = to_append_logit[idxs].softmax(dim=-1)
                            if frame_token_interval_threshold > 0:
                                lower_threshold_mask = to_append_score[:, frame_token_interval_id] < frame_token_interval_threshold
                                to_append_score[lower_threshold_mask] = 0
                            to_append_score_pred_mask = to_append_score.argmax(dim=-1) != frame_token_interval_id
                            if to_append_score_pred_mask.any():
                                frame_diff = -(to_append_score_pred_mask.nonzero()[0,0] + 1)
                            else:
                                frame_diff = -to_append_num_frames
                frame_diffs.append(frame_diff.abs())

            ## 2.6 fluency
            if turn_lm_mask.any() and turn_stream_mask.any():
                num_learn_v_tokens = turn_stream_mask.sum()
                num_learn_valid_tokens = turn_lm_masked_label.numel() + num_learn_v_tokens
                if frame_diff == 0:
                    fluency = (num_learn_v_tokens + num_lm_correct_tokens) / num_learn_valid_tokens
                elif frame_diff > 0:
                    fluency = (num_learn_v_tokens - frame_diff) / num_learn_valid_tokens
                else:
                    fluency = (num_learn_v_tokens - 1) / num_learn_valid_tokens
                fluencies.append(fluency)
            ## 2.7 next turn
            past_num_frames += turn_num_frames
        lm_ppl = torch.stack(lm_ppls).mean() if lm_ppls else one
        frame_diff = torch.stack(frame_diffs).float().mean() if frame_diffs else zero
        fluency = torch.stack(fluencies).float().mean() if fluencies else one
        lm_correctness = torch.stack(lm_correctness).float().mean() if lm_correctness else one
        return torch.stack([lm_ppl, frame_diff, fluency, lm_correctness])

    def trim_past_key_values(self, past_key_values, start, stop):
        return [[past_keys[:,:,start:stop], past_values[:,:,start:stop]] for past_keys, past_values in past_key_values]

def fast_greedy_generate(*, model: LiveMixin, inputs_embeds: torch.Tensor, past_key_values: Cache, eos_token_id: int, inplace_output_ids: torch.Tensor):
    for i in range(inplace_output_ids.size(1)):
        outputs = model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        new_token_id = outputs.logits[:, -1:].argmax(dim=-1)
        inplace_output_ids[:, i] = new_token_id
        if new_token_id == eos_token_id:
            break
        inputs_embeds = model.get_input_embeddings()(new_token_id)
    return inplace_output_ids[:, :i+1], past_key_values

def build_live(
    *,
    is_training: bool,
    config_class: type,
    model_class: type,
    llm_pretrained: str = None,
    finetune_modules: list[str] = None,
    lora_modules: str = None,
    lora_r: int = None,
    lora_alpha: int = None,
    set_vision_inside: bool = False,
    resume_from_checkpoint: str = '',
    attn_implementation: str = 'flash_attention_2',
    torch_dtype: str | torch.dtype = 'auto',
    **kwargs
):
    model = model_class.from_pretrained(llm_pretrained, config=config_class.from_pretrained(llm_pretrained, **kwargs), torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    tokenizer = build_live_tokenizer_and_update_config(llm_pretrained, model.config)
    if is_training:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            modules_to_save=finetune_modules,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=False)
        else:
            logger.warning(f'!!! Fail to load checkpoint: {resume_from_checkpoint}. Return a new initialized model.')
        if set_vision_inside:
            model.set_vision_inside()
        model.requires_grad_(False)
    return model, tokenizer

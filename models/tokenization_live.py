import torch
from transformers import AutoTokenizer
from functools import partial

from .configuration_live import LiveConfigMixin

def get_stream_placeholder_len(num_frames: int, model_config: LiveConfigMixin) -> str:
    return num_frames * model_config.frame_num_tokens * len(model_config.v_placeholder) + len(model_config.frame_token_interval) * (num_frames - 1)

def get_stream_placeholder_jinja2(model_config: LiveConfigMixin) -> str:
    return f"'{model_config.frame_token_interval}'.join([{model_config.frame_num_tokens} * '{model_config.v_placeholder}'] * message['num_frames'])"

def get_stream_learn_ranges(num_frames: int, model_config: LiveConfigMixin) -> torch.Tensor:
    len_frame_placeholder_with_interval = model_config.frame_num_tokens * len(model_config.v_placeholder) + len(model_config.frame_token_interval)
    intermediate_interval_idxs = torch.arange(
        len_frame_placeholder_with_interval,
        len_frame_placeholder_with_interval * num_frames + 1,
        len_frame_placeholder_with_interval
    ) - len(model_config.frame_token_interval)
    len_learn = len(model_config.frame_token_interval) if model_config.frame_token_interval else len(model_config.v_placeholder)
    learn_ranges = torch.stack([
        intermediate_interval_idxs,
        intermediate_interval_idxs + len_learn
    ], dim=1)
    return learn_ranges

def chat_template(self, stream_placeholder_jinja2: str):
    """
    system prompt
    [<v>,<v>,<v>]
    User: ...
    Assistant: ...</s>
    [<v>,<v>]
    Assistant: ...</s>
    User: ...
    Assistant: ...</s>
    """
    template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ bos_token + messages[0]['content'] + '\n' }}" # system
        "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{% if add_stream_query_prompt %}"
        "{{ ']\nUser: ' + message['content'] }}"
        "{% else %}"
        "{{ '\nUser: ' + message['content'] }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '\nAssistant: '  + message['content'] + eos_token }}"
        "{% elif message['role'] == 'stream' and message['num_frames'] > 0: %}"
        "{{ '\n[' + STREAM_PLACEHOLDER + ']' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '\nAssistant:' }}"
        "{% elif add_stream_prompt %}"
        "{{ '\n[' }}"
        "{% elif add_stream_generation_prompt %}"
        "{{ ']\nAssistant:' }}"
        "{% endif %}"
    )
    template = template.replace('STREAM_PLACEHOLDER', stream_placeholder_jinja2)
    return template

def chat_template_transition(tokenizer):
    return {
        (None, 'system'): tokenizer.bos_token,
        ('system', 'user'): '\n\nUser: ',
        ('system', 'stream'): '\n\n[',
        ('user', 'assistant'): '\nAssistant: ',
        ('user', 'stream'): '\n[',
        ('user', 'user'): '\nUser: ',
        ('assistant', 'user'): f'{tokenizer.eos_token}\nUser: ',
        ('assistant', 'stream'): f'{tokenizer.eos_token}\n[',
        ('stream', 'user'): ']\nUser: ',
        ('stream', 'assistant'): ']\nAssistant: ',
        'assistant': 'Assistant: ',
        'eos_token': tokenizer.eos_token,
    }

def chat_template_offsets(tokenizer):
    return {k:len(v) for k, v in chat_template_transition(tokenizer).items()}

def get_learn_ranges(conversation: list[dict], *, chat_template_offsets: dict[tuple, int], model_config: LiveConfigMixin):
    offset = 0
    learn_ranges = []
    last_role = None
    for message in conversation:
        role = message['role']
        offset += chat_template_offsets[(last_role, role)]
        last_role = role
        if role == 'stream':
            if message.get('learn', False):
                ranges = get_stream_learn_ranges(message['num_frames'], model_config) + offset
                # the last one has ]\n, should also consider \n
                ranges[-1, 1] += 1
                if not isinstance(message['learn'], bool):
                    ranges = ranges[:message['learn']]
                learn_ranges.extend([range(r[0], r[1]) for r in ranges])
            offset += get_stream_placeholder_len(message['num_frames'], model_config)
        else:
            if role == 'assistant':
                if message.get('learn', False):
                    learn_ranges.append(range(offset - chat_template_offsets['assistant'], offset + len(message['content']) + chat_template_offsets['eos_token']))
            offset += len(message['content'])
    return learn_ranges

def build_live_tokenizer_and_update_config(llm_pretrained: str, model_config: LiveConfigMixin) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(llm_pretrained, use_fast=True, padding_side='left')
    tokenizer.add_special_tokens({'additional_special_tokens': [model_config.v_placeholder]})
    v_placeholder_id = len(tokenizer) - 1
    if model_config.frame_token_interval:
        frame_token_interval_id = tokenizer.convert_tokens_to_ids(model_config.frame_token_interval)
    else:
        frame_token_interval_id = None
    tokenizer.pad_token = tokenizer.eos_token
    model_config.update(dict(v_placeholder_id=v_placeholder_id, frame_token_interval_id=frame_token_interval_id, eos_token_id=tokenizer.eos_token_id))
    tokenizer.chat_template = chat_template(tokenizer, get_stream_placeholder_jinja2(model_config))
    tokenizer.get_learn_ranges = partial(get_learn_ranges, chat_template_offsets=chat_template_offsets(tokenizer), model_config=model_config)
    return tokenizer

if __name__ == '__main__':
    config = LiveConfigMixin(frame_token_interval=',', frame_token_cls=True, frame_token_pooled=[3,3], frame_num_tokens=10)
    tokenizer = build_live_tokenizer_and_update_config('meta-llama/Meta-Llama-3-8B-Instruct', config)
    chat = [
        {'role': 'system', 'content': 'cool.'},
        {'role': 'stream', 'num_frames': 2, 'learn': 1},
        {'role': 'user', 'content': 'cool?'},
        {'role': 'assistant', 'content': 'cool.', 'learn': True},
        {'role': 'stream', 'num_frames': 3, 'learn': 3},
        {'role': 'assistant', 'content': 'so cool.', 'learn': True},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    learn_ranges = tokenizer.get_learn_ranges(chat)
    batch = tokenizer([prompt], return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)
    batch_labels = torch.full_like(batch.input_ids, -100, dtype=torch.long)
    for text, labels, input_ids, offset_mapping, learn_range in zip(
        [prompt], batch_labels, batch.input_ids, batch.offset_mapping, [learn_ranges]
    ):
        for learn_r in learn_range:
            start = torch.nonzero(offset_mapping[:,0] == learn_r.start).item()
            if offset_mapping[:,0][-1] >= learn_r.stop:
                stop = torch.nonzero(offset_mapping[:,0] == learn_r.stop).item()
            else: # the last eos token
                stop = len(input_ids)
            labels[start-1:stop-1] = input_ids[start:stop]
            # NOTE: input_ids may out of boundary of len(tokenizer) - 1. (1 is the added vision placeholder)
            # this is because some frames has v_placeholder_id target. so replace it with eos token.
            labels[labels >= len(tokenizer) - 1] = tokenizer.eos_token_id
    print(batch.input_ids)
    print(batch_labels)
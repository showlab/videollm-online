import torch
from transformers import AutoTokenizer
from functools import partial

from .configuration_live import LiveConfigMixin

def get_stream_placeholder_len(num_frames: int, model_config: LiveConfigMixin) -> str:
    frame_num_tokens = int(model_config.frame_token_cls)
    if model_config.frame_token_pooled:
        frame_num_tokens += model_config.frame_token_pooled[0] * model_config.frame_token_pooled[1]
    return num_frames * frame_num_tokens * len(model_config.v_placeholder) + len(model_config.frame_token_interval) * (num_frames - 1)

def get_stream_placeholder_jinja2(model_config: LiveConfigMixin) -> str:
    frame_num_tokens = int(model_config.frame_token_cls)
    if model_config.frame_token_pooled:
        frame_num_tokens += model_config.frame_token_pooled[0] * model_config.frame_token_pooled[1]
    return f"'{model_config.frame_token_interval}'.join([{frame_num_tokens} * '{model_config.v_placeholder}'] * message['num_frames'])"

def get_stream_learn_ranges(num_frames: int, model_config: LiveConfigMixin) -> torch.Tensor:
    frame_num_tokens = int(model_config.frame_token_cls)
    if model_config.frame_token_pooled:
        frame_num_tokens += model_config.frame_token_pooled[0] * model_config.frame_token_pooled[1]
    len_frame_placeholder_with_interval = frame_num_tokens * len(model_config.v_placeholder) + len(model_config.frame_token_interval)
    intermediate_interval_idxs = torch.arange(
        len_frame_placeholder_with_interval,
        len_frame_placeholder_with_interval * num_frames,
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
    User: xxxx
    Assistant: ...</s><v>Assistant: ...</s><v>|<v>|<v>Assistant: ...
    User: xxxx
    ...
    """
    template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ bos_token + messages[0]['content'] + '\n' }}" # system
        "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '\nUser: ' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ 'Assistant: '  + message['content'] + eos_token }}"
        "{% elif message['role'] == 'stream' %}"
        "{{ STREAM_PLACEHOLDER }}"
        "{% else %}"
        "{{ raise_exception('Unknown role: ' + message['role']) }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ 'Assistant:' }}"
        "{% endif %}"
    )
    template = template.replace('STREAM_PLACEHOLDER', stream_placeholder_jinja2)
    return template

def chat_template_offsets(tokenizer):
    # last role, this role -> offset
    return {
        (None, 'system'): len(tokenizer.bos_token),
        ('system', 'user'): len('\n\nUser: '),
        ('system', 'stream'): len('\n'),
        ('user', 'assistant'): len('\nAssistant: '),
        ('user', 'stream'): len('\n'),
        ('user', 'user'): len('\nUser: '),
        ('assistant', 'user'): len(f'{tokenizer.eos_token}\nUser: '),
        ('assistant', 'assistant'): len(f'{tokenizer.eos_token}Assistant: '),
        ('assistant', 'stream'): len(tokenizer.eos_token),
        ('stream', 'user'): len('\nUser: '),
        ('stream', 'assistant'): len('Assistant: '),
        ('stream', 'stream'): 0,
        'assistant_prefix': len('Assistant: '),
        'assistant_postfix': len(tokenizer.eos_token),
    }

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
                learn_ranges.extend([range(r[0], r[1]) for r in ranges])
            offset += get_stream_placeholder_len(message['num_frames'], model_config)
        else:
            if role == 'assistant':
                if message.get('learn', False):
                    learn_ranges.append(range(offset - chat_template_offsets['assistant_prefix'], offset + len(message['content']) + chat_template_offsets['assistant_postfix']))
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
    if 'Llama-3' in llm_pretrained:
        tokenizer.eos_token = '<|eot_id|>'
    model_config.update(dict(v_placeholder_id=v_placeholder_id, frame_token_interval_id=frame_token_interval_id, eos_token_id=tokenizer.eos_token_id))
    tokenizer.chat_template = chat_template(tokenizer, get_stream_placeholder_jinja2(model_config))
    tokenizer.get_learn_ranges = partial(get_learn_ranges, chat_template_offsets=chat_template_offsets(tokenizer), model_config=model_config)
    return tokenizer

if __name__ == '__main__':
    config = LiveConfigMixin(frame_token_interval='|', frame_token_cls=True, frame_token_pooled=[3,3])
    tokenizer = build_live_tokenizer_and_update_config('meta-llama/Meta-Llama-3-8B-Instruct', config)
    chat = [
        {'role': 'system', 'content': 'cool.'},
        {'role': 'user', 'content': 'cool?'},
        {'role': 'stream', 'num_frames': 3, 'learn': True},
        {'role': 'assistant', 'content': 'cool.', 'learn': True},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    learn_ranges = tokenizer.get_learn_ranges(chat)
    print(prompt, learn_ranges)
    print([prompt[r.start:r.stop] for r in learn_ranges])

from transformers import HfArgumentParser

from .arguments_live import LiveTrainingArguments, get_args_class
from .live_llama import build_live_llama as build_model_and_tokenizer

def parse_args():
    args, = HfArgumentParser(LiveTrainingArguments).parse_args_into_dataclasses()
    args, = HfArgumentParser(get_args_class(args.live_version)).parse_args_into_dataclasses()
    return args
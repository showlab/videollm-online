import json, torch, tqdm, os, functools, submitit
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from dataclasses import dataclass

from models.arguments_live import LiveOnePlusTrainingArguments

@dataclass
class LiveOnePlusEncodingArguments(LiveOnePlusTrainingArguments):
    num_nodes: int = 1
    num_gpus: int = 8
    slurm_partition: str = None
    anno_root: str = 'datasets/ego4d/v2/annotations'
    split: str = 'train'

@torch.no_grad()
def distributed_refine_narration(args: LiveOnePlusEncodingArguments):
    env = submitit.JobEnvironment()
    torch.cuda.set_device(env.local_rank)
    
    model = AutoModelForCausalLM.from_pretrained(args.llm_pretrained, torch_dtype='auto', attn_implementation='sdpa')
    tokenizer = AutoTokenizer.from_pretrained(args.llm_pretrained, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.to('cuda')
    generator = functools.partial(model.generate, max_new_tokens=64, do_sample=False, top_p=1.0, temperature=1.0, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    
    anno_path = os.path.join(args.ego4d_anno_root, f'narration_stream_{args.split}.json')
    save_dir = os.path.join(args.ego4d_anno_root, f'refined_narration_stream_{args.split}')
    annos = json.load(open(anno_path))
    os.makedirs(save_dir, exist_ok=True)
    mapping = {}

    annos = {video_uid: _annotation_uid_narrations for i, (video_uid, _annotation_uid_narrations) in tqdm.tqdm(enumerate(annos.items())) if not os.path.exists(os.path.join(save_dir, f'{video_uid}.json'))}  
    for i, (video_uid, _annotation_uid_narrations) in tqdm.tqdm(enumerate(annos.items())):
        if i % env.num_tasks != env.global_rank:
            continue
        save_path = os.path.join(save_dir, f'{video_uid}.json')
        for _annotation_uid, narrations in _annotation_uid_narrations.items():
            for narration in narrations:
                if narration['text'] not in mapping:
                    chat = [
                        {
                            "role": "user", "content": ("Please help me to refine the text, e.g., [C looks around.] -> [You look around.]"
                            "In the text, There are many uppercase letters to denote persons. Rewrite the sentence to avoid these uppercase letters, improve the text quality, make the text clear and concise. "
                            "For example:\n[C looks around.] -> [You look around.]\n[A man X watches the phone.] -> [A man watches the phone.]\n"
                            f"[C plays a piano, and a woman O comes to him.] -> [You play a piano, and a woman comes to you.]\n[Man A approaches C] -> [A man approaches you.]\n\nNow, please refine [{narration['text']}] -> ?, make the answer in [].")
                        },
                        {"role": "assistant", "content": f"[{narration['text']}] -> ["}
                    ]
                    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors='pt')[:,:-1].cuda()
                    output_ids = generator(input_ids)[:, input_ids.size(1):]
                    text = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    try:
                        mapping[narration['text']] = text[:text.index(']')]
                    except:
                        print('fuck', narration['text'], text)
                        mapping[narration['text']] = 'Not sure what you are doing.'
                narration['text'] = mapping[narration['text']]
        
        json.dump(_annotation_uid_narrations, open(save_path, 'w'), indent=4)

if __name__ == "__main__":
    args, = HfArgumentParser(LiveOnePlusEncodingArguments).parse_args_into_dataclasses()
    executor = submitit.AutoExecutor(folder=f"outputs/preprocess/", cluster='local' if args.num_nodes == 1 else 'slurm')
    executor.update_parameters(
        tasks_per_node=args.num_gpus,
        nodes=args.num_nodes,
        gpus_per_node=args.num_gpus,
        slurm_partition=args.slurm_partition,
        cpus_per_task=10,
        mem_gb=240,
        slurm_time='24:00:00',
        timeout_min=600,
    )
    job = executor.submit(distributed_refine_narration, args)

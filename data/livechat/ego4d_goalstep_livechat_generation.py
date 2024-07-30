import json, torch, tqdm, os, submitit, random
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from dataclasses import dataclass, asdict

from models.arguments_live import LiveOnePlusTrainingArguments
from .templates import Templates
from ..utils import ceil_time_by_fps
from ..ego4d import Ego4D

@dataclass
class LiveOnePlusLiveChatGenerationArguments(LiveOnePlusTrainingArguments):
    num_nodes: int = 1
    num_gpus: int = 8
    num_queries_each_conversation: int = 3
    num_conversations_each_video: int = 10
    slurm_partition: str = None

class Ego4DLiveChatGeneration(Ego4D):
    @staticmethod
    def get_narrations(split):
        annos = []
        sources = json.load(open(os.path.join(Ego4D.anno_root, f'goalstep_{split}.json')))['videos']
        for source in sources:
            if source['segments']:
                annos.append({
                    'video_uid': source['video_uid'],
                    'summary': (source['start_time'], source['end_time'], source['goal_description'].strip()),
                    'narrations': [(segment['start_time'], segment['end_time'], segment['step_description'].strip()) for segment in source['segments']],
                })
            for segment in source['segments']:
                if segment['segments']:
                    annos.append({
                        'video_uid': source['video_uid'],
                        'summary': (segment['start_time'], segment['end_time'], segment['step_description'].strip()),
                        'narrations': [(seg['start_time'], seg['end_time'], seg['step_description'].strip()) for seg in segment['segments']],
                    })
        return annos
    
    def __init__(self, num_queries_each_conversation: int, num_conversations_each_video: int, **kwargs):
        super().__init__(vision_pretrained=kwargs['vision_pretrained'], embed_mark=kwargs['embed_mark'], frame_fps=kwargs['frame_fps'])
        self.num_queries_each_conversation = num_queries_each_conversation
        self.num_conversations_each_video = num_conversations_each_video
        annos = Ego4DLiveChatGeneration.get_narrations('train') + Ego4DLiveChatGeneration.get_narrations('val')
        self.annos = []
        for anno in annos:
            prompt = 'A very intelligent multimodal assistant helps the user to do the following activities:\n\n'
            timestamps = []
            for narration in anno['narrations']:
                timestamps.append(narration[0])
                if len(narration) > 2:
                    timestamps.append(narration[1])
                prompt += self.to_text('', narration) + '\n'
            prompt += f"\nNow, please complete the conversation between user and assistant. Note that the assistant will actively provides clear, concise, real-time language assistance. The assistant does not know the absolute time. Sometimes the user may ask irrelevant questions, the assistant is very helpful and will also answer that."
            timestamps = [ceil_time_by_fps(t, self.frame_fps, 0, self.metadata[anno['video_uid']]['duration']) for t in timestamps]
            timestamps = sorted(list(set(timestamps)))
            self.annos.append({
                'video_uid': anno['video_uid'],
                'prompt': prompt,
                'timestamps': timestamps,
            })
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', torch_dtype='auto', attn_implementation='sdpa')
        self.model.to('cuda')
        self.model.eval()

    @torch.no_grad()
    def __call__(self, index):
        anno = self.annos[index]
        video_uid, timestamps, prompt = anno['video_uid'], anno['timestamps'], anno['prompt']
        for nt in range(self.num_conversations_each_video):
            user_times = (torch.rand(3) * (timestamps[-1] - timestamps[0]) + timestamps[0]).sort().values.tolist()
            user_times = [round(ut, 1) for ut in user_times]
            user_queries = random.sample(Templates.queries, self.num_queries_each_conversation)
            example = ''
            for ui in range(len(user_queries)):
                example += f"\n{user_times[ui]}s User: {user_queries[ui]}\n{user_times[ui]}s Assistant: ..."
                for i, t in enumerate(timestamps):
                    if t < user_times[ui]:
                        continue
                    if ui+1 < len(user_times) and t >= user_times[ui+1]: 
                        break
                    example += f"\n{t}s Assistant: ..."
            input_ids = self.tokenizer.apply_chat_template([
                {'role': 'user', 'content': prompt + '\n' + example},
            ], return_tensors='pt', add_generation_prompt=True).cuda()
            output_ids = self.model.generate(input_ids, max_length=8192)[:,input_ids.size(1):]
            text = self.tokenizer.decode(output_ids[0])
            lines = [t.replace('<|eot_id|>', '') for t in text.split('\n') if t and ('User:' in t or 'Assistant:' in t)]
            try:
                anno = {'video_uid': video_uid, 'conversation': []}
                for line in lines:
                    role = 'User' if 'User:' in line else 'Assistant'
                    role_index = line.index(role)
                    time = float(line[:role_index].rstrip(' s'))
                    content = line[role_index+len(role)+2:]
                    anno['conversation'].append({'role': role.lower(), 'content': content, 'time': time})
                os.makedirs(f'{Ego4D.anno_root}/livechat/', exist_ok=True)
                json.dump(anno, open(f'{Ego4D.anno_root}/livechat/{video_uid}_{index}_{nt}.json', 'w'), indent=4)
            except:
                print('\n---\n' + text + '\n---\n')
    
    @staticmethod
    def to_text(prefix: str, narration: list):
        assert len(narration) >= 2 and len(narration) <= 3
        if len(narration) == 2:
            text = f"{prefix}{narration[0]:.2f}s: {narration[1]}"
        else:
            text = f"{prefix}{narration[0]:.2f}s-{narration[1]:.2f}s: {narration[2]}"
        return text

def distributed_livechat_generation(args):
    env = submitit.JobEnvironment()
    torch.cuda.set_device(env.local_rank)
    generator = Ego4DLiveChatGeneration(**asdict(args))
    for i in tqdm.tqdm(range(len(generator))):
        if i % env.num_tasks != env.global_rank:
            continue
        generator(i)
    
if __name__ == "__main__":
    args, = HfArgumentParser(LiveOnePlusLiveChatGenerationArguments).parse_args_into_dataclasses()
    executor = submitit.AutoExecutor(folder=f"outputs/preprocess/", cluster='local' if args.num_nodes == 1 else 'slurm')
    executor.update_parameters(
        tasks_per_node=args.num_gpus,
        nodes=args.num_nodes,
        gpus_per_node=args.num_gpus,
        cpus_per_task=10,
        slurm_partition=args.slurm_partition,
        mem_gb=240,
        slurm_time='24:00:00',
        timeout_min=600,
    )
    job = executor.submit(distributed_livechat_generation, args)

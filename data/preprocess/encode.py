import submitit, functools, transformers
from dataclasses import asdict, dataclass
from models.vision_live import build_live_vision

from models.configuration_live import LiveConfigMixin
from models.arguments_live import LiveOnePlusTrainingArguments
from ..utils import distributed_encode

@dataclass
class LiveOnePlusEncodingArguments(LiveOnePlusTrainingArguments):
    num_nodes: int = 1
    num_gpus: int = 8
    video_dir: str = 'datasets/ego4d/v2/full_scale_2fps_384'
    slurm_partition: str = None

if __name__ == "__main__":
    args, = transformers.HfArgumentParser(LiveOnePlusEncodingArguments).parse_args_into_dataclasses()
    vision_config = LiveConfigMixin(**asdict(args))
    _, vision_encode = build_live_vision(vision_config)
    task = functools.partial(
        distributed_encode, src_root=args.video_dir, 
        vision_pretrained=args.vision_pretrained, 
        embed_mark=args.embed_mark, 
        vision_encode=vision_encode, 
        batch_size=256, save_bf16=True
    )
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
    job = executor.submit(task)
    job.results()
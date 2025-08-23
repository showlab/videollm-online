import submitit, functools, argparse
from models.vision_live import build_live_vision
from ..utils import distributed_encode

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--vision_pretrained', type=str, default='google/siglip-large-patch16-384')
    parser.add_argument('--ffmpeg', type=str, default='2fps_384')
    parser.add_argument('--tokens', type=str, default='1+3x3')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    _, vision_encode = build_live_vision(vision_pretrained=args.vision_pretrained, frame_strategy=f'{args.ffmpeg}_{args.tokens}')
    task = functools.partial(
        distributed_encode, src_root=f'datasets/ego4d/v2/full_scale_{args.ffmpeg}', 
        vision_pretrained=args.vision_pretrained, vision_encode=vision_encode, 
        batch_size=256, tokens=args.tokens, save_bf16=True
    )
    executor = submitit.AutoExecutor(folder=f"outputs/preprocess/")
    executor.update_parameters(
        tasks_per_node=8,
        nodes=args.nodes,
        gpus_per_node=8,
        cpus_per_task=10,
        slurm_partition='learnfair',
        slurm_constraint='volta32gb',
        mem_gb=240,
        slurm_time='24:00:00',
        # slurm_qos='eht_ava',
    )
    job = executor.submit(task)

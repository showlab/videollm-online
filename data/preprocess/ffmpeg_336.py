import submitit
from functools import partial
from ..utils import distributed_ffmpeg

nodes = 32
tasks_per_node = 8
gpus_per_node = 1

task = partial(distributed_ffmpeg, src_root='datasets/ego4d/v2/full_scale', resolution=336, pad='#000000', fps=2)

if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder=f"outputs/preprocess/")
    executor.update_parameters(
        tasks_per_node=tasks_per_node,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        cpus_per_task=80//tasks_per_node,
        slurm_partition='learnaccel',
        mem_gb=240,
        slurm_time='24:00:00',
        # slurm_qos='eht_ava',
    )
    job = executor.submit(task)

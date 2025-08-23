# VideoLLM-MoD (NeurIPS'24)

## Quick Start

Please refer to the core implementation in https://github.com/showlab/videollm-online/blob/5bb806e6ab070de2ef64d2a15ac80b37c36ae056/models/live_llama/modeling_live_llama.py#L121-L189

## Install

```
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install mpi4py
pip install transformers accelerate deepspeed peft editdistance tensorboard gradio
pip install flash-attn --no-build-isolation

pip install gpustat
```

```
conda update -y pytorch torchvision -c pytorch -c nvidia
conda update mpi4py
pip install --upgrade transformers accelerate deepspeed peft editdistance tensorboard gradio
pip uninstall -y flash-attn && pip install flash-attn --no-build-isolation
```

```
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xvf ffmpeg-release-amd64-static.tar.xz
rm ffmpeg-release-amd64-static.tar.xz
mv ffmpeg-6.1-amd64-static ffmpeg
```

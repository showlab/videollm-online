# VideoLLM-online

## Quick Start

Try our gradio demo here:

Or after installation, you can launch

or use CLI program (only support narration):

```
python -m apps.cli
```

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

## Released Models

### Model Cards

* Live-Llama3-8B-S1
    * LLM: Meta-Llama-3-8B-Instruct
    * Vision Strategy:
        * Frame Encoder: google/siglip-large-patch16-384
        * Frame Tokens: Only CLS token
        * Frame FPS: 2 for training, 10 for inference
        * Frame Resolution: max resolution 384, with zero-padding to keep aspect ratio
        * Supported video length: 60 minutes
    * Training Data: Ego4D Narration Stream 120K + Ego4D GoalStep Stream 55K + Ego4D NLQ Stream 18K

* Live-Llama3-8B-S1+3x3
    * LLM: Meta-Llama-3-8B-Instruct
    * Vision Strategy:
        * Frame Encoder: google/siglip-large-patch16-384
        * Frame Tokens: CLS token + 3x3 spatial tokens after average pooling (18 frames)
        * Frame FPS: 2 for training, 10 for inference
        * Frame Resolution: max resolution 384, with zero-padding to keep aspect ratio
        * Supported video length: 5 minutes
    * Training Data: Ego4D Narration Stream 120K + Ego4D GoalStep Stream 55K + Ego4D NLQ Stream 18K

## Reimplement Paper Experiments

### Prepare Video Frame Embeddings

### Ego4D Narration Stream Benchmark

### Ego4D LTA Benchmark

### COIN Related Benchmarks

We are sorry that the COIN-related YouTube video may violate Meta's policy. We did not release that. But if you want to do so, you can send an email to me at joyachen@u.nus.edu, and I will instruct you.

### Distributed Preprocess Video Frames for VideoLLM-online

#### Sample video frames to 2 FPS and max resolution 384 (with zero padding)

```
python -m data.preprocess.ffmpeg --num_gpus 8 --frame_fps 2 --frame_resolution 384 --num_tasks 16 --video_dir datasets/ego4d/v2/full_scale
```

- Please run the script in ```videollm-online/``` root folder.

- The results will be saved in a new folder with '{fps}fps_{resolution}' suffix. For example, ```datasets/ego4d/v2/full_scale -> datasets/ego4d/v2/full_scale_2fps_384```.

- Increase ```--num_tasks``` according to the CPU cores. 1/10 number of CPU cores is recommended.

- If you are on a cluster, you can set ```--num_nodes ... --slurm_partition ...``` to use them. The more nodes, the faster preprocessing.

#### Encode sampled 2fps_384 video frames

```
python -m data.preprocess.encode --num_gpus 8 --vision_pretrained google/siglip-large-patch16-384 --video_dir datasets/ego4d/v2/full_scale_2fps_384 
```

- Please run the script in ```videollm-online/``` root folder.

- The results will be saved in a new folder with '{embed_mark}_{model}' suffix. For example, ```datasets/ego4d/v2/full_scale_2fps_384 -> datasets/ego4d/v2/full_scale_2fps_384_1+3x3_google--siglip-large-patch16-384```.

- If you are on a cluster, you can set ```--num_nodes ... --slurm_partition ...``` to use them. The more nodes and GPUs, the faster preprocessing.

#### Narration Refinement

```
python -m data.preprocess.ego4d_narration_refinement --llm_pretrained meta-llama/Meta-Llama-3-8B-Instruct --anno_root datasets/ego4d/v2/annotations --split train

python -m data.preprocess.ego4d_narration_refinement --llm_pretrained meta-llama/Meta-Llama-3-8B-Instruct --anno_root datasets/ego4d/v2/annotations --split val
```

- Please run the script in ```videollm-online/``` root folder.

- The results will be saved in a new json of 'refined_narration_stream_{args.split}' name. For example, ```datasets/ego4d/v2/annotations/narration_stream_train.json -> datasets/ego4d/v2/annotations/refined_narration_stream_train.json```.

- If you are on a cluster, you can set ```--num_nodes ... --slurm_partition ...``` to use them. The more nodes and GPUs, the faster preprocessing.
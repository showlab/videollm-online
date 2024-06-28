## Distributed Streaming Dialogue Data Generation

This part describes how to generate streaming dialogue data on Ego4D GoalStep dataset.

### Download Ego4D GoalStep Annotation JSON

Try to download Ego4D annotations. Refer to [Ego4D](https://ego4d-data.org/docs/start-here/) for details.

After that, you can use symbolic link to ensure you have ego4d annotations as the following:

```
datasets/ego4d/v2/annotations/
├── ...
├── goalstep_train.json
├── goalstep_val.json
├── ...
```

### Run Streaming Data Generation Script

```
python -m data.ego4d.livechat.ego4d_goalstep_livechat_generation --num_gpus 8 --anno_root 
```

- Please run the script in ```videollm-online/``` root folder.

- If you are on a cluster, you can set ```--num_nodes ... --slurm_partition ...``` to use them. The more nodes and GPUs, the faster preprocessing.

- The results will be saved into ```datasets/ego4d/v2/annotations/livechat/```. 

### Filtering out Data expose ground-truth

We find sometimes the generated dialogue will expose the timestamp information in ground-truth annotation. The pattern is "second", "..s" appeared in generated assistant responses. Furthermore, we remove dialogue less than 1 minute and larger than 60 minutes.

See python [filter.py](filter.py) for details.

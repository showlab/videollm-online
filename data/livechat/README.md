## Distributed Streaming Dialogue Data Generation

This section describes how to generate streaming dialogue data on the Ego4D GoalStep dataset.

### Download Ego4D GoalStep Annotation JSON

Try to download Ego4D annotations. Refer to [Ego4D](https://ego4d-data.org/docs/start-here/) for details.

After that, you can use a symbolic link to ensure you have the Ego4D annotations as shown below:

```
datasets/ego4d/v2/annotations/
├── ...
├── goalstep_train.json
├── goalstep_val.json
├── ...
```

### Run Streaming Data Generation Script

```
python -m data.livechat.ego4d_goalstep_livechat_generation --num_gpus 1 --num_queries_each_conversation 3 --num_conversations_each_video 10
```

- Please run the script in ```videollm-online/``` root folder.

- If you are on a cluster, you can set ```--num_nodes ... --slurm_partition ...``` to use them. The more nodes and GPUs, the faster preprocessing.

- The results will be saved into ```datasets/ego4d/v2/annotations/livechat/```. 

### Filtering out Data expose ground-truth

We find that the generated dialogue may expose the timestamp information in the ground-truth annotations. The pattern is "second", "..s" appearing in the generated assistant responses. So we remove them. Furthermore, we remove dialogues less than 1 minute and longer than 60 minutes.

See python [filter.py](filter.py) for details.

import tqdm, torch, transformers
from torch.utils.data import DataLoader
from dataclasses import asdict

from models import build_model_and_tokenizer, parse_args
from data import build_concat_train_dataset, build_eval_dataset_dict, get_data_collator, get_compute_metrics_dict

# TOKENIZERS_PARALLELISM=False python -m test.dataloader --live_version live1+ --output_dir outputs/debug --train_datasets ego4d_refined_narration_val --eval_datasets ego4d_refined_narration_val --augmentation True --llm_pretrained meta-llama/Meta-Llama-3-8B-Instruct --attn_implementation sdpa

if __name__ == '__main__':
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(is_training=True, **asdict(args))
    train_dataset = build_concat_train_dataset(tokenizer=tokenizer, **asdict(args))
    eval_dataset_dict = build_eval_dataset_dict(tokenizer=tokenizer, **asdict(args))
    collator_fn = get_data_collator(tokenizer=tokenizer)
    compute_metrics_dict = get_compute_metrics_dict(dataset_dict=eval_dataset_dict, tokenizer=tokenizer)

    max_length = 0
    all_length = 0
    if train_dataset:
        dl = DataLoader(train_dataset, batch_size=1, collate_fn=collator_fn, shuffle=True, num_workers=16, drop_last=False)
        for batch in tqdm.tqdm(dl, desc=f'debug run for training'):
            length = batch.input_ids.size(1)
            max_length = max(max_length, length)
            all_length += length
            print(tokenizer.decode(batch.input_ids[0]))
            print('input', batch.input_ids[0, -1000:])
            print('label', batch.labels[0, -1000:])
        print('avg_length', all_length / len(train_dataset))
        print('max_length', max_length)

    if eval_dataset_dict:
        for dataset_name, dataset in eval_dataset_dict.items():
            dl = DataLoader(dataset, batch_size=1, collate_fn=collator_fn, shuffle=False, num_workers=16, drop_last=False)
            dummy_predictions, label_ids = [], []
            for i, batch in enumerate(tqdm.tqdm(dl, desc=f'debug run for evaluation')):
                length = (batch.labels != -100).sum()
                # print(tokenizer.decode(batch.input_ids[0]))
                dummy_predictions.append(torch.tensor(tokenizer('\n'.join(dataset.labels[batch.sample_idxs[0]])).input_ids[1:]))
                label_ids.append(batch.sample_idxs)
            print(compute_metrics_dict[dataset_name](
                transformers.EvalPrediction(
                    predictions=torch.nn.utils.rnn.pad_sequence(dummy_predictions, batch_first=True, padding_value=-100).numpy(),
                    label_ids=torch.cat(label_ids).numpy()
                )
            ))

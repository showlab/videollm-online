from dataclasses import asdict

from models import build_model_and_tokenizer, parse_args
from data import build_eval_dataset_dict, get_data_collator, get_compute_metrics_dict
from engine import TrainerWithGenToEval

def evaluate():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(is_training=False, **asdict(args))
    eval_dataset_dict = build_eval_dataset_dict(tokenizer=tokenizer, model_config=model.config, **asdict(args))
    data_collator = get_data_collator(tokenizer=tokenizer, model_config=model.config, **asdict(args))
    compute_metrics_dict = get_compute_metrics_dict(dataset_dict=eval_dataset_dict, tokenizer=tokenizer, **asdict(args))

    trainer = TrainerWithGenToEval(
        model=model, tokenizer=tokenizer,
        args=args,
        eval_dataset=eval_dataset_dict,
        data_collator=data_collator,
        compute_metrics=compute_metrics_dict,
    )

    metrics = {}
    for eval_dataset_name, eval_dataset in eval_dataset_dict.items():
        trainer.compute_metrics = compute_metrics_dict[eval_dataset_name]
        dataset_metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix=f"eval_{eval_dataset_name}",
        )
        metrics.update(dataset_metrics)
    print(metrics)

if __name__ == "__main__":
    evaluate()

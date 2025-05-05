from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


def apply_lora(model, args):
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args['rank'],
        lora_alpha=args['alpha'],
        lora_dropout=0.1
    )
    return get_peft_model(model, peft_config)

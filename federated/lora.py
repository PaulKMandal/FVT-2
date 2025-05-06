from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def apply_lora(model, args):
    """
    Wrap `model` in LoRA adapters targeting attention projections,
    then patch the underlying base_model.forward to drop text-only args.
    """
    # Always treat vision models as FEATURE_EXTRACTION
    peft_task = TaskType.FEATURE_EXTRACTION

    # Default attention modules on ViT if none specified
    target_modules = args.get('target_modules', ['query', 'key', 'value'])

    # Create LoRA config and apply
    peft_config = LoraConfig(
        task_type=peft_task,
        inference_mode=False,
        r=args['rank'],
        lora_alpha=args['alpha'],
        target_modules=target_modules,
        lora_dropout=args.get('lora_dropout', 0.1),
    )
    peft_model = get_peft_model(model, peft_config)

    # Patch the base_model.forward to remove text-only kwargs
    base = peft_model.base_model if isinstance(peft_model, PeftModel) else peft_model
    orig_base_forward = base.forward

    def patched_base_forward(*args, **kwargs):
        # Drop text-only arguments
        for key in ('input_ids', 'attention_mask', 'token_type_ids'):
            kwargs.pop(key, None)
        return orig_base_forward(*args, **kwargs)

    base.forward = patched_base_forward
    return peft_model

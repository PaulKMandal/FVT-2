from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def _patch_peft_forward(model: PeftModel):
    """
    Monkey‑patch the PeftModel.forward to drop text-only kwargs
    before calling the original forward (so base_model never sees input_ids).
    """
    orig_forward = model.forward

    def patched_forward(*args, **kwargs):
        # Remove any keys that ViTForImageClassification (or Detection) doesn't accept:
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            kwargs.pop(key, None)
        return orig_forward(*args, **kwargs)

    # Bind our patched_forward to the instance
    model.forward = patched_forward.__get__(model, model.__class__)


def apply_lora(model, args):
    """
    Wrap `model` in LoRA adapters targeting attention projections
    and patch its forward to drop text-specific args.
    """
    # Always treat vision models as feature-extraction tasks
    peft_task = TaskType.FEATURE_EXTRACTION

    # Default attention modules if none specified
    target_modules = args.get("target_modules", ["query", "key", "value"])

    peft_config = LoraConfig(
        task_type=peft_task,
        inference_mode=False,
        r=args["rank"],
        lora_alpha=args["alpha"],
        target_modules=target_modules,
        lora_dropout=args.get("lora_dropout", 0.1),
    )
    peft_model = get_peft_model(model, peft_config)

    # Now patch its forward so it never passes input_ids downstream
    _patch_peft_forward(peft_model)

    return peft_model

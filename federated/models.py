
from transformers import AutoModelForImageClassification, AutoModelForObjectDetection
from .lora import apply_lora

MODEL_MAP = {
    "resnet18": "microsoft/resnet-18",
    "vit":      "google/vit-base-patch16-224",
    "frcnn":    "blesot/Faster-R-CNN-Object-detection",
    "detr":     "facebook/detr-resnet-50"
}

def build_model(config):
    name = MODEL_MAP[config["model"]]
    if config["task"] == "classification":
        model = AutoModelForImageClassification.from_pretrained(name)
    else:
        model = AutoModelForObjectDetection.from_pretrained(name)

    if config.get("lora", False):
        model = apply_lora(model, config["lora_args"])
    return model

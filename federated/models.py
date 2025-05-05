import torch
from transformers import AutoModelForImageClassification, AutoModelForObjectDetection
from .lora import apply_lora

MODEL_MAP = {
    'resnet18': 'microsoft/resnet-18',
    'vit': 'google/vit-base-patch16-224',
    'frcnn': 'blesot/Faster-R-CNN-Object-detection',
    'detr': 'facebook/detr-resnet-50'
}


def build_model(config):
    model_name = MODEL_MAP[config['model']]
    if config['task'] == 'classification':
        model = AutoModelForImageClassification.from_pretrained(model_name)
    else:
        model = AutoModelForObjectDetection.from_pretrained(model_name)
    if config.get('lora', False):
        model = apply_lora(model, config['lora_args'])
    return model

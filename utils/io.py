import yaml
import logging
import io
import torch


def setup_logging():
    logging.basicConfig(level=logging.INFO)


def get_model_size(model):
    param_count = sum(p.numel() for p in model.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return {'param_count': param_count, 'disk_size_bytes': buffer.getbuffer().nbytes}

import torch

def aggregate_weights(states, method='average', sizes=None, lora=False):
    """Aggregate list of state_dicts. If lora=True, only aggregate LoRA adapter params; base stays from first client."""
    aggregated = {}
    keys = states[0].keys()
    def is_adapter_key(k):
        return 'lora_' in k or '.lora' in k

    for k in keys:
        if lora:
            if is_adapter_key(k):
                if method == 'average' or sizes is None:
                    aggregated[k] = torch.stack([s[k] for s in states], dim=0).mean(dim=0)
                else:
                    total = sum(sizes)
                    aggregated[k] = sum(states[i][k] * (sizes[i] / total) for i in range(len(states)))
            else:
                aggregated[k] = states[0][k]
        else:
            if method == 'average' or sizes is None:
                aggregated[k] = torch.stack([s[k] for s in states], dim=0).mean(dim=0)
            else:
                total = sum(sizes)
                aggregated[k] = sum(states[i][k] * (sizes[i] / total) for i in range(len(states)))
    return aggregated

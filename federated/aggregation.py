import torch

def aggregate_weights(states, method='average', sizes=None, lora=False):
    aggregated = {}
    total = sum(sizes) if sizes else None
    for k in states[0].keys():
        vals = [s[k] for s in states]
        first = vals[0]
        if not torch.is_floating_point(first) or (lora and '.lora' not in k and 'lora_' not in k):
            aggregated[k] = first
            continue
        stacked = torch.stack(vals, 0)
        if method == 'average' or not sizes:
            aggregated[k] = stacked.mean(0)
        else:
            agg = sum(vals[i] * (sizes[i]/total) for i in range(len(vals)))
            aggregated[k] = agg
    return aggregated

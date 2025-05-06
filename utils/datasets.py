from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, default_data_collator
from federated.models import MODEL_MAP

def get_data_loaders(config):
    ds = load_dataset(config['dataset'])
    train, test = ds['train'], ds['test']
    proc = AutoImageProcessor.from_pretrained(MODEL_MAP[config['model']])
    # Determine keep_in_memory flag
    keep_mem = bool(config.get('keep_in_memory', True))

    def prep(batch):
        imgs = batch.get('image', batch.get('img'))
        enc = proc(images=imgs, return_tensors='pt')
        batch['pixel_values'] = enc['pixel_values']
        return batch

    # Use map with controlled caching
    train = train.map(prep, batched=True, keep_in_memory=keep_mem)
    test  = test.map(prep, batched=True, keep_in_memory=keep_mem)
    train.set_format('torch', columns=['pixel_values', 'label'])
    test .set_format('torch', columns=['pixel_values', 'label'])

    # Federated shards
    shard = len(train) // config['n_federates']
    loaders = []
    for i in range(config['n_federates']):
        subset = train.select(range(i*shard, (i+1)*shard if i<config['n_federates']-1 else len(train)))
        loaders.append(
            DataLoader(subset,
                       batch_size=config['hyperparams'][config['task']][config['model']]['batch_size'],
                       shuffle=True,
                       collate_fn=default_data_collator)
        )
    loaders.append(
        DataLoader(test,
                   batch_size=config['hyperparams'][config['task']][config['model']]['batch_size'],
                   shuffle=False,
                   collate_fn=default_data_collator)
    )
    return loaders

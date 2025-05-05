from datasets import load_dataset
from torch.utils.data import DataLoader


def get_data_loaders(config):
    ds = load_dataset(config['dataset'])
    train = ds['train']
    test = ds['test']
    shard_size = len(train) // config['n_federates']
    loaders = []
    for i in range(config['n_federates']):
        start = i * shard_size
        end = start + shard_size if i < config['n_federates'] - 1 else len(train)
        subset = train.select(range(start, end))
        loaders.append(DataLoader(subset, batch_size=config['hyperparams'][config['task']][config['model']]['batch_size'], shuffle=True))
    loaders.append(DataLoader(test, batch_size=config['hyperparams'][config['task']][config['model']]['batch_size']))
    return loaders

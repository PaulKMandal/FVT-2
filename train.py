import argparse
import yaml
from federated.trainer import FederatedTrainer
from utils.io import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Training Runner")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--keep_in_memory", dest="keep_in_memory", action="store_true",
                        help="Use map with keep_in_memory=True to avoid disk writes")
    parser.add_argument("--no-keep_in_memory", dest="keep_in_memory", action="store_false",
                        help="Allow map to write cache to disk (keep_in_memory=False)")
    parser.set_defaults(keep_in_memory=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    # CLI override for keep_in_memory
    if args.keep_in_memory is not None:
        config['keep_in_memory'] = args.keep_in_memory
    setup_logging()
    trainer = FederatedTrainer(config)
    results = trainer.run()
    with open(config['output'], 'w') as f:
        yaml.dump(results, f)

if __name__ == '__main__':
    main()

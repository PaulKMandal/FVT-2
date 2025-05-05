import argparse
import yaml
from federated.trainer import FederatedTrainer
from utils.io import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Training Runner")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    # CLI overrides
    parser.add_argument("--task", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--n_federates", type=int)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--no-simulation", dest="simulation", action="store_false")
    parser.add_argument("--aggregation", type=str)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--local_epochs", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--output", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    # Apply CLI overrides to config
    for key, val in vars(args).items():
        if val is not None and key != 'config':
            config[key] = val
    setup_logging()

    trainer = FederatedTrainer(config)
    results = trainer.run()
    with open(config['output'], 'w') as f:
        yaml.dump(results, f)


if __name__ == '__main__':
    main()

import time
import torch
from .aggregation import aggregate_weights
from utils.datasets import get_data_loaders
from utils.io import get_model_size
from utils.metrics import compute_metrics
from .models import build_model

class FederatedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(config)
        self.client_loaders = get_data_loaders(config)
        # Support fractional local_epochs
        self.local_epochs = float(config.get('local_epochs', 1.0))

    def run(self):
        # Report initial model size
        size = get_model_size(self.model)
        print(f"Initial model size: {size}")
        start_time = time.time()

        # Initialize global weights
        global_state = self.model.state_dict()

        # Communication rounds
        for round_idx in range(1, int(self.config['rounds']) + 1):
            print(f"\n=== Round {round_idx}/{self.config['rounds']} ===")
            local_states = []
            local_losses = []

            # Local training on each client
            for client_idx, loader in enumerate(self.client_loaders[:-1], start=1):
                model_copy = build_model(self.config)
                model_copy.load_state_dict(global_state, strict=False)
                model_copy.to(self.device)

                # Train and capture average loss
                loss = self._local_train(model_copy, loader)
                local_states.append(model_copy.state_dict())
                local_losses.append(loss)
                print(f"Client {client_idx} avg loss: {loss:.4f}")

            # Aggregate client updates
            global_state = aggregate_weights(
                local_states,
                method=self.config['aggregation'],
                sizes=[len(l.dataset) for l in self.client_loaders[:-1]],
                lora=self.config.get('lora', False)
            )
            if self.config.get('lora', False):
                base_state = self.model.state_dict()
                base_state.update(global_state)
                global_state = base_state

            # Load and evaluate global model after each round
            self.model.load_state_dict(global_state, strict=False)
            metrics = compute_metrics(
                self.model,
                self.client_loaders[-1],
                self.config['task'],
                self.config['evaluation'][f"{self.config['task']}_metrics"],
                annotation_json=self.config.get('annotation_json')
            )
            print(f"Global metrics after round {round_idx}: {metrics}")

        # Final evaluation
        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.2f}s")
        print(f"Final metrics: {metrics}")

        return {
            'model_size': size,
            'time_elapsed': total_time,
            'metrics': metrics
        }

    def _local_train(self, model, loader):
        # Setup optimizer with numeric hyperparameters
        hp = self.config['hyperparams'][self.config['task']][self.config['model']]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(hp['learning_rate']),
            weight_decay=float(hp.get('weight_decay', 0.0))
        )
        model.train()

        total_loss = 0.0
        count = 0
        # Determine full and fractional epochs
        full_epochs = int(self.local_epochs)
        frac = self.local_epochs - full_epochs

        # Full epochs
        for _ in range(full_epochs):
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = getattr(outputs, 'loss', outputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1

        # Fractional epoch: run a fraction of the batches
        if frac > 0:
            num_batches = int(frac * len(loader))
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = getattr(outputs, 'loss', outputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1

        # Return average loss
        return total_loss / count if count > 0 else 0.0

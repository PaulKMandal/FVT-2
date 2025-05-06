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

    def run(self):
        size = get_model_size(self.model)
        print(f"Initial model size: {size}")
        start_time = time.time()
        global_state = self.model.state_dict()

        for round_idx in range(1, int(self.config['rounds']) + 1):
            print(f"\n=== Round {round_idx}/{self.config['rounds']} ===")
            local_states = []
            local_losses = []

            for client_idx, loader in enumerate(self.client_loaders[:-1], start=1):
                model_copy = build_model(self.config)
                model_copy.load_state_dict(global_state, strict=False)
                model_copy.to(self.device)
                loss = self._local_train(model_copy, loader)
                local_states.append(model_copy.state_dict())
                local_losses.append(loss)
                print(f"Client {client_idx} avg loss: {loss:.4f}")

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

            self.model.load_state_dict(global_state, strict=False)
            metrics = compute_metrics(
                self.model,
                self.client_loaders[-1],
                self.config['task'],
                self.config['evaluation'][f"{self.config['task']}_metrics"],
                annotation_json=self.config.get('annotation_json')
            )
            print(f"Global metrics: {metrics}\n")

        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.2f}s")
        return {
            'model_size': size,
            'time_elapsed': total_time,
            'metrics': metrics
        }

    def _local_train(self, model, loader):
        cfg = self.config['hyperparams'][self.config['task']][self.config['model']]
        lr = float(cfg['learning_rate'])
        wd = float(cfg.get('weight_decay', 0.0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        model.train()

        total_loss = 0.0
        count = 0
        epochs = float(self.config['local_epochs'])
        full_epochs = int(epochs)
        frac = epochs - full_epochs

        def train_batch(batch):
            # Explicitly whitelist inputs: pixel_values and labels
            inputs = {'pixel_values': batch['pixel_values'].to(self.device)}
            if 'labels' in batch:
                inputs['labels'] = batch['labels'].to(self.device)
            elif 'label' in batch:
                inputs['labels'] = batch['label'].to(self.device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            loss.backward()
            optimizer.step()
            return loss.item()

        # Full epochs
        for _ in range(full_epochs):
            for batch in loader:
                total_loss += train_batch(batch)
                count += 1

        # Fractional epoch
        if frac > 0:
            num_batches = int(frac * len(loader))
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                total_loss += train_batch(batch)
                count += 1

        return total_loss / max(count, 1)

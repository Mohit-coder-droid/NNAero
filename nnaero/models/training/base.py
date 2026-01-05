import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import os

class BaseTrainer(ABC):
    """
    Abstract Base Class for model training.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def compute_loss(self, batch: Any, epoch: int) -> Dict[str, torch.Tensor]:
        pass

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_metrics = {}

        for batch_idx, batch in enumerate(dataloader):
            batch = self._to_device(batch)
            
            self.optimizer.zero_grad()
            
            metrics = self.compute_loss(batch, epoch)
            loss = metrics["loss"]
            
            loss.backward()
            
            # Optional: Add gradient clipping here if needed
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()

            # Accumulate metrics for logging
            for k, v in metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + v.item()

        # Average metrics
        return {k: v / len(dataloader) for k, v in running_metrics.items()}

    def _to_device(self, data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, (list, tuple)):
            return [self._to_device(x) for x in data]
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        return data

    def save_checkpoint(self, epoch: int, name: str = "model.pth"):
        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}_{name}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        self.logger.info(f"Checkpoint saved to {path}")
        return path

    def fit(self, train_loader: DataLoader, epochs: int):
        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(train_loader, epoch)
            
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Epoch {epoch:03d} | {metric_str}")
            
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
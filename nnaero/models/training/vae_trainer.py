import torch.nn.functional as F
from nnaero.models.training.base import *
from dataclasses import dataclass

@dataclass
class VAETrainingConfig:
    """Hyperparameters and training settings."""
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 50
    kl_weight: float = 1.0  # Beta parameter in Beta-VAE
    kl_annealing: bool = True
    warmup_epochs: int = 10
    grad_clip: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "vae_model.pth"

class VAETrainer(BaseTrainer):
    """
    VAE-specific trainer implementing the Evidence Lower Bound (ELBO) objective.
    """
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        kl_weight: float = 1.0,
        kl_annealing: bool = True,
        warmup_epochs: int = 10,
        **kwargs
    ):
        super().__init__(model, optimizer, **kwargs)
        self.kl_weight = kl_weight
        self.kl_annealing = kl_annealing
        self.warmup_epochs = warmup_epochs

    def compute_loss(self, batch: torch.Tensor, epoch: int) -> Dict[str, torch.Tensor]:
        # Batch can be (data, labels) or just data
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        
        x = x.view(x.size(0), -1) 
        recon_x, mu, logvar = self.model(x)
        
        # 1. Reconstruction Loss (MSE for continuous data)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # 2. KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 3. KL Annealing Weight
        beta = self.kl_weight
        if self.kl_annealing:
            beta *= min(1.0, epoch / self.warmup_epochs)
            
        total_loss = (recon_loss + beta * kl_loss) / x.size(0)
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss / x.size(0),
            "kl_loss": kl_loss / x.size(0),
            "beta": torch.tensor(beta) # For logging tracking
        }
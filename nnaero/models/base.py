import torch
import torch.nn as nn
from typing import List, Union, Type

class BaseNetwork(nn.Module):
    # 1. Class-level registry shared by all instances
    _ACTIVATION_REGISTRY = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU
    }

    def __init__(self, input_dim: int, config: List[Union[int, str]]):
        super(BaseNetwork, self).__init__()
        self.input_dim = input_dim
        self.config = config
        
        # Build network immediately upon initialization
        self.model = self._build_network(input_dim, config)
        self.output_dim = self._get_last_dim(input_dim, config)

    @classmethod
    def register_activation(cls, name: str, module_class: Type[nn.Module]):
        """
        Registers a new activation function globally for this class.
        
        Args:
            name (str): The string key to use in the config list (e.g., 'swish').
            module_class (Type[nn.Module]): The class definition (not an instance).
        """
        if not issubclass(module_class, nn.Module):
            raise TypeError(f"Custom activation must be a subclass of nn.Module, got {module_class}")
        
        cls._ACTIVATION_REGISTRY[name.lower()] = module_class

    def _build_network(self, input_dim: int, config: List[Union[int, str]]) -> nn.Sequential:
        layers = []
        current_dim = input_dim

        for layer_spec in config:
            if isinstance(layer_spec, int):
                # Linear Layer
                layers.append(nn.Linear(current_dim, layer_spec))
                current_dim = layer_spec
            elif isinstance(layer_spec, str):
                # Activation
                layers.append(self._get_activation(layer_spec))
            else:
                raise ValueError(f"Unsupported layer specification: {layer_spec}")
        
        return nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Retrieves activation from the registry and instantiates it."""
        name = name.lower()
        if name not in self._ACTIVATION_REGISTRY:
            supported = list(self._ACTIVATION_REGISTRY.keys())
            raise ValueError(f"Activation '{name}' not found. Supported: {supported}")
        
        # Instantiate the class (e.g., nn.ReLU())
        return self._ACTIVATION_REGISTRY[name]()

    def _get_last_dim(self, input_dim, config):
        last_dim = input_dim
        for item in config:
            if isinstance(item, int):
                last_dim = item
        return last_dim

    def forward(self, x):
        return self.model(x)
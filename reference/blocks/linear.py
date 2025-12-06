# reference/blocks/linear.py
"""
Linear (Fully Connected): Projeção linear
Componente fundamental para embeddings e transformações

ENTRADA:  x[batch, seq_len, in_features]
SAÍDA:    y[batch, seq_len, out_features]

OPERAÇÃO:  y = x @ W^T + b

PARÂMETROS:
  - weight[out_features, in_features]
  - bias[out_features]
"""

import numpy as np


class Linear:
    """Camada linear: y = x @ W^T + b"""
    
    def __init__(self, in_features, out_features, seed=None):
        """
        Args:
            in_features: número de features de entrada
            out_features: número de features de saída
            seed: seed para reprodutibilidade
        """
        self.in_features = in_features
        self.out_features = out_features
        
        # Inicializar pesos com distribuição uniforme (Xavier)
        rng = np.random.RandomState(seed)
        limit = np.sqrt(6.0 / (in_features + out_features))
        
        self.weight = rng.uniform(-limit, limit, 
                                   size=(out_features, in_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
    
    def forward(self, x):
        """
        Args:
            x: array (..., in_features)
        
        Returns:
            y: array (..., out_features)
        """
        # Pode ter qualquer número de dimensões, desde que última = in_features
        return x @ self.weight.T + self.bias
    
    def set_weights(self, weight, bias):
        """Carregar pesos do PyTorch"""
        self.weight = weight.copy()
        self.bias = bias.copy()
    
    def get_weights(self):
        """Retorna cópia dos pesos"""
        return self.weight.copy(), self.bias.copy()


if __name__ == "__main__":
    # Teste rápido
    lin = Linear(in_features=4, out_features=8, seed=42)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = lin.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected: (2, 3, 8), Got: {y.shape}")

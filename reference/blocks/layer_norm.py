# reference/blocks/layer_norm.py
"""
LayerNorm: Normalização por camada
Componente fundamental para o Vision Transformer

ENTRADA:  x[batch, seq_len, dim] - valores em float32
SAÍDA:    y[batch, seq_len, dim] - valores normalizados em float32

PARÂMETROS:
  - weight[dim]: ganho (inicializado com 1)
  - bias[dim]: offset (inicializado com 0)
  - eps: pequeno valor para estabilidade numérica (default: 1e-6)
"""

import numpy as np


class LayerNorm:
    """LayerNorm em NumPy puro para use como golden reference"""
    
    def __init__(self, dim, eps=1e-6):
        """
        Args:
            dim: dimensão do último eixo
            eps: pequeno valor para evitar divisão por zero
        """
        self.dim = dim
        self.eps = eps
        
        # Parâmetros treináveis
        self.weight = np.ones(dim, dtype=np.float32)  # gamma
        self.bias = np.zeros(dim, dtype=np.float32)   # beta
    
    def forward(self, x):
        """
        Args:
            x: array (..., dim) - pode ter qualquer formato com último eixo = dim
        
        Returns:
            y: array (..., dim) - normalizado
        """
        # Calcular estatísticas sobre último eixo
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        
        # Normalizar
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Escalar e deslocar
        return x_norm * self.weight + self.bias
    
    def set_weights(self, weight, bias):
        """Utilitário para carregar weights do PyTorch"""
        self.weight = weight.copy()
        self.bias = bias.copy()
    
    def get_weights(self):
        """Retorna cópia dos pesos para verificação"""
        return self.weight.copy(), self.bias.copy()


if __name__ == "__main__":
    # Teste rápido
    ln = LayerNorm(dim=4)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = ln.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output mean (deve ser ~0): {y.mean(axis=-1)}")
    print(f"Output var (deve ser ~1): {y.var(axis=-1)}")

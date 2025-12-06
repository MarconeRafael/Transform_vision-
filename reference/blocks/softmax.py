# reference/blocks/softmax.py
"""
Softmax: Normalização exponencial
Componente crítico para attention mechanism

ENTRADA:  x[batch, heads, seq_len, seq_len] (scores após escala)
SAÍDA:    y[batch, heads, seq_len, seq_len] (probabilidades, soma = 1)

OPERAÇÃO:  
  y_i = exp(x_i) / sum_j(exp(x_j))
  
NUMERICAMENTE ESTÁVEL (com subtração do máximo):
  m = max(x)
  y_i = exp(x_i - m) / sum_j(exp(x_j - m))
"""

import numpy as np


class Softmax:
    """Softmax com estabilidade numérica"""
    
    def __init__(self, dim=-1):
        """
        Args:
            dim: dimensão sobre a qual fazer softmax (default: última = -1)
        """
        self.dim = dim
    
    def forward(self, x, mask=None):
        """
        Args:
            x: array de scores
            mask: opcional, valores muito negativos (ex: -1e9) para posições mascaradas
        
        Returns:
            y: array de probabilidades (sum = 1 ao longo de dim)
        """
        # Copiar para não modificar entrada
        scores = x.copy()
        
        # Aplicar mask aditivo se fornecido
        if mask is not None:
            scores = scores + mask
        
        # Subtração do máximo para estabilidade (log-sum-exp trick)
        max_val = np.max(scores, axis=self.dim, keepdims=True)
        scores_shifted = scores - max_val
        
        # Exponencial e normalização
        exp_scores = np.exp(scores_shifted)
        sum_exp = np.sum(exp_scores, axis=self.dim, keepdims=True)
        
        return exp_scores / (sum_exp + 1e-10)


if __name__ == "__main__":
    # Teste rápido
    sm = Softmax(dim=-1)
    x = np.random.randn(2, 4, 5, 5).astype(np.float32)
    y = sm.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Sum over last axis (deve ser 1): {y.sum(axis=-1)[0, 0]}")
    print(f"Min value (deve ser ~0): {y.min()}")
    print(f"Max value (deve ser ~1): {y.max()}")

# reference/blocks/gelu.py
"""
GELU (Gaussian Error Linear Unit): Função de ativação
Componente do MLP do Vision Transformer

ENTRADA:  x[batch, seq_len, dim]
SAÍDA:    y[batch, seq_len, dim]

FÓRMULA (aproximação tanh):
  y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""

import numpy as np


class GELU:
    """GELU ativação com aproximação tanh"""
    
    def forward(self, x):
        """
        Args:
            x: array de valores
        
        Returns:
            y: array com GELU aplicado
        """
        # Aproximação tanh (mais rápida que erf exato)
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        x_cubed = x ** 3
        cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x_cubed)))
        return x * cdf


if __name__ == "__main__":
    # Teste rápido
    gelu = GELU()
    x = np.linspace(-3, 3, 7).astype(np.float32)
    y = gelu.forward(x)
    
    print(f"Input: {x}")
    print(f"Output: {y}")
    print(f"Propriedade: para x=0, y deve ser ~0: {gelu.forward(np.array([0.0]))}")

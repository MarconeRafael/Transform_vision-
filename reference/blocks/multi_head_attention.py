# reference/blocks/multi_head_attention.py
"""
Multi-Head Self-Attention: Mecanismo de atenção
Componente central do Vision Transformer

ENTRADA:  
  - query, key, value: [batch, seq_len, dim]
SAÍDA:    
  - output: [batch, seq_len, dim]
  - attention_weights: [batch, num_heads, seq_len, seq_len]

OPERAÇÃO:
  1. Projeto Q, K, V para múltiplas cabeças
  2. Calcula scores: Q @ K^T / sqrt(dim_head)
  3. Aplica softmax
  4. Multiplica por V
  5. Concatena cabeças e projeta novamente
"""

import numpy as np
from softmax import Softmax


class MultiHeadAttention:
    """Multi-Head Self-Attention em NumPy"""
    
    def __init__(self, dim, num_heads, seed=None):
        """
        Args:
            dim: dimensão de embedding (deve ser divisível por num_heads)
            num_heads: número de cabeças de atenção
            seed: seed para reprodutibilidade
        """
        assert dim % num_heads == 0, "dim deve ser divisível por num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        
        # Projeções lineares
        rng = np.random.RandomState(seed)
        limit = np.sqrt(6.0 / (2 * dim))
        
        self.w_q = rng.uniform(-limit, limit, size=(dim, dim)).astype(np.float32)
        self.w_k = rng.uniform(-limit, limit, size=(dim, dim)).astype(np.float32)
        self.w_v = rng.uniform(-limit, limit, size=(dim, dim)).astype(np.float32)
        self.w_o = rng.uniform(-limit, limit, size=(dim, dim)).astype(np.float32)
        
        self.softmax = Softmax(dim=-1)
    
    def _split_heads(self, x):
        """
        Reshapa (batch, seq_len, dim) -> (batch, num_heads, seq_len, dim_head)
        """
        batch, seq_len, dim = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.dim_head)
        return x.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, dim_head)
    
    def _merge_heads(self, x):
        """
        Reshapa (batch, num_heads, seq_len, dim_head) -> (batch, seq_len, dim)
        """
        x = x.transpose(0, 2, 1, 3)  # (batch, seq_len, num_heads, dim_head)
        batch, seq_len, num_heads, dim_head = x.shape
        return x.reshape(batch, seq_len, self.dim)
    
    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: [batch, seq_len_q, dim]
            key: [batch, seq_len_kv, dim]
            value: [batch, seq_len_kv, dim]
            attn_mask: opcional, máscara aditiva ou booleana
        
        Returns:
            output: [batch, seq_len_q, dim]
            attn_weights: [batch, num_heads, seq_len_q, seq_len_kv]
        """
        # Projeções lineares
        Q = query @ self.w_q
        K = key @ self.w_k
        V = value @ self.w_v
        
        # Split em múltiplas cabeças
        Q = self._split_heads(Q)  # (batch, num_heads, seq_len_q, dim_head)
        K = self._split_heads(K)  # (batch, num_heads, seq_len_kv, dim_head)
        V = self._split_heads(V)  # (batch, num_heads, seq_len_kv, dim_head)
        
        # Calcula scores (com escala)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.dim_head)
        # scores: (batch, num_heads, seq_len_q, seq_len_kv)
        
        # Aplicar máscara se fornecida
        if attn_mask is not None:
            scores = scores + attn_mask[np.newaxis, np.newaxis, :, :]
        
        # Softmax para obter pesos de atenção
        attn_weights = self.softmax.forward(scores)
        
        # Aplica atenção aos values
        context = np.matmul(attn_weights, V)  # (batch, num_heads, seq_len_q, dim_head)
        
        # Merge cabeças
        context = self._merge_heads(context)  # (batch, seq_len_q, dim)
        
        # Projeção final
        output = context @ self.w_o
        
        return output, attn_weights
    
    def set_weights(self, w_q, w_k, w_v, w_o):
        """Carregar pesos do PyTorch"""
        self.w_q = w_q.copy()
        self.w_k = w_k.copy()
        self.w_v = w_v.copy()
        self.w_o = w_o.copy()
    
    def get_weights(self):
        """Retorna cópia dos pesos"""
        return (self.w_q.copy(), self.w_k.copy(), 
                self.w_v.copy(), self.w_o.copy())


if __name__ == "__main__":
    # Teste rápido
    attn = MultiHeadAttention(dim=64, num_heads=8, seed=42)
    
    batch, seq_len = 2, 10
    query = np.random.randn(batch, seq_len, 64).astype(np.float32)
    key = np.random.randn(batch, seq_len, 64).astype(np.float32)
    value = np.random.randn(batch, seq_len, 64).astype(np.float32)
    
    output, attn_weights = attn.forward(query, key, value)
    
    print(f"Query shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention sum (deve ser 1): {attn_weights[0, 0, 0].sum()}")

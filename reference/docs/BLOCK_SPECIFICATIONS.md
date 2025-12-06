# reference/docs/BLOCK_SPECIFICATIONS.md

# Especificações de Blocos para Verilog

Este documento define a interface e operação de cada bloco fundamental do Vision Transformer. Use como referência para implementar em Verilog/FPGA.

---

## 1. LINEAR (Fully Connected)

### Especificação
- **Operação**: `y = x @ W^T + b`
- **Entrada**: `x[batch, seq_len, in_features]`
- **Parâmetros**: `W[out_features, in_features]`, `b[out_features]`
- **Saída**: `y[batch, seq_len, out_features]`
- **Tipo de dado**: `float32`

### Datapath Verilog
```
Entrada x[i][j][k] (k = in_features)
  ↓
MAC loop: acumular W[l][k] * x[i][j][k] para k=0..in_features-1
  ↓
Somar bias: result += b[l]
  ↓
Saída y[i][j][l] (l = out_features)
```

### Requisitos de Hardware
- Multiplicador: `in_features` bits × `out_features` bits
- Acumulador: `out_features` (paralelismo = 1..out_features)
- Memória: W[out_features × in_features] + b[out_features]
- Latência: `ceil(in_features / paralelismo) + 1` ciclos

### Teste de Referência
```python
from reference.blocks.linear import Linear
lin = Linear(in_features=768, out_features=3072, seed=42)
x = np.random.randn(2, 196, 768).astype(np.float32)
y = lin.forward(x)
# Salvar em arquivo para testbench Verilog
```

---

## 2. LAYERNORM

### Especificação
- **Operação**: 
  ```
  mean = E[x]
  var = Var[x]
  x_norm = (x - mean) / sqrt(var + eps)
  y = x_norm * weight + bias
  ```
- **Entrada**: `x[batch, seq_len, dim]`
- **Parâmetros**: `weight[dim]`, `bias[dim]`, `eps=1e-6`
- **Saída**: `y[batch, seq_len, dim]`
- **Tipo de dado**: `float32`

### Datapath Verilog
```
Entrada: x[i][j][k]
  ↓
Redução 1: Calcular sum(x) e sum(x^2) over k
  ↓
Cálculo: mean = sum(x) / dim, var = (sum(x^2) - dim*mean^2) / dim
  ↓
Normalização: x_norm[k] = (x[k] - mean) / sqrt(var + eps)
  ↓
Escala: y[k] = x_norm[k] * weight[k] + bias[k]
  ↓
Saída: y[i][j][k]
```

### Requisitos de Hardware
- Redução: `O(log dim)` para soma (tree adder)
- Divisor de ponto flutuante: para calcular mean e 1/sqrt(var+eps)
- Multiplicadores: 2 (x_norm * weight, depois + bias)
- Latência: `O(log dim) + 3` ciclos

### Teste de Referência
```python
from reference.blocks.layer_norm import LayerNorm
ln = LayerNorm(dim=768)
x = np.random.randn(2, 196, 768).astype(np.float32)
y = ln.forward(x)
# Verificar: y.mean(axis=-1) ≈ 0, y.var(axis=-1) ≈ 1
```

---

## 3. SOFTMAX

### Especificação
- **Operação**: `y[i] = exp(x[i]) / sum_j(exp(x[j]))`
- **Entrada**: `x[batch, num_heads, seq_len, seq_len]` (scores)
- **Saída**: `y[batch, num_heads, seq_len, seq_len]` (probabilidades)
- **Restrição**: `sum(y[i], axis=-1) = 1` para todo batch e cabeça
- **Tipo de dado**: `float32`

### Datapath Verilog (com estabilidade numérica)
```
Entrada: scores[i][h][q][k]
  ↓
Redução 1: max_val = max(scores) over k
  ↓
Shifting: shifted[k] = scores[k] - max_val
  ↓
Exponencial: exp_vals[k] = exp(shifted[k])
  ↓
Redução 2: sum_exp = sum(exp_vals) over k
  ↓
Normalização: y[k] = exp_vals[k] / sum_exp
  ↓
Saída: y[i][h][q][k]
```

### Requisitos de Hardware
- Comparador para max (tree reduction, `O(log seq_len)`)
- Exponencial: LUT ou aproximação polinomial
- Divisor: para normalização final
- Latência: `O(log seq_len) + 2` ciclos

### Teste de Referência
```python
from reference.blocks.softmax import Softmax
sm = Softmax(dim=-1)
x = np.random.randn(2, 8, 196, 196).astype(np.float32)
y = sm.forward(x)
# Verificar: np.allclose(y.sum(axis=-1), 1.0)
```

---

## 4. GELU (Gaussian Error Linear Unit)

### Especificação
- **Operação**: `y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))`
- **Entrada**: `x[batch, seq_len, dim]`
- **Saída**: `y[batch, seq_len, dim]`
- **Tipo de dado**: `float32`

### Datapath Verilog
```
Entrada: x[i][j][k]
  ↓
Cálculo: x_cubed = x * x * x (3 multiplicadores sequenciais)
  ↓
Soma: temp = x + 0.044715 * x_cubed
  ↓
Multiplicação: temp2 = sqrt(2/π) * temp ≈ 1.8133 * temp
  ↓
Tanh: tanh_val = tanh(temp2) [LUT ou aproximação polinomial]
  ↓
Final: y = 0.5 * x * (1 + tanh_val)
  ↓
Saída: y[i][j][k]
```

### Requisitos de Hardware
- 3 multiplicadores (x³), ou 1 multiplicador sequencial
- 1 multiplicador (0.044715)
- 1 multiplicador (sqrt(2/π))
- tanh: LUT ou aproximação série de Taylor
- 1 multiplicador final (0.5)
- Latência: `4 + tanh_latency` ciclos

### Teste de Referência
```python
from reference.blocks.gelu import GELU
gelu = GELU()
x = np.random.randn(2, 196, 3072).astype(np.float32)
y = gelu.forward(x)
# Propriedade: y[x<0] < x[x<0] (maioria dos valores)
```

---

## 5. MULTI-HEAD ATTENTION

### Especificação
- **Entrada**: 
  - query: `[batch, seq_len_q, dim]`
  - key: `[batch, seq_len_k, dim]`
  - value: `[batch, seq_len_v, dim]`
- **Saída**: 
  - output: `[batch, seq_len_q, dim]`
  - attention_weights: `[batch, num_heads, seq_len_q, seq_len_k]`
- **Tipo de dado**: `float32`

### Operação Detalhada
```
1. Projeções lineares:
   Q = query @ W_q     [batch, seq_len_q, dim]
   K = key @ W_k       [batch, seq_len_k, dim]
   V = value @ W_v     [batch, seq_len_v, dim]

2. Split em múltiplas cabeças:
   Q_h = reshape e transpose → [batch, num_heads, seq_len_q, dim_head]
   K_h = reshape e transpose → [batch, num_heads, seq_len_k, dim_head]
   V_h = reshape e transpose → [batch, num_heads, seq_len_v, dim_head]

3. Calcula scores:
   scores = Q_h @ K_h^T / sqrt(dim_head)
            → [batch, num_heads, seq_len_q, seq_len_k]

4. Softmax (opcional: aplicar máscara antes):
   attn_weights = softmax(scores, dim=-1)

5. Aplica atenção:
   context = attn_weights @ V_h
             → [batch, num_heads, seq_len_q, dim_head]

6. Merge cabeças:
   context = reshape → [batch, seq_len_q, dim]

7. Projeção final:
   output = context @ W_o → [batch, seq_len_q, dim]
```

### Datapath Verilog (arquitetura pipelining)
```
Etapa 0: Lê query[i], key[j], value[j]
  ↓
Etapa 1: W_q @ query (LINEAR)
  ↓
Etapa 2: W_k @ key, W_v @ value (LINEAR)
  ↓
Etapa 3: Split em cabeças (RESHAPE)
  ↓
Etapa 4: Q_h @ K_h^T com escala (MATMUL + SCALE)
  ↓
Etapa 5: Softmax (SOFTMAX)
  ↓
Etapa 6: attn_weights @ V (MATMUL)
  ↓
Etapa 7: Merge cabeças (RESHAPE)
  ↓
Etapa 8: W_o @ context (LINEAR)
  ↓
Saída: output
```

### Requisitos de Hardware
- 4 módulos Linear (W_q, W_k, W_v, W_o)
- 1 módulo MultiplyMatmul (Q@K^T)
- 1 módulo Softmax
- 1 módulo MultiplyMatmul (attn@V)
- Memória: 4 × (dim × dim) + intermediários
- Latência: `~25-30 ciclos` com pipelining completo

### Teste de Referência
```python
from reference.blocks.multi_head_attention import MultiHeadAttention
attn = MultiHeadAttention(dim=768, num_heads=12, seed=42)
query = np.random.randn(2, 196, 768).astype(np.float32)
key = np.random.randn(2, 196, 768).astype(np.float32)
value = np.random.randn(2, 196, 768).astype(np.float32)
output, attn_weights = attn.forward(query, key, value)
# Verificar: attn_weights.sum(axis=-1) ≈ 1.0
```

---

## Padrões de Teste

### Verificações Obrigatórias para cada Bloco Verilog

1. **Dimensões de saída corretas**
   - Entrada/saída deve ter shapes especificados

2. **Precisão numérica**
   - Comparar com NumPy usando máximo erro relativo < 0.1%
   - Usar ponto flutuante de 32 bits ou fixed-point equivalente

3. **Casos extremos**
   - Valores muito grandes: `x = 1e6`
   - Valores muito pequenos: `x = 1e-6`
   - Valores negativos e positivos

4. **Propriedades matemáticas**
   - LayerNorm: saída deve ter média ~0 e variância ~1
   - Softmax: soma = 1 para cada linha
   - GELU: função contínua e diferenciável

---

## Estrutura de Teste Recomendada

```
reference/
├── blocks/
│   ├── layer_norm.py
│   ├── linear.py
│   ├── softmax.py
│   ├── gelu.py
│   └── multi_head_attention.py
├── testbenches/
│   ├── test_layer_norm.py
│   ├── test_linear.py
│   ├── test_softmax.py
│   ├── test_gelu.py
│   └── test_multi_head_attention.py
├── vectors/
│   ├── linear_test_vectors.npy
│   ├── softmax_test_vectors.npy
│   └── ...
└── docs/
    ├── BLOCK_SPECIFICATIONS.md (este arquivo)
    └── EQUIVALENCE_GUIDE.md
```

---

## Próximas Fases

1. **Blocos de Teste**: Usar scripts em `reference/testbenches/` para gerar vetores
2. **Implementação Verilog**: Um arquivo por bloco em `rtl/blocks/`
3. **Verificação**: Testbenches que comparam Verilog com NumPy
4. **Integração**: Combinar blocos em encoder/decoder completo

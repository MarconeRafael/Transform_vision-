# Transform Vision – Implementação de Vision Transformer para Hardware (Verilog/FPGA)

## 🎯 Objetivo

Implementar um **Vision Transformer (ViT)** completo e funcional em **Verilog/FPGA**, usando implementações NumPy como **golden reference** (modelo de verdade).

### Abordagem em 3 Camadas

```
┌─────────────────────────────────────────────────────────────────┐
│ CAMADA 1: ALTO NÍVEL (PyTorch)                                  │
│ • Validação da arquitetura                                      │
│ • Baseline comportamental                                       │
│ • Fácil de modificar e experimentar                            │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ CAMADA 2: REFERÊNCIA NUMÉRICA (NumPy Puro)                      │
│ • Golden reference para hardware                                │
│ • Determinística e reprodutível                                │
│ • Gera vetores de teste para Verilog                          │
│ • Cada bloco isolado com interface clara                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ CAMADA 3: IMPLEMENTAÇÃO HARDWARE (Verilog/FPGA)                 │
│ • Blocos sintetizáveis                                         │
│ • Validados contra NumPy                                       │
│ • Otimizados para área/velocidade                              │
│ • Pronto para síntese e deployment                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Estrutura do Repositório

```
Transform_vision-/
│
├── pythons/                          # Implementações em Python
│   ├── alto_nivel/                   # Nível alto (PyTorch)
│   │   ├── encoder.py               # Encoder completo
│   │   └── decoder.py               # Decoder completo
│   └── baixo_nivel/                 # Nível baixo (NumPy)
│       ├── encoder.py               # Encoder em NumPy
│       └── decoder.py               # Decoder em NumPy
│
├── reference/                        # GOLDEN REFERENCE para Hardware
│   ├── blocks/                       # Blocos individuais NumPy
│   │   ├── linear.py                # Camada linear: y = x @ W^T + b
│   │   ├── layer_norm.py            # Normalização: (x - mean) / sqrt(var)
│   │   ├── softmax.py               # Softmax: exp(x) / sum(exp(x))
│   │   ├── gelu.py                  # Ativação: GELU
│   │   └── multi_head_attention.py  # Atenção multi-cabeça
│   │
│   ├── testbenches/                 # Geração de vetores de teste
│   │   ├── generate_test_vectors.py # Script principal
│   │   ├── test_*.py                # Testes individuais (em progresso)
│   │   └── verify_equivalence.py    # Validação Verilog vs NumPy
│   │
│   ├── vectors/                      # Vetores de teste gerados
│   │   ├── linear_test_0.npz
│   │   ├── softmax_test_0.npz
│   │   └── ... (criados por generate_test_vectors.py)
│   │
│   └── docs/                         # Documentação detalhada
│       ├── BLOCK_SPECIFICATIONS.md  # Especificação de cada bloco
│       └── EQUIVALENCE_GUIDE.md     # Como validar equivalência
│
├── rtl/                              # Implementação em Verilog (EM DESENVOLVIMENTO)
│   ├── blocks/                       # Módulos Verilog individuais
│   │   ├── linear.v
│   │   ├── layer_norm.v
│   │   ├── softmax.v
│   │   ├── gelu.v
│   │   └── multi_head_attention.v
│   │
│   └── testbenches/                 # Testbenches em Verilog
│       ├── tb_linear.v
│       ├── tb_layer_norm.v
│       ├── tb_softmax.v
│       ├── tb_gelu.v
│       └── tb_attention.v
│
├── test_all.py                       # Script para testar todos os Python
├── VALIDACAO_TESTES.md               # Relatório de validação anterior
├── README.md                         # Este arquivo
└── LICENSE
```

---

## 🚀 Como Usar

### 1️⃣ Fase 1: Validar Implementações NumPy (CONCLUÍDO)

As implementações NumPy são a **verdade fundamental** para hardware.

```bash
# Testar todos os componentes NumPy
python3 test_all.py

# Resultado esperado:
# ✓ Encoder PyTorch   - Funcionando
# ✓ Decoder PyTorch   - Funcionando
# ✓ Encoder NumPy     - Funcionando
# ✓ Decoder NumPy     - Funcionando
```

### 2️⃣ Fase 2: Gerar Vetores de Teste (PARA VERILOG)

Cria arquivos `.npz` com entrada, pesos e saída esperada de cada bloco.

```bash
# Gerar TODOS os vetores de teste
python3 reference/testbenches/generate_test_vectors.py --block all --output reference/vectors/

# Gerar apenas um bloco específico
python3 reference/testbenches/generate_test_vectors.py --block linear --output reference/vectors/
python3 reference/testbenches/generate_test_vectors.py --block softmax --output reference/vectors/
```

**Saída**: Arquivos em `reference/vectors/`

```
linear_test_0.npz                    (entrada, pesos, saída esperada)
linear_test_0_metadata.json          (shapes e configurações)
softmax_test_0.npz
softmax_test_0_metadata.json
layer_norm_test_0.npz
...
```

### 3️⃣ Fase 3: Implementar Blocos em Verilog (PRÓXIMO PASSO)

Cada arquivo Verilog deve ser testado contra o arquivo `.npz` correspondente.

```bash
cd rtl/testbenches/

# Simular um bloco com Verilator
verilator --trace tb_linear.v -o sim_linear
./sim_linear

# Comparar com NumPy
python3 ../../reference/testbenches/verify_equivalence.py \
    ../../reference/vectors/linear_test_0.npz \
    output_linear_test_0.csv
```

---

## 📊 Especificação dos Blocos

### Blocos Fundamentais (implementados em NumPy, para converter para Verilog)

Cada bloco tem:
- ✓ Interface clara (entrada/saída/parâmetros)
- ✓ Implementação NumPy pura
- ✓ Vetores de teste gerados automaticamente
- ✓ Documentação de hardware (datapath, latência, área)

| Bloco | Entrada | Saída | Operação | Arquivo |
|-------|---------|-------|----------|---------|
| **Linear** | `[B, S, D_in]` | `[B, S, D_out]` | `y = x @ W^T + b` | `reference/blocks/linear.py` |
| **LayerNorm** | `[B, S, D]` | `[B, S, D]` | Normalização por camada | `reference/blocks/layer_norm.py` |
| **Softmax** | `[B, H, S, S]` | `[B, H, S, S]` | Normalização exponencial | `reference/blocks/softmax.py` |
| **GELU** | `[B, S, D]` | `[B, S, D]` | Ativação não-linear | `reference/blocks/gelu.py` |
| **MultiHeadAttention** | Q,K,V `[B, S, D]` | `[B, S, D]` | Atenção multi-cabeça | `reference/blocks/multi_head_attention.py` |

Cada bloco tem documentação completa em: **`reference/docs/BLOCK_SPECIFICATIONS.md`**

---

## 📖 Documentação Importante

### Para Implementadores Verilog

1. **`reference/docs/BLOCK_SPECIFICATIONS.md`**
   - Especificação detalhada de cada bloco
   - Operações matemáticas
   - Requisitos de hardware (multiplicadores, divisores, LUTs)
   - Latência e paralelismo sugeridos

2. **`reference/docs/EQUIVALENCE_GUIDE.md`**
   - Como validar que Verilog é equivalente a NumPy
   - Fluxo de teste completo
   - Tolerâncias de precisão numérica
   - Scripts de verificação

### Arquivos de Teste

```bash
# Carregar e inspecionar vetores de teste
python3 -c "
import numpy as np
data = np.load('reference/vectors/linear_test_0.npz')
print('Input x shape:', data['x'].shape)
print('Weight shape:', data['weight'].shape)
print('Output y shape:', data['y'].shape)
print('First element of y:', data['y'].flat[0])
"
```

---

## 🔄 Fluxo de Desenvolvimento Verilog

### Para cada bloco (ex: Linear)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Implementar rtl/blocks/linear.v                              │
│    • Interface: clock, reset, x, weight, bias, y                │
│    • Operação: multiplicação-acumulação (MAC)                  │
│    • Pipelined: 3-5 estágios típico                            │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Criar rtl/testbenches/tb_linear.v                            │
│    • Carregar entrada do arquivo .npz                          │
│    • Fornecer clock, reset, sinais de controle                │
│    • Capturar saída em arquivo (VCD ou CSV)                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Validar com NumPy                                             │
│    • Comparar saída Verilog com expected (do .npz)             │
│    • Erro relativo < 0.1%                                      │
│    • Checar propriedades (dimensões, ranges)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💡 Dicas Importantes

### 1. **Comece pelo Linear**
- Bloco mais simples
- Usa multiplicadores/acumuladores
- Padrão para todos os outros blocos

### 2. **Teste Individualmente Antes de Integrar**
```
✓ Linear + LayerNorm
  ├─ ✓ Attention (usa 4x Linear)
  ├─ ✓ GELU
  └─ ✓ Softmax
```

### 3. **Use Seeds Determinísticas**
```python
# NumPy com seed para reprodutibilidade
bloco = Linear(in_features=768, out_features=3072, seed=42)
```

### 4. **Salve Intermediários para Debug**
```verilog
// rtl/blocks/attention.v
assign debug_q = Q;           // Saída de projeção Q
assign debug_scores = scores; // Após matmul
assign debug_attn = attn_weights; // Após softmax
```

### 5. **Tipos de Dado**
- Use `float32` (IEEE 754 single precision) ou equivalente
- Fixed-point também funciona se bem calibrado
- Manter precisão de mantissa ~23-24 bits

---

## ✅ Checklist de Desenvolvimento

- [x] Implementação NumPy com blocos isolados
- [x] Documentação de especificações (BLOCK_SPECIFICATIONS.md)
- [x] Gerador de vetores de teste automático
- [x] Guia de equivalência (EQUIVALENCE_GUIDE.md)
- [ ] Implementar Linear em Verilog
- [ ] Implementar LayerNorm em Verilog
- [ ] Implementar Softmax em Verilog
- [ ] Implementar GELU em Verilog
- [ ] Implementar MultiHeadAttention em Verilog
- [ ] Validar cada bloco contra NumPy
- [ ] Integrar blocos em Encoder completo
- [ ] Integrar blocos em Decoder completo
- [ ] Síntese e simulação de timing
- [ ] Deploy em FPGA

---

## 📞 Contato / Ajuda

Estrutura: Projeto Transform Vision - 2025

### Arquivos Principais para Consultoria

1. **Para entender a arquitetura**: `README.md` (este arquivo) + `reference/docs/BLOCK_SPECIFICATIONS.md`
2. **Para gerar testes**: `python3 reference/testbenches/generate_test_vectors.py`
3. **Para validar Verilog**: `reference/testbenches/verify_equivalence.py`
4. **Para debug detalhado**: `reference/docs/EQUIVALENCE_GUIDE.md`

---

## 🏆 Status do Projeto

| Componente | Status | Localização |
|-----------|--------|------------|
| PyTorch Encoder/Decoder | ✅ Completo | `pythons/alto_nivel/` |
| NumPy Encoder/Decoder | ✅ Completo | `pythons/baixo_nivel/` |
| Blocos NumPy isolados | ✅ Completo | `reference/blocks/` |
| Gerador de vetores | ✅ Completo | `reference/testbenches/` |
| Documentação specs | ✅ Completo | `reference/docs/BLOCK_SPECIFICATIONS.md` |
| Documentação equivalência | ✅ Completo | `reference/docs/EQUIVALENCE_GUIDE.md` |
| Blocos Verilog | 🔨 Em desenvolvimento | `rtl/blocks/` |
| Testbenches Verilog | 🔨 Em desenvolvimento | `rtl/testbenches/` |
| Integração Verilog | ⏳ Pendente | `rtl/` |

---

## 📚 Referências

- Vision Transformer (ViT): https://arxiv.org/abs/2010.11929
- Verilog/SystemVerilog para HDL: IEEE 1364, IEEE 1800
- NumPy NumPy numerical computing: https://numpy.org
- Verilator (simulador): https://www.veripool.org/verilator/

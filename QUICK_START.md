# QUICK_START.md

# Quick Start - Transform Vision para Verilog

## 🎯 Objetivo

Você tem uma **referência NumPy completa** e **vetores de teste prontos**. Agora pode implementar em Verilog.

---

## ✅ Checklist: O que você tem agora

- [x] **Implementações NumPy isoladas** em `reference/blocks/`
  - Linear, LayerNorm, Softmax, GELU, MultiHeadAttention
  - Cada uma pode ser testada independentemente

- [x] **Vetores de teste gerados** em `reference/vectors/`
  - `linear_test_0.npz` ... `linear_test_2.npz`
  - `softmax_test_0.npz` ... `softmax_test_1.npz`
  - `layer_norm_test_0.npz` ... `layer_norm_test_1.npz`
  - `gelu_test_0.npz` ... `gelu_test_1.npz`
  - `attention_test_0.npz` ... `attention_test_1.npz`

- [x] **Documentação técnica completa**
  - `reference/docs/BLOCK_SPECIFICATIONS.md` - Spec de cada bloco
  - `reference/docs/EQUIVALENCE_GUIDE.md` - Como validar Verilog

- [x] **Scripts de validação** em `reference/testbenches/`
  - `generate_test_vectors.py` - para gerar novos vetores
  - `verify_equivalence.py` - para comparar Verilog com NumPy

---

## 🚀 Próximos Passos

### Passo 1: Escolher um bloco para começar

**Recomendação**: Comece com **Linear** (mais simples)

```bash
# Inspecionar vetor de teste
python3 -c "
import numpy as np
import json

# Carregar dados
data = np.load('reference/vectors/linear_test_0.npz')
with open('reference/vectors/linear_test_0_metadata.json') as f:
    meta = json.load(f)

print('=== LINEAR TEST 0 ===')
print('Config:', meta)
print('Input x shape:', data['x'].shape)
print('Weight shape:', data['weight'].shape)
print('Bias shape:', data['bias'].shape)
print('Output y shape:', data['y'].shape)
print('First output value:', data['y'].flat[0])
"
```

### Passo 2: Implementar em Verilog

```verilog
// rtl/blocks/linear.v

module linear #(
    parameter IN_FEATURES = 768,
    parameter OUT_FEATURES = 3072,
    parameter BATCH = 2,
    parameter SEQ_LEN = 196,
    parameter DATA_WIDTH = 32  // float32
)(
    input clk,
    input reset,
    input [DATA_WIDTH-1:0] x [BATCH][SEQ_LEN][IN_FEATURES],
    input [DATA_WIDTH-1:0] weight [OUT_FEATURES][IN_FEATURES],
    input [DATA_WIDTH-1:0] bias [OUT_FEATURES],
    output reg [DATA_WIDTH-1:0] y [BATCH][SEQ_LEN][OUT_FEATURES]
);

    // TODO: Implementar lógica
    // - Multiplicação: x[i][j][k] * weight[l][k]
    // - Acumulação: soma sobre k
    // - Soma de bias: result += bias[l]

endmodule
```

### Passo 3: Criar testbench

```verilog
// rtl/testbenches/tb_linear.v

module tb_linear;
    // Parâmetros
    localparam IN_FEATURES = 768;
    localparam OUT_FEATURES = 3072;
    localparam BATCH = 2;
    localparam SEQ_LEN = 196;
    
    // Sinais
    reg clk, reset;
    reg [31:0] x [BATCH][SEQ_LEN][IN_FEATURES];
    reg [31:0] weight [OUT_FEATURES][IN_FEATURES];
    reg [31:0] bias [OUT_FEATURES];
    wire [31:0] y [BATCH][SEQ_LEN][OUT_FEATURES];
    
    // DUT
    linear #(...) dut (
        .clk(clk),
        .reset(reset),
        .x(x),
        .weight(weight),
        .bias(bias),
        .y(y)
    );
    
    // Teste
    initial begin
        // Carregar dados do arquivo .npz
        // (implementar manualmente ou usar ferramenta de conversão)
        
        // Simulação
        reset = 1;
        #10 reset = 0;
        
        // Aplicar entradas
        // Esperar saídas
        
        // Verificar resultados
        
        $finish;
    end
    
endmodule
```

### Passo 4: Simular e Comparar

```bash
# Simular com Verilator
cd rtl/testbenches
verilator --trace tb_linear.v -o sim_linear
./sim_linear

# Comparar com NumPy
python3 ../../reference/testbenches/verify_equivalence.py \
    ../../reference/vectors/linear_test_0.npz \
    output_linear.csv
```

---

## 📋 Ordem Recomendada de Implementação

```
1. LINEAR (most fundamental)
   ├─ Input: x[batch, seq, in]
   ├─ Weights: W[out, in], b[out]
   └─ Output: y[batch, seq, out] = x @ W^T + b
   
2. LAYERNORM (used after linear)
   ├─ Requires: reductions (sum, mean, var)
   ├─ Requires: division (1/sqrt)
   └─ Output: normalized x with learnable scale/shift
   
3. SOFTMAX (used in attention)
   ├─ Requires: exponential (exp)
   ├─ Requires: reductions (max, sum)
   └─ Output: probabilities (sum = 1)
   
4. GELU (activation in MLP)
   ├─ Requires: tanh approximation
   ├─ Requires: polynomial evaluation
   └─ Output: gated linear unit
   
5. MULTIHEADATTENTION (combines 1-4)
   ├─ Uses: 4x LINEAR (W_q, W_k, W_v, W_o)
   ├─ Uses: SOFTMAX
   └─ Output: attention context
```

---

## 🔍 Como Usar os Vetores de Teste

### Formato .npz

```python
import numpy as np

# Carregar
data = np.load('reference/vectors/linear_test_0.npz')

# Acessar campos
x = data['x']                  # Entrada
weight = data['weight']        # Pesos
bias = data['bias']            # Bias
y_expected = data['y']         # Saída esperada (NumPy)

# Usar em teste
y_verilog = sua_implementacao_verilog(x, weight, bias)

# Comparar
error = np.abs(y_verilog - y_expected) / np.abs(y_expected)
max_error = np.max(error)
print(f"Max relative error: {max_error}")
if max_error < 1e-3:
    print("✓ PASS")
else:
    print("✗ FAIL")
```

### Metadados

```json
// reference/vectors/linear_test_0_metadata.json
{
  "name": "linear_test_0",
  "input_shapes": {
    "x": [2, 196, 768],
    "weight": [3072, 768],
    "bias": [3072],
    "y": [2, 196, 3072]
  },
  "dtypes": {
    "x": "float32",
    "weight": "float32",
    "bias": "float32",
    "y": "float32"
  },
  "in_features": 768,
  "out_features": 3072,
  "batch": 2,
  "seq_len": 196
}
```

---

## 📚 Documentos Essenciais

| Documento | Para quem | Conteúdo |
|-----------|-----------|----------|
| `PROJECT_STRUCTURE.md` | Visão geral | Estrutura do projeto inteiro |
| `BLOCK_SPECIFICATIONS.md` | Implementadores Verilog | Spec de cada bloco + hardware |
| `EQUIVALENCE_GUIDE.md` | Verificadores | Como testar Verilog vs NumPy |
| `QUICK_START.md` | Você agora | Passo-a-passo rápido |

---

## 🛠️ Comandos Úteis

### Gerar novos vetores (se mudou alguma config)

```bash
python3 reference/testbenches/generate_test_vectors.py --block linear
python3 reference/testbenches/generate_test_vectors.py --block all
```

### Testar blocos NumPy

```bash
# Teste de cada bloco
cd reference/blocks
python3 linear.py
python3 layer_norm.py
python3 softmax.py
python3 gelu.py
python3 multi_head_attention.py
```

### Inspecionar um vetor

```python
import numpy as np
import json

data = np.load('reference/vectors/linear_test_0.npz')
with open('reference/vectors/linear_test_0_metadata.json') as f:
    config = json.load(f)

print(f"Input shape: {data['x'].shape}")
print(f"Output shape: {data['y'].shape}")
print(f"Max input: {data['x'].max()}, Min input: {data['x'].min()}")
print(f"Max output: {data['y'].max()}, Min output: {data['y'].min()}")
```

---

## 🎓 Exemplo Completo: Implementar Linear em Verilog

### 1. Ler especificação

```
De BLOCK_SPECIFICATIONS.md:
- Operação: y[i,j,l] = sum_k(x[i,j,k] * W[l,k]) + b[l]
- Entrada: x[batch, seq_len, in_features]
- Saída: y[batch, seq_len, out_features]
- Tipo: float32
```

### 2. Implementar

```verilog
// rtl/blocks/linear.v
module linear #(
    parameter IN = 768,
    parameter OUT = 3072
)(
    input clk, reset,
    input [31:0] x, w, b,
    output [31:0] y
);
    
    reg [31:0] acc;
    
    // MAC (Multiply-Accumulate)
    always @(posedge clk) begin
        if (reset)
            acc <= 32'h0;
        else
            acc <= acc + (x * w);  // Usar multiplicador FP
    end
    
    assign y = acc + b;
    
endmodule
```

### 3. Simular

```verilog
// rtl/testbenches/tb_linear.v
module tb_linear;
    ...
    initial begin
        $dumpfile("linear.vcd");
        $dumpvars;
        
        // Teste
        test_vector_0();
        test_vector_1();
        test_vector_2();
        
        $finish;
    end
    
    task test_vector_0;
        // Carregar reference/vectors/linear_test_0.npz
        // Aplicar entradas
        // Verificar saídas
    endtask
endmodule
```

### 4. Validar

```bash
verilator --trace tb_linear.v -o sim_linear
./sim_linear
python3 reference/testbenches/verify_equivalence.py \
    reference/vectors/linear_test_0.npz \
    linear_output.csv
```

---

## ❓ FAQs

**P: Por onde começo?**
R: Leia `PROJECT_STRUCTURE.md` para visão geral, depois `BLOCK_SPECIFICATIONS.md` para detalhes técnicos.

**P: Como carrego dados do .npz em Verilog?**
R: Opção 1 (manual): Exportar para CSV e parser em Verilog
   Opção 2: Usar script Python para gerar inicializações $readmemh
   Opção 3: Usar testbench SystemVerilog com DPI-C

**P: Qual tipo de dado usar?**
R: float32 (IEEE 754 single precision) é o padrão recomendado.
   Fixed-point também funciona se calibrado corretamente.

**P: Como validar equivalência?**
R: Usar `verify_equivalence.py` - compara saída Verilog com NumPy esperada.
   Tolerância típica: erro relativo < 0.1% (10^-3)

**P: E se meu Verilog não passar?**
R: Veja `EQUIVALENCE_GUIDE.md` seção "Técnicas de Depuração".
   - Teste componentes individuais (multiplicador, acumulador)
   - Compare intermediários (Q, K, V em attention)
   - Verifique tipo de dado (precision loss)

---

## 🎯 Meta: Implementar Encoder Completo

Quando terminar todos os blocos:

```
rtl/blocks/linear.v                ✓
rtl/blocks/layer_norm.v            ✓
rtl/blocks/softmax.v               ✓
rtl/blocks/gelu.v                  ✓
rtl/blocks/multi_head_attention.v  ✓
        ↓
rtl/blocks/transformer_block.v     (combinação dos 5 acima)
        ↓
rtl/blocks/encoder.v               (stack de N transformer blocks)
        ↓
rtl/blocks/vit_encoder_top.v       (patch embedding + encoder + classificação)
```

---

**Boa sorte com a implementação Verilog!** 🚀

Para dúvidas específicas, consulte os documentos em `reference/docs/`.

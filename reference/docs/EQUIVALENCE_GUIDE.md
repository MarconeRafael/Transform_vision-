# reference/docs/EQUIVALENCE_GUIDE.md

# Guia de Equivalência NumPy → Verilog

Este documento mostra como validar que sua implementação Verilog é equivalente à referência NumPy.

---

## Fluxo de Verificação

```
1. Gerar vetores de teste em NumPy
   └─→ reference/vectors/*.npz

2. Executar testes em Verilog (simulação com Verilator)
   └─→ rtl/testbenches/tb_*.v → saídas.vcd

3. Comparar resultados NumPy vs Verilog
   └─→ Máximo erro relativo < 0.1%
```

---

## 1. Geração de Vetores de Teste

### Comando
```bash
cd reference/testbenches
python3 generate_test_vectors.py --block all --output ../vectors
```

### Saída
```
reference/vectors/
├── linear_test_0.npz              (entrada, pesos, saída esperada)
├── linear_test_0_metadata.json    (shapes e configurações)
├── linear_test_1.npz
├── linear_test_1_metadata.json
├── ...
├── attention_test_0.npz
└── attention_test_0_metadata.json
```

### Estrutura de um arquivo .npz

```python
# Carregar no Python/Verilog
data = np.load('linear_test_0.npz')

# Disponível:
data['x']           # Entrada [batch, seq_len, in_features]
data['weight']      # Pesos [out_features, in_features]
data['bias']        # Bias [out_features]
data['y']           # Saída esperada [batch, seq_len, out_features]
```

---

## 2. Formato de Teste em Verilog

### Padrão para cada bloco

```verilog
// rtl/testbenches/tb_linear.v
`timescale 1ns / 1ps

module tb_linear;
    // Parâmetros
    localparam IN_FEATURES = 768;
    localparam OUT_FEATURES = 3072;
    localparam BATCH = 2;
    localparam SEQ_LEN = 196;
    
    // Interfaces
    reg clk, reset;
    reg signed [31:0] x [BATCH][SEQ_LEN][IN_FEATURES];
    reg signed [31:0] weight [OUT_FEATURES][IN_FEATURES];
    reg signed [31:0] bias [OUT_FEATURES];
    wire signed [31:0] y [BATCH][SEQ_LEN][OUT_FEATURES];
    
    // Instanciar módulo DUT (Device Under Test)
    linear #(
        .IN_FEATURES(IN_FEATURES),
        .OUT_FEATURES(OUT_FEATURES),
        .BATCH(BATCH),
        .SEQ_LEN(SEQ_LEN)
    ) dut (
        .clk(clk),
        .reset(reset),
        .x(x),
        .weight(weight),
        .bias(bias),
        .y(y)
    );
    
    // Carregamento de vetores de teste
    initial begin
        // Carregar dados do arquivo .npz
        // (usar systemverilog ou parsear manualmente)
        load_test_vectors("../vectors/linear_test_0.npz");
        
        // Executar teste
        run_test();
        
        $finish;
    end
    
endmodule
```

---

## 3. Comparação de Resultados

### Script Python para validação

```python
#!/usr/bin/env python3
# reference/testbenches/verify_equivalence.py

import numpy as np
import json
import sys

def load_reference(npz_file):
    """Carrega resultado esperado (NumPy)"""
    data = np.load(npz_file)
    return data

def load_verilog_output(csv_file):
    """Carrega resultado do Verilog (conversão de VCD ou dump)"""
    # Implementar leitura de formato específico
    # Pode ser CSV, VCD, ou formato customizado
    pass

def compare_outputs(reference, verilog, tolerance=1e-3):
    """
    Compara outputs com tolerância
    
    Args:
        reference: np.ndarray saída NumPy
        verilog: np.ndarray saída Verilog
        tolerance: máximo erro relativo permitido
    
    Returns:
        (passou, erro_máximo, localização_máximo)
    """
    # Evitar divisão por zero
    ref_abs = np.abs(reference)
    ref_abs = np.where(ref_abs < 1e-10, 1.0, ref_abs)
    
    # Erro relativo
    error = np.abs(reference - verilog) / ref_abs
    
    # Encontrar máximo erro
    max_error = np.max(error)
    max_idx = np.unravel_index(np.argmax(error), error.shape)
    
    passed = max_error < tolerance
    
    return passed, max_error, max_idx

def main(npz_file, verilog_output):
    """Valida equivalência entre NumPy e Verilog"""
    
    print(f"Carregando referência: {npz_file}")
    data = load_reference(npz_file)
    
    print(f"Carregando saída Verilog: {verilog_output}")
    verilog_y = load_verilog_output(verilog_output)
    
    print("\n" + "=" * 60)
    print("VALIDAÇÃO DE EQUIVALÊNCIA")
    print("=" * 60)
    
    reference_y = data['y']
    
    passed, max_error, max_idx = compare_outputs(reference_y, verilog_y)
    
    print(f"\nSaída NumPy shape: {reference_y.shape}")
    print(f"Saída Verilog shape: {verilog_y.shape}")
    print(f"\nMáximo erro relativo: {max_error:.6e}")
    print(f"Localização: {max_idx}")
    print(f"Valor NumPy: {reference_y[max_idx]:.6e}")
    print(f"Valor Verilog: {verilog_y[max_idx]:.6e}")
    
    print("\n" + "=" * 60)
    if passed:
        print("✓ TESTE PASSOU - Implementações são equivalentes!")
    else:
        print("✗ TESTE FALHOU - Erro acima da tolerância")
    print("=" * 60)
    
    return 0 if passed else 1

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: python3 verify_equivalence.py <npz_file> <verilog_output>")
        sys.exit(1)
    
    sys.exit(main(sys.argv[1], sys.argv[2]))
```

### Uso
```bash
python3 reference/testbenches/verify_equivalence.py \
    reference/vectors/linear_test_0.npz \
    results/linear_test_0_verilog_output.csv
```

---

## 4. Checklist de Validação por Bloco

### [ ] LINEAR
- [ ] Dimensões de saída: `y.shape == (batch, seq_len, out_features)`
- [ ] Valores numéricos: erro relativo < 0.1%
- [ ] Casos extremos: `x = ±1e6` e `x = ±1e-6`

### [ ] LAYER_NORM
- [ ] Dimensões de saída: `y.shape == x.shape`
- [ ] Valores numéricos: erro relativo < 0.5% (mais sensível)
- [ ] Propriedade: `y.mean(axis=-1) ≈ 0`, `y.var(axis=-1) ≈ 1`

### [ ] SOFTMAX
- [ ] Dimensões de saída: `y.shape == x.shape`
- [ ] Propriedade: `y.sum(axis=-1) ≈ 1` para todas as linhas
- [ ] Range: `0 <= y <= 1`

### [ ] GELU
- [ ] Dimensões de saída: `y.shape == x.shape`
- [ ] Valores numéricos: erro relativo < 0.5%
- [ ] Propriedade de simetria: `gelu(-x) ≈ -gelu(x)` (não exato)

### [ ] MULTI-HEAD ATTENTION
- [ ] Dimensões de saída: `output.shape == (batch, seq_len_q, dim)`
- [ ] Attention pesos: `attn_weights.sum(axis=-1) ≈ 1`
- [ ] Valores numéricos: erro relativo < 1%
- [ ] Máscara causal funciona corretamente

---

## 5. Técnicas de Depuração

### Comparação por Camada

Se ATTENTION falha mas LINEAR passa:

```python
# Debugar LINEAR dentro de ATTENTION
attention_numpy_py = MultiHeadAttention(...)
query, key, value = ...

# Passo 1: Testar projeções
Q = query @ attention.w_q
# Comparar Q com saída de linear_rtl em Verilog

# Passo 2: Testar split e scores
Q_h = reshape_split_heads(Q)
K_h = reshape_split_heads(K)
scores = matmul(Q_h, K_h.T) / sqrt(dim_head)
# Comparar com saída de matmul_rtl

# ... continuar isolando problema
```

### Diferenças Acumulativas

Erros pequenos podem acumular:

```
LINEAR  → erro: 1e-5
SOFTMAX → erro: 1e-5
GELU    → erro: 1e-5
...
TOTAL   → erro pode ser 1e-4 (tolerável)
```

Se erro final > 1%, verificar:
1. Acurácia de ponto flutuante (mantissa bits)
2. Ordem de operações (associatividade não é garantida)
3. Arredondamento de constantes (sqrt(2/π), tanh coeffs)

---

## 6. Padrão de Teste para Bloco Customizado

```python
# reference/testbenches/test_meu_bloco.py

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'blocks'))

from meu_bloco import MeuBloco

def test_meu_bloco():
    """Teste completo do bloco"""
    
    # Configuração
    config = {
        'param1': 768,
        'param2': 12,
    }
    
    bloco = MeuBloco(**config)
    
    # Gerar entrada de teste
    x = np.random.randn(2, 196, 768).astype(np.float32)
    
    # Executar
    y = bloco.forward(x)
    
    # Validações
    assert y.shape == (2, 196, 768), f"Shape mismatch: {y.shape}"
    assert y.dtype == np.float32, f"Dtype mismatch: {y.dtype}"
    assert not np.isnan(y).any(), "Output contém NaN"
    assert not np.isinf(y).any(), "Output contém Inf"
    
    # Propriedades específicas do bloco
    assert np.max(np.abs(y)) < 100, "Valores muito grandes"
    assert np.abs(y.mean()) < 10, "Média muito distante de 0"
    
    print("✓ Teste passou!")

if __name__ == '__main__':
    test_meu_bloco()
```

---

## Resumo: Fluxo Completo

```bash
# 1. Gerar vetores de teste
python3 reference/testbenches/generate_test_vectors.py --block all

# 2. Simular cada bloco em Verilog
cd rtl/testbenches
verilator --trace tb_linear.v -o sim_linear
./sim_linear

# 3. Converter VCD para formato comparável
python3 convert_vcd_to_csv.py sim_linear.vcd

# 4. Validar equivalência
python3 reference/testbenches/verify_equivalence.py \
    reference/vectors/linear_test_0.npz \
    results/linear_output.csv

# 5. Repetir para cada bloco
# ... LAYER_NORM, SOFTMAX, GELU, ATTENTION
```

---

## Dicas Finais

1. **Teste blocos individualmente antes de integrar**
2. **Use tipos `float32` / `float64` para máxima precisão**
3. **Implemente máscaras booleanas como **aditivas** em Verilog**
4. **Salve outputs intermediários para debug**
5. **Documente configurações (batch size, seq_len) de cada teste**

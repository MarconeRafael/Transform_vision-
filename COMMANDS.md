# Comandos Úteis - Transform Vision

## �� Documentação

```bash
# Visão geral (leia primeiro)
cat README.md
cat PROJECT_STRUCTURE.md

# Guia rápido para implementar Verilog
cat QUICK_START.md

# Status e conclusão
cat PROJETO_FINALIZADO.md

# Especificações técnicas detalhadas
cat reference/docs/BLOCK_SPECIFICATIONS.md
cat reference/docs/EQUIVALENCE_GUIDE.md
```

## 🧪 Testes

```bash
# Testar implementações Python (alto nível e baixo nível)
python3 test_all.py

# Testar bloco NumPy específico
cd reference/blocks
python3 linear.py
python3 layer_norm.py
python3 softmax.py
python3 gelu.py
python3 multi_head_attention.py
```

## 🔬 Gerar Vetores de Teste

```bash
# Gerar TODOS os vetores (recomendado)
python3 reference/testbenches/generate_test_vectors.py --block all --output reference/vectors

# Gerar apenas um bloco
python3 reference/testbenches/generate_test_vectors.py --block linear --output reference/vectors
python3 reference/testbenches/generate_test_vectors.py --block softmax --output reference/vectors
python3 reference/testbenches/generate_test_vectors.py --block layer_norm --output reference/vectors
python3 reference/testbenches/generate_test_vectors.py --block gelu --output reference/vectors
python3 reference/testbenches/generate_test_vectors.py --block attention --output reference/vectors
```

## 📊 Inspecionar Dados de Teste

```bash
# Inspecionar um vetor de teste
python3 -c "
import numpy as np
import json

# Carregar dados
data = np.load('reference/vectors/linear_test_0.npz')
with open('reference/vectors/linear_test_0_metadata.json') as f:
    config = json.load(f)

print('=== LINEAR TEST 0 ===')
print('Configuração:', config)
print('Entrada x:', data['x'].shape)
print('Pesos W:', data['weight'].shape)
print('Bias b:', data['bias'].shape)
print('Saída y esperada:', data['y'].shape)
print('Valor mín de y:', data['y'].min())
print('Valor máx de y:', data['y'].max())
"

# Listar todos os vetores gerados
ls -lh reference/vectors/*.npz | awk '{print $9, $5}'
```

## 🔧 Verilog - Simulação com Verilator

```bash
# Simular um testbench
cd rtl/testbenches
verilator --trace tb_linear.v -o sim_linear
./sim_linear

# Com debug
verilator --trace --debug tb_linear.v -o sim_linear_debug
./sim_linear_debug

# Verificar com gtkwave
gtkwave tb_linear.vcd
```

## ✅ Validar Equivalência Verilog vs NumPy

```bash
# Comparar saída Verilog com NumPy esperada
python3 reference/testbenches/verify_equivalence.py \
    reference/vectors/linear_test_0.npz \
    output_linear_test_0.csv

# Com tolerância customizada
python3 reference/testbenches/verify_equivalence.py \
    reference/vectors/linear_test_0.npz \
    output_linear_test_0.csv \
    --tolerance 1e-3
```

## 📁 Estrutura de Diretórios

```bash
# Ver estrutura do projeto
ls -R

# Contar linhas de código
find . -name "*.py" | xargs wc -l | tail -1

# Tamanho total
du -sh .

# Tamanho de reference
du -sh reference/
```

## 🔬 Desenvolvimento Iterativo

```bash
# 1. Ler especificação de um bloco
grep -A 30 "## 1. LINEAR" reference/docs/BLOCK_SPECIFICATIONS.md

# 2. Inspecionar vetor de teste
python3 reference/testbenches/generate_test_vectors.py --block linear

# 3. Editar Verilog
code rtl/blocks/linear.v

# 4. Simular
cd rtl/testbenches
verilator --trace tb_linear.v -o sim && ./sim

# 5. Comparar
python3 ../../reference/testbenches/verify_equivalence.py ...

# 6. Debug se necessário
# (salvar intermediários em testbench)
```

## 🧬 Python - Teste de Blocos NumPy

```python
# Testar Linear
from reference.blocks import Linear
lin = Linear(in_features=768, out_features=3072, seed=42)
x = np.random.randn(2, 196, 768).astype(np.float32)
y = lin.forward(x)
assert y.shape == (2, 196, 3072)
print("✓ Linear passou")

# Testar LayerNorm
from reference.blocks import LayerNorm
ln = LayerNorm(dim=768)
x = np.random.randn(2, 196, 768).astype(np.float32)
y = ln.forward(x)
assert np.abs(y.mean(axis=-1)).max() < 0.01
assert np.abs(y.var(axis=-1) - 1.0).max() < 0.1
print("✓ LayerNorm passou")

# Testar Softmax
from reference.blocks import Softmax
sm = Softmax()
x = np.random.randn(2, 8, 196, 196).astype(np.float32)
y = sm.forward(x)
assert np.allclose(y.sum(axis=-1), 1.0)
print("✓ Softmax passou")

# Testar GELU
from reference.blocks import GELU
gelu = GELU()
x = np.random.randn(2, 196, 3072).astype(np.float32)
y = gelu.forward(x)
assert y.shape == x.shape
print("✓ GELU passou")

# Testar Attention
from reference.blocks import MultiHeadAttention
attn = MultiHeadAttention(dim=768, num_heads=12, seed=42)
q = np.random.randn(2, 196, 768).astype(np.float32)
k = np.random.randn(2, 196, 768).astype(np.float32)
v = np.random.randn(2, 196, 768).astype(np.float32)
out, weights = attn.forward(q, k, v)
assert out.shape == q.shape
assert np.allclose(weights.sum(axis=-1), 1.0)
print("✓ Attention passou")
```

## 📈 Monitoramento de Progresso

```bash
# Quantas linhas de Verilog você tem
find rtl/blocks -name "*.v" | xargs wc -l | tail -1

# Qual percentual de completude
echo "Linear: IMPLEMENTADO (20%)"
echo "LayerNorm: PENDENTE"
echo "Softmax: PENDENTE"
echo "GELU: PENDENTE"
echo "Attention: PENDENTE"
echo "Progresso: 1/5 = 20%"

# Verificar o status
ls -1 rtl/blocks/*.v 2>/dev/null | wc -l
```

## 🐛 Debug - Erros Comuns

```bash
# Erro: "cannot import reference.blocks"
# Solução: Adicionar path correto
python3 -c "import sys; sys.path.insert(0, 'reference'); from blocks import Linear"

# Erro: numpy array shape mismatch
# Solução: Verificar documentação de shapes em BLOCK_SPECIFICATIONS.md

# Erro: Float precision mismatch em Verilog
# Solução: Verificar EQUIVALENCE_GUIDE.md seção de precision

# Erro: VCD não abre em gtkwave
# Solução: Gerar com $dumpfile e $dumpvars em testbench
```

## 📞 Referência Rápida

| Ação | Comando |
|------|---------|
| Testar Python | `python3 test_all.py` |
| Gerar vetores | `python3 reference/testbenches/generate_test_vectors.py --block all` |
| Ver spec Linear | `grep -A 30 "## 1. LINEAR" reference/docs/BLOCK_SPECIFICATIONS.md` |
| Ver spec Attention | `grep -A 40 "## 5. MULTI-HEAD ATTENTION" reference/docs/BLOCK_SPECIFICATIONS.md` |
| Simular Verilog | `cd rtl/testbenches && verilator --trace tb_*.v && ./sim` |
| Validar resultado | `python3 reference/testbenches/verify_equivalence.py` |

---

**Próximo passo**: Implementar Linear em Verilog! 🚀

Mais detalhes: Ver `QUICK_START.md`

# Transform Vision – Implementação do Vision Transformer para Hardware

## Objetivo do Projeto

Desenvolver um **Transformer para visão** com foco em implementação em **hardware (Verilog/FPGA)**.
O fluxo de trabalho adotado no repositório foi organizado em duas camadas principais:

1. **Alto nível (PyTorch)** – implementação funcional completa do encoder e decoder utilizando módulos de alto nível. Serve como baseline comportamental.
2. **Baixo nível (Python puro + NumPy)** – versões determinísticas e explícitas dos blocos, usadas como referência numérica (golden reference) para a implementação em Verilog.

O repositório foi ampliado para facilitar o desenvolvimento de RTL: há referências de bloco, scripts para gerar vetores de teste, documentação de especificações e diretórios reservados para o código Verilog.

---

## Visão geral da estrutura atual

```
.
├── LICENSE
├── README.md
├── pythons/
│   ├── alto_nivel/        # PyTorch: encoder/decoder (baseline funcional)
│   │   ├── encoder.py
│   │   └── decoder.py
│   └── baixo_nivel/       # NumPy: referência numérica (blocos detalhados)
│       ├── encoder.py
│       └── decoder.py
├── reference/             # Artefatos para conversão e verificação de RTL
│   ├── blocks/            # Implementações NumPy isoladas por bloco (linear, norm, softmax...)
│   ├── testbenches/       # Scripts para gerar vetores e verificar equivalência
│   ├── vectors/           # Vetores de teste gerados (.npz + metadata)
│   └── docs/              # Especificações de blocos e guia de equivalência
├── rtl/                   # Espaço reservado para arquivos Verilog (implementação do hardware)
│   ├── blocks/            # -> colocar `*.v` aqui (ex.: `linear.v`, `layer_norm.v`)
│   └── testbenches/       # Testbenches e harnesses Verilog/Verilator
├── test_all.py            # Runner unificado para validar implementações PyTorch/NumPy
└── PROJECT_STRUCTURE.md   # Documento com mapa do repositório e fluxo de trabalho
```

### O que tem nas novas pastas (resumo)

- **`reference/blocks/`**: implementações NumPy por bloco com interfaces simples (`forward`, `set_weights`/`get_weights`) para facilitar extração de parâmetros e geração de vetores de verificação.
- **`reference/testbenches/`**: scripts úteis:
  - `generate_test_vectors.py` — gera arquivos `.npz` com entradas, parâmetros e saídas esperadas para cada bloco.
  - `verify_equivalence.py` — comparador básico para validar saídas do RTL frente ao NumPy.
- **`reference/vectors/`**: coleções de vetores `.npz` e arquivos de metadata JSON usados como golden vectors para os testbenches Verilog.
- **`reference/docs/`**: especificações de datapath para cada bloco (linear, layernorm, softmax, gelu, attention) e um `EQUIVALENCE_GUIDE` com fluxo de validação recomendado.
- **`rtl/`**: diretório legado para o código Verilog — começe por criar `rtl/blocks/linear.v` e `rtl/testbenches/` e importe os vetores gerados para validação.

---

## Instruções gerais (rápidas)

- **Validar Python (rápido)**: execute `python3 test_all.py` na raiz — o script roda checagens básicas das implementações em `pythons/alto_nivel` e `pythons/baixo_nivel`.
- **Gerar vetores de teste**: rode `python3 reference/testbenches/generate_test_vectors.py` para (re)criar os `.npz` em `reference/vectors/`.
- **Fluxo de verificação RTL (resumido)**:
  1. Implementar módulo Verilog em `rtl/blocks/`.
  2. Carregar pesos/entradas do `.npz` (converter para `readmemh` ou injetar via DPI/arquivo) no testbench.
  3. Simular com Verilator (ou outro simulador) e exportar saídas.
  4. Usar `reference/testbenches/verify_equivalence.py` para comparar saídas do RTL com os arrays NumPy (pequera tolerância numérica aplicável).
- **Ordem recomendada para implementar blocos em Verilog**: `linear` → `layer_norm` → `softmax` → `gelu` → `multi_head_attention`. Implementar e validar cada bloco isoladamente primeiro.
- **Documentação e trace**: use `reference/docs/BLOCK_SPECIFICATIONS.md` e `reference/docs/EQUIVALENCE_GUIDE.md` como guias para datapath, precisão e pontos de inspeção para debug.

---

## Próximos passos sugeridos

- Escolher o primeiro bloco para implementação RTL (recomendado: `linear`) e criar um testbench que leia `reference/vectors/linear_*.npz` convertidos.
- Se quiser, posso gerar um esqueleto Verilog + testbench (com conversor `.npz`→`hex`), ou um `Makefile`/script para rodar Verilator e a verificação automaticamente.

---

## Objetivo Final

Criar uma implementação determinística, verificável e sintetizável do Vision Transformer, com blocos modulares e testes automatizados que usam `reference/` como golden reference.

Se quiser que eu já gere um template Verilog e o testbench para `linear.v`, diga qual formato de memória prefere (`readmemh`, files + DPI`, ou injeção via C++/DPI`) e eu preparo o esqueleto.


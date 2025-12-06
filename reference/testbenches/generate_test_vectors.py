# reference/testbenches/generate_test_vectors.py
"""
Gerador de vetores de teste para blocos do Vision Transformer

Uso:
  python3 generate_test_vectors.py --block linear --output vectors/
  python3 generate_test_vectors.py --block softmax --output vectors/
  python3 generate_test_vectors.py --block layer_norm --output vectors/
"""

import sys
import os
import argparse
import numpy as np
import json

# Adicionar path para importar blocos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'blocks'))

from linear import Linear
from layer_norm import LayerNorm
from softmax import Softmax
from gelu import GELU
from multi_head_attention import MultiHeadAttention


def save_test_vector(output_dir, name, inputs, outputs, metadata=None):
    """Salva um vetor de teste em arquivo"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar numpy arrays em formato npz
    output_file = os.path.join(output_dir, f"{name}.npz")
    np.savez(output_file, **inputs, **outputs)
    
    # Salvar metadados em JSON
    meta = {
        'name': name,
        'input_shapes': {k: v.shape for k, v in inputs.items()},
        'output_shapes': {k: v.shape for k, v in outputs.items()},
        'dtypes': {k: str(v.dtype) for k, v in {**inputs, **outputs}.items()},
    }
    if metadata:
        meta.update(metadata)
    
    meta_file = os.path.join(output_dir, f"{name}_metadata.json")
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved: {output_file}")
    print(f"       {meta_file}")


def generate_linear_vectors(output_dir):
    """Gera vetores de teste para Linear"""
    print("\n[LINEAR] Gerando vetores de teste...")
    
    configs = [
        {'in_features': 768, 'out_features': 3072, 'batch': 2, 'seq_len': 196},  # MLP
        {'in_features': 768, 'out_features': 768, 'batch': 2, 'seq_len': 196},   # Self-proj
        {'in_features': 64, 'out_features': 64, 'batch': 8, 'seq_len': 10},      # Small
    ]
    
    for i, config in enumerate(configs):
        lin = Linear(**{k: v for k, v in config.items() if k in ['in_features', 'out_features']},
                     seed=42)
        
        x = np.random.randn(config['batch'], config['seq_len'], 
                           config['in_features']).astype(np.float32)
        y = lin.forward(x)
        
        inputs = {'x': x, 'weight': lin.weight, 'bias': lin.bias}
        outputs = {'y': y}
        metadata = config
        
        save_test_vector(output_dir, f"linear_test_{i}", inputs, outputs, metadata)
    
    print(f"✓ {len(configs)} vetores de teste para Linear criados")


def generate_layer_norm_vectors(output_dir):
    """Gera vetores de teste para LayerNorm"""
    print("\n[LAYERNORM] Gerando vetores de teste...")
    
    configs = [
        {'dim': 768, 'batch': 2, 'seq_len': 196},  # ViT padrão
        {'dim': 64, 'batch': 8, 'seq_len': 10},    # Pequeno
    ]
    
    for i, config in enumerate(configs):
        ln = LayerNorm(dim=config['dim'], eps=1e-6)
        
        x = np.random.randn(config['batch'], config['seq_len'], 
                           config['dim']).astype(np.float32)
        y = ln.forward(x)
        
        inputs = {'x': x, 'weight': ln.weight, 'bias': ln.bias}
        outputs = {'y': y}
        metadata = {**config, 'eps': 1e-6}
        
        save_test_vector(output_dir, f"layer_norm_test_{i}", inputs, outputs, metadata)
    
    print(f"✓ {len(configs)} vetores de teste para LayerNorm criados")


def generate_softmax_vectors(output_dir):
    """Gera vetores de teste para Softmax"""
    print("\n[SOFTMAX] Gerando vetores de teste...")
    
    configs = [
        {'batch': 2, 'num_heads': 12, 'seq_len': 196},  # ViT padrão
        {'batch': 2, 'num_heads': 8, 'seq_len': 10},    # Pequeno
    ]
    
    for i, config in enumerate(configs):
        sm = Softmax(dim=-1)
        
        x = np.random.randn(config['batch'], config['num_heads'], 
                           config['seq_len'], config['seq_len']).astype(np.float32)
        y = sm.forward(x)
        
        inputs = {'x': x}
        outputs = {'y': y}
        metadata = config
        
        save_test_vector(output_dir, f"softmax_test_{i}", inputs, outputs, metadata)
    
    print(f"✓ {len(configs)} vetores de teste para Softmax criados")


def generate_gelu_vectors(output_dir):
    """Gera vetores de teste para GELU"""
    print("\n[GELU] Gerando vetores de teste...")
    
    configs = [
        {'batch': 2, 'seq_len': 196, 'dim': 3072},  # MLP
        {'batch': 8, 'seq_len': 10, 'dim': 64},     # Pequeno
    ]
    
    for i, config in enumerate(configs):
        gelu = GELU()
        
        x = np.random.randn(config['batch'], config['seq_len'], 
                           config['dim']).astype(np.float32)
        y = gelu.forward(x)
        
        inputs = {'x': x}
        outputs = {'y': y}
        metadata = config
        
        save_test_vector(output_dir, f"gelu_test_{i}", inputs, outputs, metadata)
    
    print(f"✓ {len(configs)} vetores de teste para GELU criados")


def generate_attention_vectors(output_dir):
    """Gera vetores de teste para Multi-Head Attention"""
    print("\n[ATTENTION] Gerando vetores de teste...")
    
    configs = [
        {'dim': 768, 'num_heads': 12, 'batch': 2, 'seq_len': 196},  # ViT padrão
        {'dim': 64, 'num_heads': 8, 'batch': 2, 'seq_len': 10},     # Pequeno
    ]
    
    for i, config in enumerate(configs):
        attn = MultiHeadAttention(dim=config['dim'], num_heads=config['num_heads'], seed=42)
        
        query = np.random.randn(config['batch'], config['seq_len'], 
                               config['dim']).astype(np.float32)
        key = np.random.randn(config['batch'], config['seq_len'], 
                             config['dim']).astype(np.float32)
        value = np.random.randn(config['batch'], config['seq_len'], 
                               config['dim']).astype(np.float32)
        
        output, attn_weights = attn.forward(query, key, value)
        
        w_q, w_k, w_v, w_o = attn.get_weights()
        
        inputs = {
            'query': query,
            'key': key,
            'value': value,
            'w_q': w_q,
            'w_k': w_k,
            'w_v': w_v,
            'w_o': w_o,
        }
        outputs = {
            'output': output,
            'attn_weights': attn_weights,
        }
        metadata = config
        
        save_test_vector(output_dir, f"attention_test_{i}", inputs, outputs, metadata)
    
    print(f"✓ {len(configs)} vetores de teste para Attention criados")


def main():
    parser = argparse.ArgumentParser(description='Gera vetores de teste para blocos ViT')
    parser.add_argument('--block', choices=['linear', 'layer_norm', 'softmax', 'gelu', 'attention', 'all'],
                       default='all', help='Qual bloco testar')
    parser.add_argument('--output', default='vectors', help='Diretório de saída')
    
    args = parser.parse_args()
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Gerador de Vetores de Teste - Vision Transformer")
    print("=" * 70)
    
    if args.block in ['linear', 'all']:
        generate_linear_vectors(output_dir)
    
    if args.block in ['layer_norm', 'all']:
        generate_layer_norm_vectors(output_dir)
    
    if args.block in ['softmax', 'all']:
        generate_softmax_vectors(output_dir)
    
    if args.block in ['gelu', 'all']:
        generate_gelu_vectors(output_dir)
    
    if args.block in ['attention', 'all']:
        generate_attention_vectors(output_dir)
    
    print("\n" + "=" * 70)
    print(f"✓ Todos os vetores salvos em: {os.path.abspath(output_dir)}")
    print("=" * 70)


if __name__ == '__main__':
    main()

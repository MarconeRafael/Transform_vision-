#!/usr/bin/env python3
"""
Script de teste abrangente para validar:
- Encoder de alto nível (PyTorch)
- Decoder de alto nível (PyTorch)
- Encoder de baixo nível (NumPy)
- Decoder de baixo nível (NumPy)
"""

import sys
import os
import traceback
import numpy as np
import torch

print("=" * 80)
print("TESTE COMPLETO - Transform Vision")
print("=" * 80)

# ============================================================================
# 1. TESTE ENCODER DE ALTO NÍVEL (PyTorch)
# ============================================================================
print("\n[1/4] Testando ENCODER DE ALTO NÍVEL (PyTorch)...")
try:
    # Adicionar path de alto_nivel
    sys.path.insert(0, os.path.join(os.getcwd(), 'pythons/alto_nivel'))
    
    from encoder import VisionTransformerEncoder as EncoderPyTorch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_pt = EncoderPyTorch(
        img_size=224,
        canais_entrada=3,
        tamanho_patch=16,
        dimensao_embedding=768,
        num_blocos=2,
        num_cabecas=12,
        dimensao_mlp=3072,
        dropout=0.1,
        attn_dropout=0.1,
        uso_cls=True,
        num_classes=1000
    ).to(device)
    
    x_pt = torch.randn(2, 3, 224, 224, device=device)
    logits_pt = encoder_pt(x_pt)
    features_pt = encoder_pt.forward_features(x_pt)
    
    print(f"  ✓ Encoder PyTorch funcionando!")
    print(f"    - Entrada shape: {x_pt.shape}")
    print(f"    - Saída logits shape: {logits_pt.shape}")
    print(f"    - Features shape: {features_pt.shape}")
    
except Exception as e:
    print(f"  ✗ ERRO no encoder PyTorch:")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 2. TESTE DECODER DE ALTO NÍVEL (PyTorch)
# ============================================================================
print("\n[2/4] Testando DECODER DE ALTO NÍVEL (PyTorch)...")
try:
    from decoder import VisionTransformerDecoder as DecoderPyTorch
    
    enc_tokens = encoder_pt.forward_features(x_pt)
    num_patches = (224 // 16) ** 2
    
    # Remove CLS token se presente
    if enc_tokens.shape[1] > num_patches:
        enc_kv = enc_tokens[:, 1:, :]
    else:
        enc_kv = enc_tokens
    
    decoder_pt = DecoderPyTorch(
        dimensao_embedding=768,
        num_blocos=2,
        num_cabecas=12,
        dimensao_mlp=3072,
        dropout=0.1,
        attn_dropout=0.1,
        uso_queries=True,
        num_queries=enc_kv.shape[1],
        reconstruct=True,
        patch_size=16,
        canais_saida=3,
        num_classes=None
    ).to(device)
    
    out_pt = decoder_pt(encoder_tokens=enc_kv, encoder_padding_mask=None, img_size=(224, 224), causal=False)
    
    print(f"  ✓ Decoder PyTorch funcionando!")
    print(f"    - Encoder tokens shape: {enc_kv.shape}")
    print(f"    - Reconstruction shape: {out_pt['reconstruction'].shape}")
    print(f"    - Output tokens shape: {out_pt['tokens'].shape}")
    
except Exception as e:
    print(f"  ✗ ERRO no decoder PyTorch:")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 3. TESTE ENCODER DE BAIXO NÍVEL (NumPy)
# ============================================================================
print("\n[3/4] Testando ENCODER DE BAIXO NÍVEL (NumPy)...")
try:
    # Limpar módulos antigos e paths
    for mod in list(sys.modules.keys()):
        if mod in ['encoder', 'decoder']:
            del sys.modules[mod]
    
    sys.path = [p for p in sys.path if 'alto_nivel' not in p]
    sys.path.insert(0, os.path.join(os.getcwd(), 'pythons/baixo_nivel'))
    
    from encoder import VisionTransformerEncoder as EncoderNumPy
    
    encoder_np = EncoderNumPy(
        img_size=224,
        canais_entrada=3,
        tamanho_patch=16,
        dimensao_embedding=768,
        num_blocos=2,
        num_cabecas=12,
        dimensao_mlp=3072,
        dropout=0.1,
        attn_dropout=0.1,
        uso_cls=True,
        num_classes=None,
        seed=42
    )
    
    x_np = np.random.randn(2, 3, 224, 224).astype(np.float32)
    features_np = encoder_np.forward_features(x_np)
    output_np = encoder_np.forward(x_np)
    
    print(f"  ✓ Encoder NumPy funcionando!")
    print(f"    - Entrada shape: {x_np.shape}")
    print(f"    - Features shape: {features_np.shape}")
    print(f"    - Output shape: {output_np.shape}")
    
except Exception as e:
    print(f"  ✗ ERRO no encoder NumPy:")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 4. TESTE DECODER DE BAIXO NÍVEL (NumPy)
# ============================================================================
print("\n[4/4] Testando DECODER DE BAIXO NÍVEL (NumPy)...")
try:
    from decoder import VisionTransformerDecoder as DecoderNumPy
    
    enc_tokens_np = encoder_np.forward_features(x_np)
    num_patches_np = (224 // 16) ** 2
    
    # Remove CLS token se presente
    if enc_tokens_np.shape[1] > num_patches_np:
        enc_kv_np = enc_tokens_np[:, 1:, :]
    else:
        enc_kv_np = enc_tokens_np
    
    decoder_np = DecoderNumPy(
        dimensao_embedding=768,
        num_blocos=2,
        num_cabecas=12,
        dimensao_mlp=3072,
        dropout=0.1,
        attn_dropout=0.1,
        uso_queries=True,
        num_queries=enc_kv_np.shape[1],
        reconstruct=True,
        patch_size=16,
        canais_saida=3,
        num_classes=None,
        seed=42
    )
    
    out_np = decoder_np.forward(encoder_tokens=enc_kv_np, encoder_padding_mask=None, img_size=(224, 224), causal=False)
    
    print(f"  ✓ Decoder NumPy funcionando!")
    print(f"    - Encoder tokens shape: {enc_kv_np.shape}")
    print(f"    - Reconstruction shape: {out_np['reconstruction'].shape}")
    print(f"    - Output tokens shape: {out_np['tokens'].shape}")
    
except Exception as e:
    print(f"  ✗ ERRO no decoder NumPy:")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("✓ TODOS OS TESTES PASSARAM COM SUCESSO!")
print("=" * 80)
print("\nResumo:")
print("  [✓] Encoder PyTorch   - Funcionando")
print("  [✓] Decoder PyTorch   - Funcionando")
print("  [✓] Encoder NumPy     - Funcionando")
print("  [✓] Decoder NumPy     - Funcionando")
print("\nOs arquivos Python estão prontos para uso!")
print("=" * 80)

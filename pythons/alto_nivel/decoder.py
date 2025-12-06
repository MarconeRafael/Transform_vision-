# vit_decoder.py (corrigido, autônomo)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import VisionTransformerEncoder


# ----------------- MLP (usado no decoder) -----------------
class MLP(nn.Module):
    def __init__(self, dimensao, dimensao_oculta, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dimensao, dimensao_oculta)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dimensao_oculta, dimensao)
        self.dropout = nn.Dropout(dropout)

        # inicialização simples
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ----------------- BlocoDecoder -----------------
class BlocoDecoder(nn.Module):
    """
    pre-norm:
    1) LayerNorm -> Masked Self-Attention -> Residual
    2) LayerNorm -> Cross-Attention (Q=dec, K/V=enc) -> Residual
    3) LayerNorm -> MLP -> Residual
    """
    def __init__(self, dimensao, num_cabecas=8, dimensao_mlp=None, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        if dimensao_mlp is None:
            dimensao_mlp = 4 * dimensao

        self.norm1 = nn.LayerNorm(dimensao, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(embed_dim=dimensao, num_heads=num_cabecas,
                                               dropout=attn_dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dimensao, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dimensao, num_heads=num_cabecas,
                                                dropout=attn_dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(dimensao, eps=1e-6)
        self.mlp = MLP(dimensao, dimensao_mlp, dropout=dropout)

    def forward(self, x_dec, enc_kv=None, self_attn_mask=None, enc_key_padding_mask=None):
        # masked self-attention (queries/keys/values = x_dec)
        x_norm = self.norm1(x_dec)
        # attn_mask can be float additive (L_dec,L_dec) or bool mask
        self_attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=self_attn_mask)
        x_dec = x_dec + self.dropout1(self_attn_out)

        # cross-attention (queries = x_dec, keys/values = enc_kv)
        if enc_kv is not None:
            x_norm = self.norm2(x_dec)
            cross_attn_out, _ = self.cross_attn(x_norm, enc_kv, enc_kv,
                                                key_padding_mask=enc_key_padding_mask)
            x_dec = x_dec + self.dropout2(cross_attn_out)

        # mlp
        x_norm = self.norm3(x_dec)
        mlp_out = self.mlp(x_norm)
        x_dec = x_dec + mlp_out

        return x_dec


# ----------------- DecoderStack -----------------
class DecoderStack(nn.Module):
    def __init__(self, num_blocos, dimensao, num_cabecas, dimensao_mlp=None, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        blocos = []
        for _ in range(num_blocos):
            blocos.append(BlocoDecoder(dimensao=dimensao,
                                      num_cabecas=num_cabecas,
                                      dimensao_mlp=dimensao_mlp,
                                      dropout=dropout,
                                      attn_dropout=attn_dropout))
        self.blocos = nn.ModuleList(blocos)

    def forward(self, x_dec, enc_kv=None, self_attn_mask=None, enc_key_padding_mask=None):
        for bloco in self.blocos:
            x_dec = bloco(x_dec, enc_kv=enc_kv, self_attn_mask=self_attn_mask,
                          enc_key_padding_mask=enc_key_padding_mask)
        return x_dec


# ----------------- VisionTransformerDecoder -----------------
class VisionTransformerDecoder(nn.Module):
    """
    Decoder compatível com VisionTransformerEncoder (PyTorch).
    - Se reconstruct=True: projeta tokens de volta para patches e unpatchify para imagem.
    - Pode usar learned queries (num_queries) ou embeddings fornecidas (decoder_inputs).
    """
    def __init__(self,
                 dimensao_embedding=768,
                 num_blocos=6,
                 num_cabecas=12,
                 dimensao_mlp=None,
                 dropout=0.0,
                 attn_dropout=0.0,
                 uso_queries=True,
                 num_queries=None,
                 reconstruct=False,
                 patch_size=16,
                 canais_saida=3,
                 num_classes=None,
                 seed=None):
        super().__init__()
        self.dimensao = dimensao_embedding
        self.uso_queries = uso_queries
        self.reconstruct = reconstruct
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        if uso_queries:
            assert num_queries is not None, "num_queries required when uso_queries=True"
            self.query = nn.Parameter(torch.randn(1, num_queries, dimensao_embedding) * 0.02)
        else:
            self.query = None

        self.pos_embedding = None  # opcional

        self.decoder = DecoderStack(num_blocos=num_blocos,
                                    dimensao=dimensao_embedding,
                                    num_cabecas=num_cabecas,
                                    dimensao_mlp=dimensao_mlp,
                                    dropout=dropout,
                                    attn_dropout=attn_dropout)

        self.norm = nn.LayerNorm(dimensao_embedding, eps=1e-6)

        # cabeças finais
        self.num_classes = num_classes
        if reconstruct:
            assert num_queries is not None, "num_queries must match number of patches when reconstruct=True"
            self.patch_size = patch_size
            self.canais_saida = canais_saida
            self.head_recon = nn.Linear(dimensao_embedding, patch_size * patch_size * canais_saida)
        else:
            self.head_recon = None

        if num_classes is not None:
            self.head_class = nn.Linear(dimensao_embedding, num_classes)
        else:
            self.head_class = None

    @staticmethod
    def _build_causal_mask(num_tokens, device):
        """
        Retorna máscara aditiva (float) shape (num_tokens, num_tokens)
        com -inf onde deve ser bloqueado e 0.0 onde permitido.
        Compatível com nn.MultiheadAttention(attn_mask=...).
        """
        if num_tokens <= 0:
            return None
        mask = torch.triu(torch.full((num_tokens, num_tokens), float("-inf"), device=device), diagonal=1)
        return mask.to(dtype=torch.float32)

    def unpatchify(self, tokens, img_h, img_w):
        # tokens: (B, N, D)
        B, N, D = tokens.shape
        P = self.patch_size
        Hp = img_h // P
        Wp = img_w // P
        assert N == Hp * Wp, "tokens N mismatch unpatchify grid"
        patches = self.head_recon(tokens)  # (B, N, P*P*C)
        patches = patches.view(B, Hp, Wp, P, P, self.canais_saida)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, Hp, P, Wp, P)
        patches = patches.view(B, self.canais_saida, Hp * P, Wp * P)
        return patches

    def forward(self,
                decoder_inputs=None,
                encoder_tokens=None,
                encoder_padding_mask=None,
                img_size=None,
                causal=True):
        """
        decoder_inputs: (B, L_dec, D) se uso_queries=False; caso contrário learned queries são usadas.
        encoder_tokens: (B, L_enc, D)
        encoder_padding_mask: (B, L_enc) bool True onde há padding
        img_size: necessário se reconstruct=True (height, width)
        causal: aplicar máscara causal na self-attention do decoder
        """
        # determina device para criar máscaras
        device = None
        if encoder_tokens is not None:
            device = encoder_tokens.device
        elif decoder_inputs is not None:
            device = decoder_inputs.device
        else:
            # fallback: use primeiro parâmetro do module
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

        if self.uso_queries:
            if encoder_tokens is None:
                raise ValueError("encoder_tokens is required when uso_queries=True")
            B = encoder_tokens.shape[0]
            q = self.query.expand(B, -1, -1)  # (B, L_dec, D)
        else:
            assert decoder_inputs is not None
            q = decoder_inputs

        L_dec = q.shape[1]
        self_attn_mask = None
        if causal:
            self_attn_mask = self._build_causal_mask(L_dec, device=device)  # (L_dec, L_dec)

        enc_key_padding_mask = None
        if encoder_padding_mask is not None:
            enc_key_padding_mask = encoder_padding_mask.to(torch.bool).to(device)

        x = q
        x = self.decoder(x, enc_kv=encoder_tokens, self_attn_mask=self_attn_mask,
                         enc_key_padding_mask=enc_key_padding_mask)
        x = self.norm(x)

        out_recon = None
        out_class = None
        if self.reconstruct:
            assert img_size is not None, "img_size required for reconstruction"
            out_recon = self.unpatchify(x, img_size[0], img_size[1])  # (B, C, H, W)
        if self.head_class is not None:
            out_class = self.head_class(x)  # (B, L_dec, num_classes)

        return {"tokens": x, "reconstruction": out_recon, "logits": out_class}


# ---------------- Example of usage (teste reduzido) ----------------
if __name__ == "__main__":
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NOTE: requer que exista uma implementação PyTorch de VisionTransformerEncoder no escopo.
    # Se tiver definido a versão NumPy com o mesmo nome, importe a versão PyTorch explicitamente.
    # Exemplo:
    # from vit_encoder import VisionTransformerEncoder as VisionTransformerEncoderTorch
    # encoder = VisionTransformerEncoderTorch(...).to(dispositivo)

    # Aqui assumimos que VisionTransformerEncoder é a versão PyTorch (nn.Module).
    encoder = VisionTransformerEncoder(
        img_size=224, canais_entrada=3, tamanho_patch=16, dimensao_embedding=256,
        num_blocos=2, num_cabecas=8, dimensao_mlp=1024, dropout=0.1, attn_dropout=0.1,
        uso_cls=False, num_classes=None
    )
    # move encoder para device se suportar
    if hasattr(encoder, "to") and callable(getattr(encoder, "to")):
        encoder = encoder.to(dispositivo)

    # entrada no mesmo device
    x_img = torch.randn(2, 3, 224, 224, device=dispositivo)
    enc_tokens = encoder.forward_features(x_img)  # (B, N, D) ou (B, N+1, D)

    # remove cls token se presente
    num_patches = (224 // 16) ** 2
    if enc_tokens.shape[1] > num_patches:
        enc_kv = enc_tokens[:, 1:, :]
    else:
        enc_kv = enc_tokens

    num_patches = enc_kv.shape[1]

    decoder = VisionTransformerDecoder(
        dimensao_embedding=256,
        num_blocos=2,
        num_cabecas=8,
        dimensao_mlp=1024,
        dropout=0.1,
        attn_dropout=0.1,
        uso_queries=True,
        num_queries=num_patches,   # queries correspond to patches for reconstruction
        reconstruct=True,
        patch_size=16,
        canais_saida=3,
        num_classes=None
    ).to(dispositivo)

    out = decoder(encoder_tokens=enc_kv, encoder_padding_mask=None, img_size=(224, 224), causal=False)
    recon = out["reconstruction"]  # (B, C, H, W)
    tokens = out["tokens"]         # (B, N, D)
    print("recon.shape:", recon.shape)
    print("tokens.shape:", tokens.shape)

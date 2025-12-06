# vit_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Converte imagem (B, C, H, W) em sequência de patches embedados (B, N+1, D)
    Implementação via conv2d com kernel = stride = tamanho_patch.
    """
    def __init__(self, canais_entrada=3, tamanho_patch=16, dimensao_embedding=768):
        super().__init__()
        self.tamanho_patch = tamanho_patch
        self.conv = nn.Conv2d(
            in_channels=canais_entrada,
            out_channels=dimensao_embedding,
            kernel_size=tamanho_patch,
            stride=tamanho_patch
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv(x)                       # (B, D, Hp, Wp)
        b, d, hp, wp = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, N, D) onde N = Hp*Wp
        return x


class MLP(nn.Module):
    def __init__(self, dimensao, dimensao_oculta, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dimensao, dimensao_oculta)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dimensao_oculta, dimensao)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class BlocoEncoder(nn.Module):
    """
    Um bloco do encoder (pre-norm):
    - LayerNorm
    - Multi-Head Self-Attention
    - Residual
    - LayerNorm
    - MLP
    - Residual
    """
    def __init__(self, dimensao, num_cabecas=8, dimensao_mlp=None, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        if dimensao_mlp is None:
            dimensao_mlp = 4 * dimensao
        self.norm1 = nn.LayerNorm(dimensao, eps=1e-6)
        # usar batch_first=True para formatos (B, N, D)
        self.attn = nn.MultiheadAttention(embed_dim=dimensao, num_heads=num_cabecas,
                                          dropout=attn_dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dimensao, eps=1e-6)
        self.mlp = MLP(dimensao, dimensao_mlp, dropout=dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # atenção com pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attn_out)

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x


class EncoderStack(nn.Module):
    def __init__(self, num_blocos, dimensao, num_cabecas, dimensao_mlp=None, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        blocos = []
        for _ in range(num_blocos):
            blocos.append(BlocoEncoder(dimensao=dimensao,
                                       num_cabecas=num_cabecas,
                                       dimensao_mlp=dimensao_mlp,
                                       dropout=dropout,
                                       attn_dropout=attn_dropout))
        self.blocos = nn.ModuleList(blocos)

    def forward(self, x, **kwargs):
        for bloco in self.blocos:
            x = bloco(x, **kwargs)
        return x


class VisionTransformerEncoder(nn.Module):
    """
    Integração: PatchEmbedding -> [CLS token + PosEmb] -> EncoderStack -> saída de features
    """
    def __init__(self,
                 img_size=224,
                 canais_entrada=3,
                 tamanho_patch=16,
                 dimensao_embedding=768,
                 num_blocos=12,
                 num_cabecas=12,
                 dimensao_mlp=None,
                 dropout=0.0,
                 attn_dropout=0.0,
                 uso_cls=True,
                 num_classes=1000):
        super().__init__()

        assert img_size % tamanho_patch == 0, "img_size precisa ser múltiplo de tamanho_patch"
        self.uso_cls = uso_cls
        self.patch_embed = PatchEmbedding(canais_entrada, tamanho_patch, dimensao_embedding)

        num_patches = (img_size // tamanho_patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimensao_embedding)) if uso_cls else None
        self.pos_embedding = nn.Parameter(torch.zeros(1, (1 if uso_cls else 0) + num_patches, dimensao_embedding))
        self.dropout_pos = nn.Dropout(dropout)

        self.encoder = EncoderStack(num_blocos=num_blocos,
                                    dimensao=dimensao_embedding,
                                    num_cabecas=num_cabecas,
                                    dimensao_mlp=dimensao_mlp,
                                    dropout=dropout,
                                    attn_dropout=attn_dropout)

        self.norm = nn.LayerNorm(dimensao_embedding, eps=1e-6)

        # cabeça final
        self.head = nn.Linear(dimensao_embedding, num_classes) if num_classes is not None else nn.Identity()

        # inicializações simples
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # x: (B, C, H, W)
        patches = self.patch_embed(x)            # (B, N, D)
        b, n, d = patches.shape

        if self.uso_cls:
            cls_tokens = self.cls_token.expand(b, -1, -1)   # (B, 1, D)
            x = torch.cat((cls_tokens, patches), dim=1)     # (B, N+1, D)
        else:
            x = patches

        # ajustar pos emb se necessário (permite img_size diferente via resize simples)
        if x.shape[1] != self.pos_embedding.shape[1]:
            # forma simples: interp pos embedding espacialmente
            pe = self._resize_pos_embedding(self.pos_embedding, x.shape[1])
        else:
            pe = self.pos_embedding

        x = x + pe
        x = self.dropout_pos(x)

        x = self.encoder(x)
        x = self.norm(x)
        return x

    def _resize_pos_embedding(self, pos_emb, target_len):
        # assume pos_emb: (1, L, D); target_len = 1+num_patches (ou num_patches)
        # usa interp bicúbica para reshaping espacial do grid de patches
        l_old = pos_emb.shape[1]
        if self.uso_cls:
            cls_pe, grid_pe = pos_emb[:, :1], pos_emb[:, 1:]
            num_old = int(grid_pe.shape[1] ** 0.5)
            num_new = int(target_len - 1) ** 0.5
            if not num_new.is_integer():
                # fallback: cortar/estender por repetição simples
                if target_len == l_old:
                    return pos_emb
                return F.interpolate(grid_pe.reshape(1, num_old, num_old, -1).permute(0, 3, 1, 2),
                                     size=(int((target_len - 1) ** 0.5), int((target_len - 1) ** 0.5)),
                                     mode='bicubic', align_corners=False).permute(0, 2, 3, 1).reshape(1, target_len - 1, -1)
            grid_pe = grid_pe.reshape(1, num_old, num_old, -1).permute(0, 3, 1, 2)
            num_new = int((target_len - 1) ** 0.5)
            grid_pe = F.interpolate(grid_pe, size=(num_new, num_new), mode='bicubic', align_corners=False)
            grid_pe = grid_pe.permute(0, 2, 3, 1).reshape(1, num_new * num_new, -1)
            return torch.cat((cls_pe, grid_pe), dim=1)
        else:
            grid_pe = pos_emb
            num_old = int(grid_pe.shape[1] ** 0.5)
            grid_pe = grid_pe.reshape(1, num_old, num_old, -1).permute(0, 3, 1, 2)
            num_new = int(target_len ** 0.5)
            grid_pe = F.interpolate(grid_pe, size=(num_new, num_new), mode='bicubic', align_corners=False)
            grid_pe = grid_pe.permute(0, 2, 3, 1).reshape(1, num_new * num_new, -1)
            return grid_pe

    def forward(self, x):
        features = self.forward_features(x)   # (B, N+1, D) ou (B, N, D)
        if self.uso_cls:
            cls_f = features[:, 0]           # (B, D)
            out = self.head(cls_f)
        else:
            # média sobre tokens de patch
            pooled = features.mean(dim=1)
            out = self.head(pooled)
        return out


if __name__ == "__main__":
    # exemplo rápido de uso
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = VisionTransformerEncoder(
        img_size=224,
        canais_entrada=3,
        tamanho_patch=16,
        dimensao_embedding=768,
        num_blocos=12,
        num_cabecas=12,
        dimensao_mlp=3072,
        dropout=0.1,
        attn_dropout=0.1,
        uso_cls=True,
        num_classes=1000
    ).to(dispositivo)

    x = torch.randn(2, 3, 224, 224, device=dispositivo)
    logits = modelo(x)
    print("logits:", logits.shape)  # (2, 1000)

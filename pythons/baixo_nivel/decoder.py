# vision_transformer_numpy.py
# Implementação completa (NumPy) do ViT encoder + decoder compatível
# Nomes em português, comentários mínimos.

import numpy as np
import math

# ---------- utilitários ----------
class LayerNorm:
    def __init__(self, dimensao, eps=1e-6):
        self.dimensao = dimensao
        self.eps = eps
        self.peso = np.ones((dimensao,), dtype=np.float32)
        self.bias = np.zeros((dimensao,), dtype=np.float32)

    def forward(self, x):
        media = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - media) / np.sqrt(var + self.eps)
        return x_norm * self.peso + self.bias

def _build_causal_mask(num_q, num_k):
    return np.triu(np.ones((num_q, num_k), dtype=np.bool_), k=1)


# ---------- PatchEmbedding ----------
class PatchEmbedding:
    def __init__(self, canais_entrada=3, tamanho_patch=16, dimensao_embedding=768, seed=None):
        self.tamanho_patch = tamanho_patch
        self.canais = canais_entrada
        self.dimensao_embedding = dimensao_embedding
        rng = np.random.RandomState(seed)
        limite = 1.0 / np.sqrt(canais_entrada * tamanho_patch * tamanho_patch)
        self.pesos = rng.uniform(-limite, limite, size=(dimensao_embedding, canais_entrada * tamanho_patch * tamanho_patch)).astype(np.float32)
        self.bias = np.zeros((dimensao_embedding,), dtype=np.float32)

    def _extrair_patches(self, x):
        B, C, H, W = x.shape
        P = self.tamanho_patch
        assert H % P == 0 and W % P == 0
        hp = H // P
        wp = W // P
        N = hp * wp
        patches = np.zeros((B, N, C * P * P), dtype=np.float32)
        idx = 0
        for i in range(hp):
            for j in range(wp):
                h0, h1 = i * P, (i + 1) * P
                w0, w1 = j * P, (j + 1) * P
                bloco = x[:, :, h0:h1, w0:w1]
                patches[:, idx] = bloco.reshape(B, -1)
                idx += 1
        return patches

    def forward(self, x):
        patches = self._extrair_patches(x)             # (B, N, C*P*P)
        saida = patches @ self.pesos.T + self.bias     # (B, N, D)
        return saida.astype(np.float32)


# ---------- MLP ----------
class MLP:
    def __init__(self, dimensao, dimensao_oculta, dropout=0.0, seed=None):
        self.dimensao = dimensao
        self.dimensao_oculta = dimensao_oculta
        self.dropout = dropout
        rng = np.random.RandomState(seed)
        limite1 = 1.0 / np.sqrt(dimensao)
        limite2 = 1.0 / np.sqrt(dimensao_oculta)
        self.peso1 = rng.uniform(-limite1, limite1, size=(dimensao, dimensao_oculta)).astype(np.float32)
        self.bias1 = np.zeros((dimensao_oculta,), dtype=np.float32)
        self.peso2 = rng.uniform(-limite2, limite2, size=(dimensao_oculta, dimensao)).astype(np.float32)
        self.bias2 = np.zeros((dimensao,), dtype=np.float32)
        self.rng = rng

    def _gelu(self, x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))

    def _dropout(self, x):
        if self.dropout <= 0.0:
            return x
        mask = (self.rng.rand(*x.shape) > self.dropout).astype(np.float32)
        return x * mask / (1.0 - self.dropout)

    def forward(self, x):
        h = x @ self.peso1 + self.bias1
        h = self._gelu(h)
        h = self._dropout(h)
        out = h @ self.peso2 + self.bias2
        out = self._dropout(out)
        return out.astype(np.float32)


# ---------- Multi-Head Self-Attention ----------
class MultiHeadSelfAttention:
    def __init__(self, dimensao, num_cabecas=8, attn_dropout=0.0, seed=None):
        assert dimensao % num_cabecas == 0
        self.dimensao = dimensao
        self.num_cabecas = num_cabecas
        self.dim_cabeca = dimensao // num_cabecas
        self.attn_dropout = attn_dropout
        self.rng = np.random.RandomState(seed)
        limite = 1.0 / np.sqrt(dimensao)
        self.w_q = self.rng.uniform(-limite, limite, size=(dimensao, dimensao)).astype(np.float32)
        self.w_k = self.rng.uniform(-limite, limite, size=(dimensao, dimensao)).astype(np.float32)
        self.w_v = self.rng.uniform(-limite, limite, size=(dimensao, dimensao)).astype(np.float32)
        self.w_o = self.rng.uniform(-limite, limite, size=(dimensao, dimensao)).astype(np.float32)

    def _split_heads(self, x):
        B, N, D = x.shape
        x = x.reshape(B, N, self.num_cabecas, self.dim_cabeca)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        x = x.transpose(0, 2, 1, 3)
        B, N, _, _ = x.shape
        return x.reshape(B, N, self.dimensao)

    def _dropout(self, x):
        if self.attn_dropout <= 0.0:
            return x
        mask = (self.rng.rand(*x.shape) > self.attn_dropout).astype(np.float32)
        return x * mask / (1.0 - self.attn_dropout)

    def forward(self, q, k=None, v=None, attn_mask=None, key_padding_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q
        Q = q @ self.w_q
        K = k @ self.w_k
        V = v @ self.w_v
        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)
        scores = np.matmul(Qh, Kh.transpose(0,1,3,2)) / np.sqrt(self.dim_cabeca)
        if attn_mask is not None:
            if attn_mask.dtype == np.bool_:
                scores = np.where(attn_mask[None, None, :, :], -1e9, scores)
            else:
                scores = scores + attn_mask[None, None, :, :]
        if key_padding_mask is not None:
            kp = key_padding_mask.astype(bool)
            scores = np.where(kp[:, None, None, :], -1e9, scores)
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)
        attn = self._dropout(attn)
        contexto = np.matmul(attn, Vh)
        contexto = self._merge_heads(contexto)
        out = contexto @ self.w_o
        return out, attn


# ---------- BlocoEncoder ----------
class BlocoEncoder:
    def __init__(self, dimensao, num_cabecas=8, dimensao_mlp=None, dropout=0.0, attn_dropout=0.0, seed=None):
        if dimensao_mlp is None:
            dimensao_mlp = 4 * dimensao
        self.norm1 = LayerNorm(dimensao, eps=1e-6)
        self.attn = MultiHeadSelfAttention(dimensao, num_cabecas, attn_dropout=attn_dropout, seed=seed)
        self.dropout1 = dropout
        self.norm2 = LayerNorm(dimensao, eps=1e-6)
        self.mlp = MLP(dimensao, dimensao_mlp, dropout=dropout, seed=seed)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x_norm = self.norm1.forward(x)
        attn_out, _ = self.attn.forward(x_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        if self.dropout1 > 0.0:
            attn_out = attn_out * (np.random.rand(*attn_out.shape) > self.dropout1) / (1.0 - self.dropout1)
        x = x + attn_out
        x_norm = self.norm2.forward(x)
        mlp_out = self.mlp.forward(x_norm)
        x = x + mlp_out
        return x


# ---------- EncoderStack ----------
class EncoderStack:
    def __init__(self, num_blocos, dimensao, num_cabecas, dimensao_mlp=None, dropout=0.0, attn_dropout=0.0, seed=None):
        self.blocos = []
        for _ in range(num_blocos):
            self.blocos.append(BlocoEncoder(dimensao=dimensao, num_cabecas=num_cabecas,
                                            dimensao_mlp=dimensao_mlp, dropout=dropout,
                                            attn_dropout=attn_dropout, seed=seed))

    def forward(self, x, **kwargs):
        for b in self.blocos:
            x = b.forward(x, **kwargs)
        return x


# ---------- VisionTransformerEncoder ----------
class VisionTransformerEncoder:
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
                 num_classes=1000,
                 seed=None):
        assert img_size % tamanho_patch == 0
        rng = np.random.RandomState(seed)
        self.img_size = img_size
        self.tamanho_patch = tamanho_patch
        self.uso_cls = uso_cls
        self.patch_embed = PatchEmbedding(canais_entrada, tamanho_patch, dimensao_embedding, seed=seed)
        num_patches = (img_size // tamanho_patch) ** 2
        len_pos = (1 if uso_cls else 0) + num_patches
        self.pos_embedding = rng.normal(scale=0.02, size=(1, len_pos, dimensao_embedding)).astype(np.float32)
        if uso_cls:
            self.cls_token = rng.normal(scale=0.02, size=(1, 1, dimensao_embedding)).astype(np.float32)
        else:
            self.cls_token = None
        self.dropout = float(dropout)
        self.encoder = EncoderStack(num_blocos=num_blocos, dimensao=dimensao_embedding,
                                    num_cabecas=num_cabecas, dimensao_mlp=dimensao_mlp,
                                    dropout=dropout, attn_dropout=attn_dropout, seed=seed)
        self.norm = LayerNorm(dimensao_embedding, eps=1e-6)
        if num_classes is not None:
            limite = 1.0 / math.sqrt(dimensao_embedding)
            self.head_w = rng.uniform(-limite, limite, size=(dimensao_embedding, num_classes)).astype(np.float32)
            self.head_b = np.zeros((num_classes,), dtype=np.float32)
        else:
            self.head_w = None
            self.head_b = None

    def _dropout(self, x):
        if self.dropout <= 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.dropout).astype(np.float32)
        return x * mask / (1.0 - self.dropout)

    def _resize_pos_embedding(self, pos_emb, target_len):
        _, L_old, D = pos_emb.shape
        if target_len == L_old:
            return pos_emb.copy()
        if self.uso_cls:
            cls_pe = pos_emb[:, :1, :]
            grid_pe = pos_emb[:, 1:, :]
            Lg_old = grid_pe.shape[1]
            side_old = int(math.sqrt(Lg_old))
            side_new = int(math.sqrt(target_len - 1)) if (target_len - 1) > 0 else 0
            if side_old * side_old != Lg_old or side_new * side_new != (target_len - 1):
                if target_len <= L_old:
                    return pos_emb[:, :target_len, :].copy()
                pad = np.zeros((1, target_len - L_old, D), dtype=pos_emb.dtype)
                return np.concatenate((pos_emb, pad), axis=1)
            grid_pe = grid_pe.reshape(1, side_old, side_old, D)
            ys = (np.linspace(0, side_old - 1, side_new)).round().astype(int)
            xs = (np.linspace(0, side_old - 1, side_new)).round().astype(int)
            new_grid = grid_pe[0][np.ix_(ys, xs)].reshape(1, side_new * side_new, D)
            return np.concatenate((cls_pe, new_grid), axis=1)
        else:
            grid_pe = pos_emb
            Lg_old = grid_pe.shape[1]
            side_old = int(math.sqrt(Lg_old))
            side_new = int(math.sqrt(target_len))
            if side_old * side_old != Lg_old or side_new * side_new != target_len:
                if target_len <= L_old:
                    return pos_emb[:, :target_len, :].copy()
                pad = np.zeros((1, target_len - L_old, D), dtype=pos_emb.dtype)
                return np.concatenate((pos_emb, pad), axis=1)
            grid_pe = grid_pe.reshape(1, side_old, side_old, D)
            ys = (np.linspace(0, side_old - 1, side_new)).round().astype(int)
            xs = (np.linspace(0, side_old - 1, side_new)).round().astype(int)
            new_grid = grid_pe[0][np.ix_(ys, xs)].reshape(1, side_new * side_new, D)
            return new_grid

    def forward_features(self, x):
        patches = self.patch_embed.forward(x)   # (B, N, D)
        B, N, D = patches.shape
        if self.uso_cls:
            cls_expand = np.tile(self.cls_token, (B, 1, 1))
            x_tokens = np.concatenate((cls_expand, patches), axis=1)
        else:
            x_tokens = patches
        if x_tokens.shape[1] != self.pos_embedding.shape[1]:
            pe = self._resize_pos_embedding(self.pos_embedding, x_tokens.shape[1])
        else:
            pe = self.pos_embedding
        x_tokens = x_tokens + pe
        x_tokens = self._dropout(x_tokens)
        x_tokens = self.encoder.forward(x_tokens)
        x_tokens = self.norm.forward(x_tokens)
        return x_tokens

    def forward(self, x):
        features = self.forward_features(x)
        if self.uso_cls:
            cls_feat = features[:, 0, :]
            if self.head_w is None:
                return cls_feat
            return cls_feat @ self.head_w + self.head_b
        else:
            pooled = features.mean(axis=1)
            if self.head_w is None:
                return pooled
            return pooled @ self.head_w + self.head_b


# ---------- BlocoDecoder ----------
class BlocoDecoder:
    def __init__(self, dimensao, num_cabecas=8, dimensao_mlp=None, dropout=0.0, attn_dropout=0.0, seed=None):
        if dimensao_mlp is None:
            dimensao_mlp = 4 * dimensao
        self.norm1 = LayerNorm(dimensao, eps=1e-6)
        self.self_attn = MultiHeadSelfAttention(dimensao, num_cabecas, attn_dropout=attn_dropout, seed=seed)
        self.dropout1 = dropout
        self.norm2 = LayerNorm(dimensao, eps=1e-6)
        self.cross_attn = MultiHeadSelfAttention(dimensao, num_cabecas, attn_dropout=attn_dropout, seed=seed)
        self.dropout2 = dropout
        self.norm3 = LayerNorm(dimensao, eps=1e-6)
        self.mlp = MLP(dimensao, dimensao_mlp, dropout=dropout, seed=seed)

    def forward(self, x_dec, enc_kv=None, self_attn_mask=None, enc_key_padding_mask=None):
        x_norm = self.norm1.forward(x_dec)
        attn_out, _ = self.self_attn.forward(x_norm, attn_mask=self_attn_mask)
        if self.dropout1 > 0.0:
            attn_out = attn_out * (np.random.rand(*attn_out.shape) > self.dropout1) / (1.0 - self.dropout1)
        x = x_dec + attn_out
        if enc_kv is not None:
            x_norm = self.norm2.forward(x)
            cross_out, _ = self.cross_attn.forward(x_norm, k=enc_kv, v=enc_kv, key_padding_mask=enc_key_padding_mask)
            if self.dropout2 > 0.0:
                cross_out = cross_out * (np.random.rand(*cross_out.shape) > self.dropout2) / (1.0 - self.dropout2)
            x = x + cross_out
        x_norm = self.norm3.forward(x)
        mlp_out = self.mlp.forward(x_norm)
        x = x + mlp_out
        return x


# ---------- DecoderStack ----------
class DecoderStack:
    def __init__(self, num_blocos, dimensao, num_cabecas, dimensao_mlp=None, dropout=0.0, attn_dropout=0.0, seed=None):
        self.blocos = []
        for _ in range(num_blocos):
            self.blocos.append(BlocoDecoder(dimensao=dimensao, num_cabecas=num_cabecas,
                                            dimensao_mlp=dimensao_mlp, dropout=dropout,
                                            attn_dropout=attn_dropout, seed=seed))

    def forward(self, x_dec, enc_kv=None, self_attn_mask=None, enc_key_padding_mask=None):
        x = x_dec
        for b in self.blocos:
            x = b.forward(x, enc_kv=enc_kv, self_attn_mask=self_attn_mask, enc_key_padding_mask=enc_key_padding_mask)
        return x


# ---------- VisionTransformerDecoder ----------
class VisionTransformerDecoder:
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
        self.dimensao = dimensao_embedding
        self.uso_queries = uso_queries
        self.reconstruct = reconstruct
        self.patch_size = patch_size
        self.canais_saida = canais_saida
        rng = np.random.RandomState(seed)
        if uso_queries:
            assert num_queries is not None
            self.query = rng.normal(scale=0.02, size=(1, num_queries, dimensao_embedding)).astype(np.float32)
        else:
            self.query = None
        self.decoder = DecoderStack(num_blocos=num_blocos, dimensao=dimensao_embedding,
                                    num_cabecas=num_cabecas, dimensao_mlp=dimensao_mlp,
                                    dropout=dropout, attn_dropout=attn_dropout, seed=seed)
        self.norm = LayerNorm(dimensao_embedding, eps=1e-6)
        if reconstruct:
            assert num_queries is not None
            limite = 1.0 / math.sqrt(dimensao_embedding)
            self.head_recon_peso = rng.uniform(-limite, limite, size=(dimensao_embedding, patch_size*patch_size*canais_saida)).astype(np.float32)
            self.head_recon_bias = np.zeros((patch_size*patch_size*canais_saida,), dtype=np.float32)
        else:
            self.head_recon_peso = None
            self.head_recon_bias = None
        if num_classes is not None:
            limite2 = 1.0 / math.sqrt(dimensao_embedding)
            self.head_class_peso = rng.uniform(-limite2, limite2, size=(dimensao_embedding, num_classes)).astype(np.float32)
            self.head_class_bias = np.zeros((num_classes,), dtype=np.float32)
        else:
            self.head_class_peso = None
            self.head_class_bias = None

    def _build_causal_mask(self, num_q, num_k):
        return _build_causal_mask(num_q, num_k)

    def unpatchify(self, tokens, img_h, img_w):
        B, N, D = tokens.shape
        P = self.patch_size
        Hp = img_h // P
        Wp = img_w // P
        assert N == Hp * Wp
        patches = tokens @ self.head_recon_peso + self.head_recon_bias
        patches = patches.reshape(B, Hp, Wp, P, P, self.canais_saida)
        patches = patches.transpose(0, 5, 1, 3, 2, 4).copy()
        patches = patches.reshape(B, self.canais_saida, Hp*P, Wp*P)
        return patches

    def forward(self, decoder_inputs=None, encoder_tokens=None, encoder_padding_mask=None, img_size=None, causal=True):
        if self.uso_queries:
            assert encoder_tokens is not None
            B = encoder_tokens.shape[0]
            q = np.tile(self.query, (B, 1, 1))
        else:
            assert decoder_inputs is not None
            q = decoder_inputs
        L_dec = q.shape[1]
        self_attn_mask = None
        if causal:
            self_attn_mask = self._build_causal_mask(L_dec, L_dec)
        x = self.decoder.forward(q, enc_kv=encoder_tokens, self_attn_mask=self_attn_mask, enc_key_padding_mask=encoder_padding_mask)
        x = self.norm.forward(x)
        out_recon = None
        out_class = None
        if self.reconstruct:
            assert img_size is not None
            out_recon = self.unpatchify(x, img_size[0], img_size[1])
        if self.head_class_peso is not None:
            out_class = x @ self.head_class_peso + self.head_class_bias
        return {"tokens": x, "reconstruction": out_recon, "logits": out_class}


# ---------- teste rápido ----------
if __name__ == "__main__":
    B = 2
    img_h, img_w = 224, 224
    P = 16
    Hp = img_h // P
    Wp = img_w // P
    N = Hp * Wp
    D = 256

    rng = np.random.RandomState(0)
    imagens = rng.randn(B, 3, img_h, img_w).astype(np.float32)

    encoder = VisionTransformerEncoder(
        img_size=224, canais_entrada=3, tamanho_patch=16, dimensao_embedding=D,
        num_blocos=4, num_cabecas=8, dimensao_mlp=4*D, dropout=0.1, attn_dropout=0.1, uso_cls=False, seed=0
    )

    enc_tokens = encoder.forward_features(imagens)  # (B, N, D)
    print("enc_tokens.shape:", enc_tokens.shape)

    decoder = VisionTransformerDecoder(
        dimensao_embedding=D, num_blocos=4, num_cabecas=8, dimensao_mlp=4*D,
        dropout=0.1, attn_dropout=0.1, uso_queries=True, num_queries=N,
        reconstruct=True, patch_size=P, canais_saida=3, seed=1
    )

    out = decoder.forward(encoder_tokens=enc_tokens, encoder_padding_mask=None, img_size=(img_h, img_w), causal=False)
    recon = out["reconstruction"]
    tokens = out["tokens"]
    print("recon.shape:", None if recon is None else recon.shape)
    print("tokens.shape:", tokens.shape)

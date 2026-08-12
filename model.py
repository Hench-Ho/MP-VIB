"""
Multi-Prototype Variational Information Bottleneck (MP-VIB)
for SSL-Based Speech Spoofing Detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavlm.WavLM import WavLM, WavLMConfig


# --------------------------------------------------------------------------- #
#  WavLM front-end + learnable multi-layer weighted fusion 
# --------------------------------------------------------------------------- #
class SSLModel(nn.Module):
    """
    WavLM front-end that returns a learnable weighted sum of the hidden
    states of every Transformer layer.
    H_end = sum_l omega_l * LN(H^l),   omega = softmax(omega_tilde)
    """

    def __init__(self, wavlm_ckpt_path):
        super().__init__()
        checkpoint = torch.load(wavlm_ckpt_path, map_location="cpu")
        self.cfg = WavLMConfig(checkpoint["cfg"])
        self.wavlm = WavLM(self.cfg)
        self.wavlm.load_state_dict(checkpoint["model"])

        self.out_dim = self.cfg.encoder_embed_dim          # D (1024 for Large)
        self.num_layers = self.cfg.encoder_layers          # L (24 for Large)

        self.layer_weights = nn.Parameter(torch.zeros(self.num_layers))

    def forward(self, x):
        """x: (B, T_wave) raw waveform -> (B, T, D) fused frame-level features."""
        if self.cfg.normalize:
            x = F.layer_norm(x, x.shape[1:])

        rep, _ = self.wavlm.extract_features(x, ret_layer_results=True)
        _, layer_results = rep
        hidden_states = [h.transpose(0, 1) for h in layer_results]
        hidden_states = hidden_states[-self.num_layers:]

        stack = torch.stack(
            [F.layer_norm(h, (h.shape[-1],)) for h in hidden_states], dim=0
        )                                                   # (L, B, T, D)

        omega = torch.softmax(self.layer_weights, dim=0)    # (L,)
        h_end = torch.einsum("l,lbtd->btd", omega, stack)   # (B, T, D)
        return h_end


# --------------------------------------------------------------------------- #
#  Multi-Head Attentive Statistical Pooling (MHASTP)
# --------------------------------------------------------------------------- #
class MHASTP(nn.Module):
    """
    Multi-Head Attentive Statistical Pooling.
    """

    def __init__(self, dim, n_heads=8, bottleneck=128, eps=1e-6):
        super().__init__()
        assert dim % n_heads == 0, "feature dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.eps = eps

        self.attention = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, n_heads),
        )

    def forward(self, x):
        """x: (B, T, D) -> (B, 2*D) utterance-level statistics."""
        B, T, D = x.shape
        logits = self.attention(x)                          # (B, T, H)
        alpha = torch.softmax(logits, dim=1)                # attention over time

        xh = x.view(B, T, self.n_heads, self.head_dim)      # (B, T, H, d)
        alpha = alpha.unsqueeze(-1)                          # (B, T, H, 1)

        mean = torch.sum(alpha * xh, dim=1)                 # (B, H, d)
        var = torch.sum(alpha * xh * xh, dim=1) - mean * mean
        std = torch.sqrt(var.clamp(min=self.eps))          # (B, H, d)

        mean = mean.reshape(B, D)
        std = std.reshape(B, D)
        return torch.cat([mean, std], dim=1)               # (B, 2*D)


# --------------------------------------------------------------------------- #
#  MP-VIB: variational posterior + asymmetric class-conditional priors
# --------------------------------------------------------------------------- #
class MPVIB(nn.Module):
    """
    Multi-Prototype Variational Information Bottleneck.     
    """

    def __init__(self, in_dim, proj_dim=256, z_dim=512, n_spoof_prototypes=6):
        super().__init__()
        self.z_dim = z_dim
        self.n_spoof_prototypes = n_spoof_prototypes

        self.f_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
        )

        self.f_mu = nn.Linear(proj_dim, z_dim)
        self.f_logvar = nn.Linear(proj_dim, z_dim)

        self.bona_prototype = nn.Parameter(torch.zeros(z_dim))
        self.spoof_prototypes = nn.Parameter(torch.randn(n_spoof_prototypes, z_dim) * 0.1)
        self.spoof_mixture_logits = nn.Parameter(torch.zeros(n_spoof_prototypes))

    def encode(self, h):
        h = self.f_proj(h)
        mu = self.f_mu(h)
        logvar = self.f_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, h, sample=False):
        mu, logvar = self.encode(h)
        z = self.reparameterize(mu, logvar) if sample else mu
        return z, mu, logvar

    # ------------------------------------------------------------------ #
    #  KL terms - used only for training-time regularisation (reference)
    # ------------------------------------------------------------------ #
    def kl_bonafide(self, mu, logvar):
        """Closed-form KL[ N(mu, diag sigma^2) || N(c_b, I) ]  """
        var = torch.exp(logvar)
        diff = mu - self.bona_prototype                      # broadcast (B, z)
        kl = 0.5 * torch.sum(var + diff * diff - 1.0 - logvar, dim=1)
        return kl

    def kl_spoof(self, mu, logvar):
        """Monte-Carlo KL to the Gaussian mixture prior.

        L_KL = E_q[ log q(z|h) - log sum_k pi_k N(z; c_k, I) ]
        estimated with a single posterior sample z ~ q(z|h).
        """
        z = self.reparameterize(mu, logvar)                  # (B, z)
        log_q = -0.5 * torch.sum(
            logvar + (z - mu) ** 2 / torch.exp(logvar), dim=1
        )                                                    # (B,)

        log_pi = torch.log_softmax(self.spoof_mixture_logits, dim=0)  # (K,)
        diff = z.unsqueeze(1) - self.spoof_prototypes.unsqueeze(0)    # (B, K, z)
        log_comp = -0.5 * torch.sum(diff * diff, dim=2)              # (B, K)
        log_p = torch.logsumexp(log_pi.unsqueeze(0) + log_comp, dim=1)  # (B,)

        return log_q - log_p


# --------------------------------------------------------------------------- #
#  Full model
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    """End-to-end MP-VIB spoofing detector."""

    def __init__(self, wavlm_ckpt_path, z_dim=512, proj_dim=256,
                 n_spoof_prototypes=6, n_heads=8, n_classes=2):
        super().__init__()
        self.ssl = SSLModel(wavlm_ckpt_path)
        feat_dim = self.ssl.out_dim

        self.pooling = MHASTP(feat_dim, n_heads=n_heads)     # -> 2*feat_dim
        self.mpvib = MPVIB(
            in_dim=2 * feat_dim,
            proj_dim=proj_dim,
            z_dim=z_dim,
            n_spoof_prototypes=n_spoof_prototypes,
        )

        self.classifier = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim // 2, n_classes),
        )

    def forward(self, x, sample=False):
        """x: (B, T_wave) -> logits (B, 2), plus posterior (mu, logvar)."""
        h_end = self.ssl(x)                 # (B, T, D)  multi-layer fusion
        h = self.pooling(h_end)             # (B, 2D)    MHASTP
        z, mu, logvar = self.mpvib(h, sample=sample)   # (B, z_dim)
        logits = self.classifier(z)         # (B, 2)
        return logits, mu, logvar

    @torch.no_grad()
    def score(self, x):
        logits, _, _ = self.forward(x, sample=False)
        log_prob = torch.log_softmax(logits, dim=1)
        return log_prob[:, 1]               # index 1 == bonafide

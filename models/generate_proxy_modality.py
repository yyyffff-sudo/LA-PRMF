import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from .basic_layers import Transformer


# ============================================================
# Variational Encoder
# ============================================================
class VariationalEncoder(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = Transformer(
            num_frames=args['model']['vae']['input_length'],
            save_hidden=False,
            token_len=None,
            dim=hidden_dim,
            depth=args['model']['vae']['depth'],
            heads=args['model']['vae']['heads'],
            mlp_dim=args['model']['vae']['hidden_dim']
        )

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.encoder(h)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var


# ============================================================
# KL Divergence
# ============================================================
def s_kl_divergence(mu, log_var):
    q = Normal(mu, torch.exp(0.5 * log_var).clamp(min=1e-6))
    p = Normal(torch.zeros_like(mu), torch.ones_like(mu))
    return kl_divergence(q, p).mean()


# ============================================================
# Decoder
# ============================================================
class Decoder(nn.Module):
    def __init__(self, args, latent_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, hidden_dim)

        self.decoder = Transformer(
            num_frames=args['model']['vae']['input_length'],
            save_hidden=False,
            token_len=None,
            dim=hidden_dim,
            depth=args['model']['vae']['depth'],
            heads=args['model']['vae']['heads'],
            mlp_dim=args['model']['vae']['hidden_dim']
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = self.decoder(h)
        return self.fc_out(h)


# ============================================================
# VAE
# ============================================================
class VAE(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = VariationalEncoder(args, input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(args, latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var


# ============================================================
# Loss functions
# ============================================================
def recon_loss(x, x_recon):
    return F.mse_loss(x_recon, x)


def vae_loss(x, x_recon, mu, log_var):
    return recon_loss(x, x_recon) + s_kl_divergence(mu, log_var)


# ============================================================
# Proxy Modality Generation
# use_prompt=True  -> 使用模态Prompt加法注入（改进版）
# use_prompt=False -> 直接输入VAE（原始行为）
# ============================================================
class Generate_Proxy_Modality(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim, use_prompt=False):
        super().__init__()
        self.use_prompt = use_prompt

        if use_prompt:
            self.text_prompt = nn.Parameter(torch.zeros(1, 1, input_dim))
            self.vision_prompt = nn.Parameter(torch.zeros(1, 1, input_dim))
            self.audio_prompt = nn.Parameter(torch.zeros(1, 1, input_dim))
            nn.init.normal_(self.text_prompt, std=0.02)
            nn.init.normal_(self.vision_prompt, std=0.02)
            nn.init.normal_(self.audio_prompt, std=0.02)

        self.text_VAE = VAE(args, input_dim, hidden_dim, latent_dim)
        self.image_VAE = VAE(args, input_dim, hidden_dim, latent_dim)
        self.audio_VAE = VAE(args, input_dim, hidden_dim, latent_dim)

    def forward(self, text, video, audio, c_text=None, c_vision=None, c_audio=None, missing_mask=None):
        # Prompt 开关
        if self.use_prompt:
            text_in = text + self.text_prompt
            video_in = video + self.vision_prompt
            audio_in = audio + self.audio_prompt
        else:
            text_in = text
            video_in = video
            audio_in = audio

        # VAE forward
        t_recon, mu_t, log_var_t = self.text_VAE(text_in)
        v_recon, mu_v, log_var_v = self.image_VAE(video_in)
        a_recon, mu_a, log_var_a = self.audio_VAE(audio_in)

        # 重建目标：训练时用complete data，测试时用自身
        if c_text is not None and c_vision is not None and c_audio is not None:
            loss_t = vae_loss(c_text, t_recon, mu_t, log_var_t)
            loss_v = vae_loss(c_vision, v_recon, mu_v, log_var_v)
            loss_a = vae_loss(c_audio, a_recon, mu_a, log_var_a)
        else:
            loss_t = vae_loss(text, t_recon, mu_t, log_var_t)
            loss_v = vae_loss(video, v_recon, mu_v, log_var_v)
            loss_a = vae_loss(audio, a_recon, mu_a, log_var_a)

        # Cross-modal KL alignment
        std_t = torch.exp(0.5 * log_var_t).clamp(min=1e-6)
        std_v = torch.exp(0.5 * log_var_v).clamp(min=1e-6)
        std_a = torch.exp(0.5 * log_var_a).clamp(min=1e-6)

        qt = Normal(mu_t, std_t)
        qv = Normal(mu_v, std_v)
        qa = Normal(mu_a, std_a)

        kl_cross = (
            kl_divergence(qv, qt).mean() +
            kl_divergence(qa, qt).mean() +
            kl_divergence(qa, qv).mean()
        )

        # Proxy generation: uncertainty-weighted fusion
        std_stack = torch.stack([std_t, std_v, std_a], dim=0)
        weight = torch.exp(1.0 / std_stack)
        weight = weight / (weight.sum(dim=0, keepdim=True) + 1e-8)

        mu_stack = torch.stack([mu_t, mu_v, mu_a], dim=0)
        proxy_m = (weight * mu_stack).sum(dim=0)

        total_loss = (loss_t + loss_v + loss_a + kl_cross) / 3

        return total_loss, proxy_m, weight
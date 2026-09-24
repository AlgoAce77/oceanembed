
from __future__ import annotations

import io
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEPTHS = (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000)
INPUT_CHANNELS = ("SST", "SSS", "SSH", "U", "V", "Wind-U", "Wind-V", "Coriolis", "Bathymetry")

INPUT_MEAN = (27.5, 34.5, 0.0, 0.0, 0.0, 2.0, 1.0, 4.0e-5, 3000.0)
INPUT_STD = (2.0, 1.5, 0.10, 0.30, 0.30, 6.0, 4.0, 2.0e-5, 1500.0)
TEMP_MEAN = (27.5, 27.4, 27.3, 27.0, 26.5, 25.0, 23.5, 21.5, 19.8, 18.0, 15.5, 12.0, 8.6, 6.9, 5.7)
TEMP_STD = (1.6, 1.6, 1.6, 1.7, 1.9, 2.4, 2.6, 2.6, 2.5, 2.4, 2.2, 1.5, 0.8, 0.5, 0.4)


# ---------------------------------------------------------------------------------------
# Attention blocks
# ---------------------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """'Which inputs matter here?' e.g. salinity in the Bay of Bengal, wind near upwelling coasts."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1))
        self.last = None

    def forward(self, x):
        w = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        self.last = w.detach()
        return x * w


class SpatialAttention(nn.Module):
    """'Where does it matter?' e.g. eddy rims, river-plume fronts, coastal upwelling belts."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.last = None

    def forward(self, x):
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        w = torch.sigmoid(self.conv(pooled))
        self.last = w.detach()
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial(self.channel(x))


class DilatedResBlock(nn.Module):
    """3x3 convs with dilation: sees 25 km upwelling zones and 500 km Rossby waves without downsampling."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        self.cbam = CBAM(channels)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.cbam(self.norm2(self.conv2(y)))
        return self.act(x + y)


# ---------------------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------------------
class HSCAN(nn.Module):
    def __init__(self, in_channels: int = 9, n_depths: int = 15, base_channels: int = 48,
                 dilations: Sequence[int] = (1, 2, 4, 8, 1, 2, 4, 8),
                 logvar_min: float = -8.0, logvar_max: float = 4.0):
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be a multiple of 8 (GroupNorm).")
        self.cfg = dict(in_channels=in_channels, n_depths=n_depths, base_channels=base_channels,
                        dilations=list(dilations), logvar_min=logvar_min, logvar_max=logvar_max)
        self.logvar_min, self.logvar_max = logvar_min, logvar_max
        self.input_cbam = CBAM(in_channels, reduction=2)
        self.stem = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, base_channels), nn.GELU())
        self.blocks = nn.Sequential(*[DilatedResBlock(base_channels, d) for d in dilations])
        self.mean_head = nn.Sequential(nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(base_channels, n_depths, 1))
        self.var_head = nn.Sequential(nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(base_channels, n_depths, 1))

    def forward(self, x):
        feats = self.blocks(self.stem(self.input_cbam(x)))
        mu = self.mean_head(feats)
        logvar = self.var_head(feats).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar

    def set_variance_trainable(self, trainable: bool):
        """Curriculum: train the mean head with MSE first (variance frozen), then unfreeze and switch to NLL."""
        for p in self.var_head.parameters():
            p.requires_grad_(trainable)

    def attention_maps(self) -> dict:
        """Last-forward attention from the input CBAM: channel weights (B,C,1,1) and spatial map (B,1,H,W)."""
        return {"channel": self.input_cbam.channel.last, "spatial": self.input_cbam.spatial.last}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------------------
# Normalisation and numpy inference helpers
# ---------------------------------------------------------------------------------------
def normalize_inputs(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(INPUT_MEAN, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(INPUT_STD, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def predict_numpy(model: HSCAN, x_raw: np.ndarray, temp_mean: Sequence[float] | None = None,
                  temp_std: Sequence[float] | None = None):
    model.eval()
    x = normalize_inputs(torch.from_numpy(np.ascontiguousarray(x_raw)).float()[None])
    mu_n, logvar = model(x)
    tm = torch.tensor(temp_mean if temp_mean is not None else TEMP_MEAN, dtype=torch.float32).view(1, -1, 1, 1)
    ts = torch.tensor(temp_std if temp_std is not None else TEMP_STD, dtype=torch.float32).view(1, -1, 1, 1)
    mu = mu_n * ts + tm
    sigma = torch.exp(0.5 * logvar) * ts
    return mu[0].numpy(), sigma[0].numpy()


def load_checkpoint_bytes(raw: bytes):
    ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cfg = dict(ckpt.get("config", {})) if isinstance(ckpt, dict) else {}
    if "dilations" in cfg:
        cfg["dilations"] = tuple(cfg["dilations"])
    model = HSCAN(**cfg)
    result = model.load_state_dict(state, strict=False)
    stats = {}
    if isinstance(ckpt, dict):
        for key in ("temp_mean", "temp_std"):
            if key in ckpt:
                v = ckpt[key]
                stats[key] = [float(t) for t in (v.tolist() if hasattr(v, "tolist") else v)]
    return model, list(result.missing_keys), list(result.unexpected_keys), stats


def save_checkpoint(model: HSCAN, path: str, temp_mean: Sequence[float], temp_std: Sequence[float]):
    torch.save({"state_dict": model.state_dict(), "config": model.cfg,
                "temp_mean": [float(v) for v in temp_mean], "temp_std": [float(v) for v in temp_std]}, path)


# ---------------------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------------------
def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    mask = mask.expand_as(x)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None = None):
    """Heteroscedastic Gaussian negative log-likelihood, parameterised by log-variance for stability:
    0.5 * exp(-s) * (y - mu)^2 + 0.5 * s,   with s = log(sigma^2)."""
    nll = 0.5 * (torch.exp(-logvar) * (y - mu) ** 2 + logvar)
    return nll if weights is None else nll * weights


def sss_gradient_mask(sss: torch.Tensor, thr: float = 0.25, width: float = 0.1) -> torch.Tensor:
    """~1 where the salinity gradient (psu per pixel) is large, i.e. river-plume fronts / barrier layers."""
    gx = F.pad(sss[..., :, 1:] - sss[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(sss[..., 1:, :] - sss[..., :-1, :], (0, 0, 0, 1))
    return torch.sigmoid((torch.sqrt(gx ** 2 + gy ** 2 + 1e-12) - thr) / width)


def depth_region_weights(depths: torch.Tensor, sss: torch.Tensor, boost: float = 3.0) -> torch.Tensor:
    """Up-weight 50-150 m where strong SSS gradients hint at barrier layers (blueprint 'White Space 3')."""
    band = ((depths >= 50) & (depths <= 150)).float().view(1, -1, 1, 1)
    return 1.0 + boost * band * sss_gradient_mask(sss)


def linear_eos_density(temp_c, salinity, rho0=1025.0, alpha=2.0e-4, beta=7.6e-4, t0=10.0, s0=35.0):
    """Linearised equation of state (kg/m^3)."""
    return rho0 * (1.0 - alpha * (temp_c - t0) + beta * (salinity - s0))


def salinity_column_proxy(sss: torch.Tensor, depths: torch.Tensor, s_deep: float = 35.0, scale: float = 60.0):
    """We only predict temperature, so salinity below the surface is approximated by relaxing SSS to 35 psu."""
    return s_deep + (sss - s_deep) * torch.exp(-depths.view(1, -1, 1, 1) / scale)


def stratification_loss(temp_c: torch.Tensor, sss: torch.Tensor, depths: torch.Tensor,
                        mask: torch.Tensor | None = None, tol: float = 0.02) -> torch.Tensor:
    """Soft static-stability penalty: density must not decrease with depth. The penalty is relaxed under
    strong SSS gradients so genuine barrier-layer temperature inversions are not punished."""
    rho = linear_eos_density(temp_c, salinity_column_proxy(sss, depths))
    unstable = F.relu(rho[:, :-1] - rho[:, 1:] - tol)                 # (B, K-1, H, W)
    relax = 1.0 - 0.8 * sss_gradient_mask(sss)                        # (B, 1, H, W)
    return masked_mean(unstable * relax, mask)


def soft_d20(temp_c: torch.Tensor, depths: torch.Tensor, tau: float = 0.5):
    """Differentiable depth (m) of the 20 degC isotherm. Returns (depth (B,H,W), crossing evidence (B,H,W))."""
    s = torch.sigmoid((temp_c - 20.0) / tau)
    p = s[:, :-1] * (1.0 - s[:, 1:])
    t0, t1 = temp_c[:, :-1], temp_c[:, 1:]
    frac = ((t0 - 20.0) / (t0 - t1).clamp(min=1e-3)).clamp(0.0, 1.0)
    z0, z1 = depths[:-1].view(1, -1, 1, 1), depths[1:].view(1, -1, 1, 1)
    zc = z0 + frac * (z1 - z0)
    evidence = p.sum(dim=1)
    return (p * zc).sum(dim=1) / evidence.clamp(min=1e-6), evidence


def d20_loss(pred_c: torch.Tensor, target_c: torch.Tensor, depths: torch.Tensor,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    dp, ep = soft_d20(pred_c, depths)
    dt, et = soft_d20(target_c, depths)
    valid = ((ep > 0.5) & (et > 0.5)).float().unsqueeze(1)
    if mask is not None:
        valid = valid * mask
    return masked_mean((dp - dt).abs().unsqueeze(1) / 100.0, valid)


def combined_loss(mu_n: torch.Tensor, logvar: torch.Tensor, target_c: torch.Tensor, sss: torch.Tensor,
                  ocean_mask: torch.Tensor | None = None, temp_mean: Sequence[float] = TEMP_MEAN,
                  temp_std: Sequence[float] = TEMP_STD, depths: Sequence[float] = DEPTHS,
                  lam_phys: float = 0.05, lam_d20: float = 0.01, warmup: bool = False) -> dict:
    dev, dt_ = mu_n.device, mu_n.dtype
    tm = torch.tensor(temp_mean, dtype=dt_, device=dev).view(1, -1, 1, 1)
    ts = torch.tensor(temp_std, dtype=dt_, device=dev).view(1, -1, 1, 1)
    z = torch.tensor(depths, dtype=dt_, device=dev)
    y_n = (target_c - tm) / ts
    w = depth_region_weights(z, sss)
    if warmup:
        fit = masked_mean(w * (mu_n - y_n) ** 2, ocean_mask)
    else:
        fit = masked_mean(gaussian_nll(mu_n, logvar, y_n, w), ocean_mask)
    mu_c = mu_n * ts + tm
    strat = stratification_loss(mu_c, sss, z, ocean_mask)
    d20 = d20_loss(mu_c, target_c, z, ocean_mask)
    return {"total": fit + lam_phys * strat + lam_d20 * d20, "fit": fit, "strat": strat, "d20": d20}


# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    net = HSCAN(base_channels=32)
    x = torch.randn(2, 9, 101, 241)
    mu, lv = net(x)
    print("params:", count_parameters(net), "| mu:", tuple(mu.shape), "| logvar:", tuple(lv.shape))
    target = 20 + 5 * torch.randn(2, 15, 101, 241)
    sss = 34 + torch.rand(2, 1, 101, 241)
    for warm in (True, False):
        out = combined_loss(mu, lv, target, sss, warmup=warm)
        out["total"].backward(retain_graph=True)
        print("warmup" if warm else "nll   ", {k: round(float(v), 4) for k, v in out.items()})
    print("self-test OK")

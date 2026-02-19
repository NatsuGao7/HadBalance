import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def _ensure_prob(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Ensure input is within the (0,1) range to prevent numerical overflow."""
    return p.clamp(min=eps, max=1.0 - eps)

def sharpen_prob(p: torch.Tensor, tau: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Soft binarization with temperature annealing, making the probability distribution sharper and differentiable."""
    p = _ensure_prob(p, eps)
    logit = torch.log(p) - torch.log(1.0 - p)
    return torch.sigmoid(logit / max(tau, 1e-6))

def gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> torch.Tensor:
    """Construct a 1D Gaussian kernel."""
    if sigma <= 0:
        return torch.tensor([1.0])
    radius = int(math.ceil(truncate * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum()

def gaussian_blur(img: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Apply differentiable Gaussian blur to a single-channel image."""
    if sigma <= 0:
        return img
    B, C, H, W = img.shape
    g1d = gaussian_kernel1d(sigma).to(img.device, img.dtype)
    kx = g1d.view(1, 1, 1, -1)
    ky = g1d.view(1, 1, -1, 1)
    out = F.conv2d(img, kx, padding=(0, kx.shape[-1] // 2))
    out = F.conv2d(out, ky, padding=(ky.shape[-2] // 2, 0))
    return out

def sobel_grad(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Sobel gradients."""
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    return F.conv2d(img, kx, padding=1), F.conv2d(img, ky, padding=1)

def perimeter_density_from_prob(p: torch.Tensor, tau: float = 0.5, blur_sigma: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Approximate perimeter density using gradient magnitude."""
    p_sharp = sharpen_prob(p, tau=tau)
    p_smooth = gaussian_blur(p_sharp, sigma=blur_sigma)
    gx, gy = sobel_grad(p_smooth)
    return torch.sqrt(gx * gx + gy * gy + eps)

def local_area_maps(x, deltas):
    out = {}
    for d in deltas:
        out[int(d)] = F.avg_pool2d(x, kernel_size=d, stride=1, padding=d//2)  # in [0,1]
    return out


def local_perimeter_maps_from_prob(p, deltas, tau=0.5, blur_sigma=1.0):
    dens = perimeter_density_from_prob(p, tau=tau, blur_sigma=blur_sigma)
    out = {}
    for d in deltas:
        out[int(d)] = F.avg_pool2d(dens, kernel_size=d, stride=1, padding=d//2)  # density
    return out


def _unfold2x2(x: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) -> (B,4,H-1,W-1), unfold into 2x2 windows."""
    B, C, H, W = x.shape
    return F.unfold(x, kernel_size=2, stride=1).view(B, 4, H - 1, W - 1)

def soft_bitquad_counts(p: torch.Tensor, eps: float = 1e-6):
    """
    Compute Q1, Q3, QD (M3)
    Q1: Exactly one pixel is 1
    Q3: Exactly three pixels are 1 (One pixel is 0)
    QD: Diagonal is 1 (formerly M3)
    """
    p = _ensure_prob(p, eps)
    
    # 2x2 window slicing
    a = p[..., :-1, :-1] # Top-Left
    b = p[..., :-1, 1:]  # Top-Right
    c = p[..., 1:, :-1]  # Bot-Left
    d = p[..., 1:, 1:]   # Bot-Right
    
    na, nb, nc, nd = (1-a), (1-b), (1-c), (1-d)

    # Q1: Exactly one pixel is 1
    Q1 = (a*nb*nc*nd + b*na*nc*nd + c*na*nb*nd + d*na*nb*nc)
    
    # Q3: Exactly three pixels are 1 (One pixel is 0)
    Q3 = (na*b*c*d + nb*a*c*d + nc*a*b*d + nd*a*b*c)
    
    # QD: Diagonals (Exactly two pixels, diagonal) -> formerly M3
    QD = (a*d*nb*nc + b*c*na*nd)
    
    # We no longer need M2 (Orthogonal) since its coefficient in the Euler characteristic formula is 0
    return Q1, Q3, QD


def _pad_or_crop_to_hw(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Align spatial dimensions using only cropping or zero-padding, without interpolation (to avoid altering values)."""
    h, w = x.shape[-2], x.shape[-1]

    # crop if larger
    if h > H:
        x = x[..., :H, :]
        h = H
    if w > W:
        x = x[..., :, :W]
        w = W

    # pad if smaller (pad right/bottom)
    if h < H or w < W:
        x = F.pad(x, (0, W - w, 0, H - h), mode="constant", value=0.0)

    return x


def local_chi_maps(p: torch.Tensor, deltas: Iterable[int], eps: float = 1e-6) -> Dict[int, torch.Tensor]:
    p = _ensure_prob(p, eps)
    H, W = p.shape[-2], p.shape[-1]

    # Get Q1, Q3, QD
    Q1, Q3, QD = soft_bitquad_counts(p, eps=eps)

    out: Dict[int, torch.Tensor] = {}
    
    # Compatibility for delta=1 in test scripts
    deltas_loop = list(deltas)
    if 1 not in deltas_loop: pass
    
    for d in deltas_loop:
        d = int(d)
        
        if d == 1:
            chi = (Q1 - Q3 + 2.0 * QD) / 4.0
        else:
            pad = d // 2
            area = float(d * d)
            
            # [Key Modification]
            # We no longer multiply by area; instead, we directly use the result of avg_pool2d 
            # (i.e., average density). Thus, S_Q1 represents the "average probability of Q1 appearing 
            # in this window". The value range strictly remains in [0, 1], making it highly stable!
            
            S_Q1 = F.avg_pool2d(Q1, kernel_size=d, stride=1, padding=pad, count_include_pad=True)
            S_Q3 = F.avg_pool2d(Q3, kernel_size=d, stride=1, padding=pad, count_include_pad=True)
            S_QD = F.avg_pool2d(QD, kernel_size=d, stride=1, padding=pad, count_include_pad=True)
            
            # The formula remains the same, but its meaning becomes "average Euler density"
            chi = (S_Q1 - S_Q3 + 2.0 * S_QD) / 4.0

        # Align dimensions
        chi = _pad_or_crop_to_hw(chi, H, W)
        out[d] = chi

    return out


class LocalGeometry(nn.Module):
    """
    Generate multi-scale local geometric quantity maps (Area / Perimeter / Euler Characteristic).
    """
    def __init__(self, deltas=(8, 12, 16), tau=0.5, blur_sigma=1.0):
        super().__init__()
        self.deltas = deltas
        self.tau = tau
        self.blur_sigma = blur_sigma

    def forward(self, pred_prob: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, Dict[int, torch.Tensor]]:
        pred_prob = pred_prob.float()
        gt_mask = gt_mask.float()
        
        # Area
        A_pred = local_area_maps(pred_prob, self.deltas)
        A_gt   = local_area_maps(gt_mask, self.deltas)
        
        # Perimeter
        P_pred = local_perimeter_maps_from_prob(pred_prob, self.deltas, tau=self.tau, blur_sigma=self.blur_sigma)
        P_gt   = local_perimeter_maps_from_prob(gt_mask, self.deltas, tau=self.tau, blur_sigma=self.blur_sigma)
        
        # Euler characteristic (Chi)
        X_pred = local_chi_maps(pred_prob, self.deltas)
        X_gt   = local_chi_maps(gt_mask, self.deltas)

        return {
            "A_pred": A_pred, "A_gt": A_gt,
            "P_pred": P_pred, "P_gt": P_gt,
            "X_pred": X_pred, "X_gt": X_gt
        }

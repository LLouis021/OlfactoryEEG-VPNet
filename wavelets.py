#   (C) Ámon Attila Miklós
#   Eötvös Loránd University

import math
import numpy as np
import torch
import torch.nn.functional as F
from numpy.polynomial import polynomial as P


# =========================================================
# Morlet wavelet (SAFE VERSION)
# =========================================================
def genfun_morlet(m, n, params, p, r, dtype=torch.float, device=None):

    device = params.device if device is None else device
    dtype = params.dtype

    t = torch.linspace(-4, 4, m, device=device, dtype=dtype)

    morl = lambda t: torch.exp(-t**2 / 2) * torch.cos(5 * t)
    dmorl = lambda t: -t * torch.exp(-t**2 / 2) * torch.cos(5 * t) - torch.exp(-t**2 / 2) * torch.sin(5 * t) * 5

    psi = torch.zeros(m, n, device=device, dtype=dtype)
    dpsi = torch.zeros(m, 2 * n, device=device, dtype=dtype)
    ind = torch.zeros(2, 2 * n, device=device, dtype=torch.int64)

    for k in range(n):

        pars = params[2 * k:2 * k + 2]

        scale = torch.clamp(pars[0], min=0.05)
        shift = pars[1]

        u = (t - shift) / (scale + 1e-6)

        psi[:, k] = torch.exp(-u**2 / 2) * torch.cos(5 * u)
        psi[:, k] = psi[:, k] / torch.sqrt(scale + 1e-6)

        dpsi[:, 2 * k] = (
            -0.5 * scale ** (-3 / 2) * morl(u)
            - scale ** (-5 / 2) * (t - shift) * dmorl(u)
        )

        dpsi[:, 2 * k + 1] = -scale ** (-3 / 2) * dmorl(u)

        ind[0, 2 * k] = k
        ind[1, 2 * k] = 2 * k
        ind[0, 2 * k + 1] = k
        ind[1, 2 * k + 1] = 2 * k + 1

    return psi, dpsi, ind


# =========================================================
# Ricker wavelet (SAFE VERSION)
# =========================================================
def genfun_ricker(m, n, params, p, r, b_min, a, b, dtype=torch.float, device=None):

    device = params.device if device is None else device
    dtype = params.dtype

    t = torch.linspace(-5, 5, m, device=device, dtype=dtype)

    c = 2 / (math.sqrt(3) * math.sqrt(math.sqrt(np.pi)))

    rick = lambda t: c * torch.exp(-t**2 / 2) * (1 - t**2)
    drick = lambda t: c * (-t * torch.exp(-t**2 / 2) * (1 - t**2) - torch.exp(-t**2 / 2) * 2 * t)

    psi = torch.zeros(m, n, device=device, dtype=dtype)
    dpsi = torch.zeros(m, 2 * n, device=device, dtype=dtype)
    ind = torch.zeros(2, 2 * n, device=device, dtype=torch.int64)

    for k in range(n):

        pars = params[2 * k:2 * k + 2]

        scale = torch.clamp(pars[0], min=0.05)
        shift = pars[1]

        u = (t - shift) / (scale + 1e-6)

        psi[:, k] = rick(u) / torch.sqrt(scale + 1e-6)

        dpsi[:, 2 * k] = (
            -0.5 * scale ** (-3 / 2) * rick(u)
            - scale ** (-5 / 2) * (t - shift) * drick(u)
        )

        dpsi[:, 2 * k + 1] = -scale ** (-3 / 2) * drick(u)

        ind[0, 2 * k] = k
        ind[1, 2 * k] = 2 * k
        ind[0, 2 * k + 1] = k
        ind[1, 2 * k + 1] = 2 * k + 1

    return psi, dpsi, ind


# =========================================================
# Rational Gaussian wavelet (stability fixes only)
# =========================================================
def adaRatGaussWav(m, n, params, p, r, b_min, a, b,
                   smin=0.01, s_square=False,
                   dtype=torch.float, device=None):

    device = params.device if device is None else device
    dtype = params.dtype

    t = torch.linspace(a, b, m, device=device, dtype=dtype)

    Phi = torch.zeros(m, n, device=device, dtype=dtype)
    dPhi = torch.zeros(m, 2 * n, device=device, dtype=dtype)
    Ind = torch.zeros(2, 2 * n, device=device, dtype=torch.int64)

    for k in range(n):

        s = torch.clamp(params[2 * k], min=0.05)
        x = params[2 * k + 1]

        tt = (t - x) / (s + 1e-6)
        
        phi_val = torch.exp(-tt**2)
        Phi[:, k] = phi_val
        
        dPhi[:, 2 * k] = phi_val * (2 * tt**2) / (s + 1e-6)
        dPhi[:, 2 * k + 1] = phi_val * (2 * tt) / (s + 1e-6)
        
        Ind[0, 2 * k] = k
        Ind[1, 2 * k] = 2 * k
        Ind[0, 2 * k + 1] = k
        Ind[1, 2 * k + 1] = 2 * k + 1

    return Phi, dPhi, Ind


# =========================================================
# Hermite (stable device fix only)
# =========================================================
def hermite_ada(m, n, params, p, r, b_min, a, b,
                dtype=torch.float, device=None):

    device = params.device if device is None else device
    dtype = params.dtype

    dilation, translation = params

    t = torch.linspace(-5, 5, m, device=device, dtype=dtype)

    x = dilation * (t - translation)

    w = torch.exp(-0.5 * x ** 2)

    Phi = torch.zeros(m, n, device=device, dtype=dtype)
    dPhi = torch.zeros(m, 2 * n, device=device, dtype=dtype)
    ind = torch.zeros(2, 2 * n, device=device, dtype=torch.int64)

    Phi[:, 0] = w
    Phi[:, 1] = 2 * x * w

    return Phi, dPhi, ind

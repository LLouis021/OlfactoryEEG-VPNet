import torch
import torch.nn as nn
from torch.autograd.function import Function

class vp_layer(nn.Module):
    def __init__(
        self, ada, n_in, n_out, nparams,
        p, r, b_min, a, b,
        penalty=0.0, target=2, dtype=torch.float,
        device=None, init=None
    ):
        super().__init__()
        self.device = device
        self.target = target
        self.penalty = penalty
        
        # Wrap ada function to ensure it receives correct parameters
        self.ada = lambda params: ada(
            n_in, n_out, params, p, r, b_min, a, b,
            dtype=dtype, device=device
        )
        
        if init is None:
            # Core fix: according to wavelets.py logic, each wavelet needs 2 parameters
            # Must initialize to nparams * 2, otherwise index out of bounds error occurs
            init = torch.randn(nparams * 2, device=device) * 0.01 
            
        self.weight = nn.Parameter(init)

    def forward(self, x):
        return vpfun.apply(x, self.weight, self.ada, self.device, self.penalty, self.target)

class vpfun(Function):
    @staticmethod
    def forward(ctx, x, params, ada, device, penalty, target):
        # Call the generation function from wavelets.py
        phi, dphi, ind = ada(params)
        phi = phi.to(device)
        dphi = dphi.to(device)

        # Core fix: use absolutely safe fill_diagonal_ instead of manual indexing
        # This way, eye always perfectly aligns regardless of phi's shape
        eye = torch.zeros_like(phi)
        if phi.dim() == 2:
            eye.fill_diagonal_(1.0)
        else:
            # Compatible with Batch mode [Batch, N, N]
            for i in range(eye.size(0)):
                eye[i].fill_diagonal_(1.0)

        # Ridge regression regularized pseudo-inverse
        phip = torch.linalg.pinv(phi + 1e-5 * eye)

        x_trans = x.transpose(1, 2).contiguous()
        
        # Matrix multiplication to extract coefficients
        coeffs = phip @ x_trans
        y_est = (phi @ coeffs).transpose(1, 2).contiguous()

        # Save necessary tensors for backward pass
        ctx.save_for_backward(x, phi, phip, dphi, ind, coeffs, y_est, params)
        ctx.device = device
        ctx.penalty = penalty
        
        return coeffs, y_est

    @staticmethod
    def backward(ctx, grad_coeffs, grad_y_est):
        x, phi, phip, dphi, ind, coeffs, y_est, params = ctx.saved_tensors
        device = ctx.device
        
        # Auto-adapt dtype (mixed precision protection)
        target_dtype = grad_coeffs.dtype if grad_coeffs is not None else grad_y_est.dtype

        x = x.to(target_dtype)
        
        phi = phi.to(target_dtype)
        phip = phip.to(target_dtype)
        dphi = dphi.to(target_dtype)
        coeffs = coeffs.to(target_dtype)

        dx = torch.zeros_like(x)
        grad_phi = torch.zeros(phi.size(0), phi.size(1), device=phi.device, dtype=target_dtype)

        # Handle classification gradient from wavelet coefficients (from cross-entropy)
        if grad_coeffs is not None:
            dy_c = grad_coeffs.contiguous()
            # 1. Compute gradient w.r.t. input x (using pseudo-inverse transpose)
            dx += torch.matmul(phip.T, dy_c).transpose(1, 2)
            
            # 2. Compute gradient w.r.t. wavelet shape Phi (approximation: dPhi ≈ X_trans * dC^T)
            x_trans = x.transpose(1, 2).contiguous()
            grad_phi += torch.matmul(x_trans, dy_c.transpose(1, 2)).mean(0)

        # Handle reconstruction gradient from J_VP
        if grad_y_est is not None:
            dy_y = grad_y_est.contiguous()
            dx += dy_y - (dy_y @ phi @ phip)
            grad_phi += torch.matmul(dy_y.transpose(1, 2), coeffs.transpose(1, 2)).mean(0)

        # Update wavelet shape parameters (Scale and Shift)
        dp = torch.zeros_like(params)
        n_wavelets = phi.size(1)
        for k in range(n_wavelets):
            dp[2 * k] = torch.sum(grad_phi[:, k] * dphi[:, 2 * k])
            dp[2 * k + 1] = torch.sum(grad_phi[:, k] * dphi[:, 2 * k + 1])
        
        # 3. Stability: clamp parameter gradients to prevent wavelet shape collapse
        dp = torch.clamp(dp, -0.1, 0.1)
        
        # Return gradients for 6 parameters, only dx and dp have values
        return dx, dp, None, None, None, None

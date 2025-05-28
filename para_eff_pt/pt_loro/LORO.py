import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer




class LORO_optimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, update_k=5):
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            params = params[0]['params']
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, update_k=update_k)
        super().__init__(params, defaults)
        self.n_step = 0
        self.update_k = update_k
        self.is_exact = False

    @torch.no_grad()
    def step(self):
        # Print whether it's an approximate update or an exact update
        # if (self.n_step + 1) % self.update_k == 0:
        #     print(f"\nExact update at step {self.n_step}")
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']

            for i in range(0, len(group['params']), 2):
                A = group['params'][i]
                B = group['params'][i+1]

                if A.grad is None or B.grad is None:
                    continue

                # Initialize momentum buffers
                state_A = self.state[A]
                state_B = self.state[B]

                # Initialize momentum for A
                if len(state_A) == 0:
                    state_A['exp_avg'] = torch.zeros_like(A)  # first moment
                    state_A['exp_avg_sq'] = torch.zeros_like(A)  # second moment
                    state_A['step'] = 0

                # Initialize momentum for B
                if len(state_B) == 0:
                    state_B['exp_avg'] = torch.zeros_like(B)  # first moment
                    state_B['exp_avg_sq'] = torch.zeros_like(B)  # second moment
                    state_B['step'] = 0

                # Update steps
                state_A['step'] += 1
                state_B['step'] += 1

                # Apply weight decay
                if group['weight_decay'] != 0:
                    A.grad.data.add_(A.data, alpha=group['weight_decay'])
                    B.grad.data.add_(B.data, alpha=group['weight_decay'])

                # Update momentum (Adam style)
                # First moment
                state_A['exp_avg'].mul_(beta1).add_(A.grad, alpha=1 - beta1)
                state_B['exp_avg'].mul_(beta1).add_(B.grad, alpha=1 - beta1)

                # Second moment
                state_A['exp_avg_sq'].mul_(beta2).addcmul_(A.grad, A.grad, value=1 - beta2)
                state_B['exp_avg_sq'].mul_(beta2).addcmul_(B.grad, B.grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state_A['step']
                bias_correction2 = 1 - beta2 ** state_A['step']

                # Compute Adam-adjusted gradients
                denom_A = (state_A['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)
                denom_B = (state_B['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = group['lr'] / bias_correction1

                # Perform update using Adam-adjusted gradients
                if (self.n_step + 1) % group['update_k'] != 0:
                    # Approximate update
                    d, r = B.shape
                    A.data.addcdiv_(state_A['exp_avg'], denom_A, value=-step_size * (r/d))
                    B.data.addcdiv_(state_B['exp_avg'], denom_B, value=-step_size * (r/d))
                                
                else:
                    # Exact update, use momentum instead of gradients
                    original_dtype = B.dtype
                    
                    # Use momentum instead of raw gradients
                    A_grad_momentum = state_A['exp_avg'].div(denom_A).to(torch.float32)
                    B_grad_momentum = state_B['exp_avg'].div(denom_B).to(torch.float32)
                    
                    B_float = B.data.to(torch.float32)
                    A_float = A.data.to(torch.float32)
                    
                    # QR decomposition
                    Qb, Rb = torch.linalg.qr(B_float)
                    Qa, Ra = torch.linalg.qr(A_float.T)
                    
                    # Solve linear equations
                    dB_Ra_inv = torch.linalg.solve(Ra.T, B_grad_momentum.T).T
                    dA_Rb_inv = torch.linalg.solve(Rb.T, A_grad_momentum).T
                    
                    # Second QR decomposition
                    Qb_, Rb_ = torch.linalg.qr(dB_Ra_inv - Qb @ Qb.T @ dB_Ra_inv)
                    Qa_, Ra_ = torch.linalg.qr(dA_Rb_inv - Qa @ Qa.T @ dA_Rb_inv)
                    
                    # Compute S
                    S = Rb @ Ra.T - step_size * dA_Rb_inv.T @ Qa
                    
                    # Perform SVD
                    zeros = torch.zeros_like(Rb)
                    combined_matrix = torch.cat([
                        torch.cat([S, -step_size * Ra_.T], dim=1),
                        torch.cat([-step_size * Rb, zeros], dim=1)
                    ], dim=0)
                    
                    # Execute SVD and truncate
                    U, Sig, Vh = torch.linalg.svd(combined_matrix)
                    r = Rb.shape[0]
                    U = U[:, :r]
                    Sig = torch.diag(Sig[:r])
                    V = Vh[:r, :].T
                    
                    # Final updates
                    sqrt_Sig = torch.sqrt(Sig)
                    cat_Q = torch.cat([Qb, Qb_], dim=1)
                    cat_Qa = torch.cat([Qa, Qa_], dim=1)
                    
                    B_new = (cat_Q @ U @ sqrt_Sig).to(original_dtype)
                    A_new = (sqrt_Sig @ V.T @ cat_Qa.T).to(original_dtype)
                    
                    # Update parameters
                    A.data.copy_(A_new)
                    B.data.copy_(B_new)
                    
                    self.is_exact = True

        self.n_step += 1
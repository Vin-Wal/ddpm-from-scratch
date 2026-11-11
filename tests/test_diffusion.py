import torch
from ddpm.diffusion import q_sample

def test_q_sample_final_step():

    torch.manual_seed(0)

    betas = torch.linspace(1e-4, 0.02, 1000)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim = 0)
    x0 = torch.ones(16, 3, 32, 32)
    t = torch.full( (16, ), 999, dtype = torch.long)
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise, alphas_bar)

    assert xt.shape == x0.shape
    coeff = alphas_bar[t].reshape(-1, 1, 1, 1)
    expected_xt = torch.sqrt(coeff) * x0 + torch.sqrt(1 - coeff) * noise

    # Regression check against the DDPM equation; not an independent validation.
    assert torch.allclose(xt, expected_xt)
    
    assert abs(xt.mean().item()) < 0.02
    assert abs(xt.std().item() - 1) < 0.02






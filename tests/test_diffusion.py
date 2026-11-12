import torch
from ddpm.diffusion import q_sample
from ddpm.schedule import linear_beta_schedule, alphas_bar_from_betas

def test_q_sample_final_step():

    torch.manual_seed(0)

    betas = linear_beta_schedule(1000)
    alphas_bar = alphas_bar_from_betas(betas)

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






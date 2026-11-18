import torch

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):

    betas = torch.linspace(beta_start, beta_end,T)

    return betas

def alphas_bar_from_betas(betas):

    alphas_t = 1 - betas
    alphas_bar = torch.cumprod(alphas_t, dim = 0)

    return alphas_bar

def cosine_beta_schedule(T, s=0.008):
    timesteps = torch.arange(T+1, dtype = torch.float64)

    angle_t = (timesteps/T + s)*(torch.pi/2)/(1 + s)
    f_t = torch.cos(angle_t) ** 2

    alphas_t_bars = f_t/f_t[0]

    beta_t = 1 - (alphas_t_bars[1:]/alphas_t_bars[:-1])

    beta_t = torch.clamp(beta_t, max = 0.999)

    return beta_t.float()

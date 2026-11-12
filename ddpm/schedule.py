import torch

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):

    betas = torch.linspace(beta_start, beta_end,T)
    
    return betas
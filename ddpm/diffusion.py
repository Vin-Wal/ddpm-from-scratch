import torch

def q_sample(x0, t, noise, alphas_bar):

    alphas_bar = alphas_bar[t]

    alphas_bar = alphas_bar.reshape(-1, 1, 1, 1)


    signal_coeff = torch.sqrt(alphas_bar)
    signal_left = torch.mul(signal_coeff, x0)

    noise_coeff = torch.sqrt(1 - alphas_bar)
    noise_left = torch.mul(noise_coeff, noise)

    return (signal_left) + (noise_left)

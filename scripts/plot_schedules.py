
""""
Compares how quickly the alpha_bar_t(cumulative product) decays to 0 under the two schedules
"""

import torch
import matplotlib.pyplot as plt

from ddpm.schedule import linear_beta_schedule, cosine_beta_schedule, alphas_bar_from_betas
from config import SCHEDULE_ASSETS_DIR

T = 1000

linear_beta = linear_beta_schedule(T)
cosine_beta = cosine_beta_schedule(T)

linear_alphas_bar = alphas_bar_from_betas(linear_beta)
cosine_alphas_bar = alphas_bar_from_betas(cosine_beta)
t = torch.arange(1,T+1)/T

fig, ax = plt.subplots(figsize = (10, 5) )
ax.plot(t, linear_alphas_bar, label = 'Linear')
ax.plot(t, cosine_alphas_bar, label = 'Cosine')
ax.set_xlabel('Normalized timestep (t/T)')
ax.set_ylabel(r'$\bar{\alpha}_t$')
ax.set_title('Signal comparison Linear Vs Cosine Schedule ')
ax.legend()

plt.tight_layout()

SCHEDULE_ASSETS_DIR.mkdir(parents = True, exist_ok = True )
fig.savefig(SCHEDULE_ASSETS_DIR / "schedules.png")

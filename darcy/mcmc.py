# From https://neuraloperator.github.io/neuraloperator/dev/auto_examples/plot_FNO_darcy.html

import numpy as np
import torch
from ml_model import model
from neuralop.datasets import load_darcy_flow_small

from GRF import GaussianRF

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use 'Agg' for non-GUI backend

train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32,
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
        positional_encoding=True
)
test_samples = test_loaders[32].dataset

data = test_samples[0]
data = data_processor.preprocess(data, batched=False)
x = data['x'].squeeze()
y = data['y'].squeeze()

model.cuda()
x = x.cuda()
grid = x[1:].clone().cuda()
out = model(x.reshape(1,3,32,32)).squeeze()

print(x.shape, y.shape, out.shape, grid.shape)

truth = x.squeeze()[0].clone()
observed = y.squeeze()
gam = 0.1
observed = observed + gam * torch.std(observed) * torch.randn(32,32)
observed = observed.cuda()

def Phi(u1, obs, h=32*32, gamma=0.1):
    diff = u1.reshape(-1) - obs.reshape(-1)
    return 1/(2*gamma**2)*h*torch.norm(diff, p=2)**2

GRF = GaussianRF(dim=2, size=32, device="cuda")

def sample_a(N=1):
    v = GRF.sample(N).real
    v = v / torch.std(v)
    return v.cpu()

def model_reshaped(x):
    u = x.clone()
    u[u>0] = 1
    u[u<=0] = 0
    u = torch.cat([u.reshape(1,32,32).cuda(), grid], dim=0)
    out = model(u.reshape(1,3,32,32))
    return out.squeeze()

u = sample_a()
u1 = model_reshaped(u)

n_steps = 10000
beta = 0.05
samples = []
threshold_list = []

with torch.no_grad():

    for i in range(n_steps):
        if i % (n_steps//100) ==0:
            print(i)

        v = np.sqrt(1-beta**2)*u + beta * sample_a()

        v1 = model_reshaped(v)

        a = min(1, torch.exp(Phi(u1, observed) - Phi(v1, observed) ))
        if torch.rand(1).cuda() < a:
            u = v
            u1 = v1

        samples.append(u.clone())
        threshold_list.append(a)

    print(n_steps, )
    samples = torch.stack(samples, dim=0)
    posterior_mean = torch.mean(samples, dim=0)
    threshold_mean = sum(threshold_list) / len(threshold_list)
    print(threshold_mean)

    posterior_mean[posterior_mean>0] = 1
    posterior_mean[posterior_mean<0] = 0

    pred = model_reshaped(posterior_mean)

fig = plt.figure(figsize=(7, 7))

ax1 = fig.add_subplot(2, 2, 1)
im1 = ax1.imshow(truth.cpu().squeeze().numpy())
ax1.set_title('Truth x')
plt.xticks([], [])
plt.yticks([], [])
cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical')

ax2 = fig.add_subplot(2, 2, 2)
im2 = ax2.imshow(observed.cpu().squeeze().numpy())
ax2.set_title('Observed y')
plt.xticks([], [])
plt.yticks([], [])
cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical')

ax3 = fig.add_subplot(2, 2, 3)
im3 = ax3.imshow(posterior_mean.cpu().squeeze().numpy())
ax3.set_title('Posterior mean x')
plt.xticks([], [])
plt.yticks([], [])
cbar3 = fig.colorbar(im3, ax=ax3, orientation='vertical')

ax4 = fig.add_subplot(2, 2, 4)
im4 = ax4.imshow(pred.cpu().squeeze().numpy())
ax4.set_title('predcition of posterior mean x')
plt.xticks([], [])
plt.yticks([], [])
cbar4 = fig.colorbar(im4, ax=ax4, orientation='vertical')

fig.suptitle('Darcy with noise level=0.1', y=0.98)
plt.tight_layout()
plt.savefig("darcy_noisy.pdf")

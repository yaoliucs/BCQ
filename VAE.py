import numpy as np
import torch

from torch import nn, optim
from torch.nn import functional as F

import gym
from utils import ExtendedReplayBuffer
import matplotlib.pyplot as plt


class VAE_state(nn.Module):
    def __init__(self, state_dim, latent_dim, max_state, device):
        super(VAE_state, self).__init__()
        self.e1 = nn.Linear(state_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, state_dim)

        self.max_state = max_state
        self.latent_dim = latent_dim
        self.device = device

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def forward(self, state):
        z = F.relu(self.e1(state))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state.shape[0], z)

        return u, mean, std

    def decode(self, batch_size, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((batch_size, self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        s = F.relu(self.d1(z))
        s = F.relu(self.d2(s))
        if self.max_state is None:
            return self.d3(s)
        else:
            return self.max_state * torch.tanh(self.d3(s))

    def elbo_loss(self, state):
        recon, mean, std = self.forward(state)
        recon_loss = F.mse_loss(recon, state, reduction='none').mean(dim=-1)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        return recon_loss + 0.5 * KL_loss

    def recon_mse(self, state):
        recon, _, _ = self.forward(state)
        return F.mse_loss(recon, state, reduction='none').mean(dim=-1)

    def latent_KL(self, state):
        recon, mean, std = self.forward(state)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        return 0.5 * KL_loss


class VAE_gumbel(nn.Module):
    def __init__(self, state_dim, latent_dim, categorical_dim, device, temp=1.0, anneal_rate=0.00003, temp_min = 0.5):
        super(VAE_gumbel, self).__init__()

        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, state_dim)

        self.relu = nn.ReLU()

        self.device = device
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.temp = temp
        self.anneal_rate = anneal_rate
        self.temp_min = temp_min

    def anneal(self,step):
        self.temp = np.maximum(self.temp*np.exp(-self.anneal_rate*step), self.temp_min)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y.view(-1, self.latent_dim * self.categorical_dim)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, self.latent_dim * self.categorical_dim)

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x, hard):
        q = self.encode(x.view(-1, self.state_dim))
        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        z = self.gumbel_softmax(q_y, self.temp, hard)
        return self.decode(z), F.softmax(q_y, dim=-1).reshape(*q.size())

    # Reconstruction + KL divergence losses summed over all elements and batch
    def elbo_loss(self, x):
        recon_x, qy = self.forward(x, hard=False)
        RECON = F.mse_loss(recon_x, x.view(-1, self.state_dim), reduction='none').sum(dim=-1)
        log_ratio = torch.log(qy * self.categorical_dim + 1e-20)
        KLD = torch.sum(qy * log_ratio, dim=-1)
        return RECON + KLD

    def recon_mse(self, x):
        recon_x, _ = self.forward(x, hard=False)
        return F.mse_loss(recon_x, x.view(-1, self.state_dim), reduction='none').mean(dim=-1).detach()

    def latent_KL(self, x):
        recon_x, qy = self.forward(x, hard=False)
        log_ratio = torch.log(qy * self.categorical_dim + 1e-20)
        KLD = torch.sum(qy * log_ratio, dim=-1)
        return KLD.detach()

    def compute_frequency(self, x):
        recon_x, qy = self.forward(x, hard=False)
        self.frequency = qy.mean(dim=0).view(self.latent_dim, self.categorical_dim).detach()

    def frequency_score(self, x, hard=False):
        assert self.frequency is not None
        recon_x, qy = self.forward(x, hard)
        return torch.prod((qy * self.frequency).sum(dim=-1), dim=-1).detach()

def train(model, optimizer, replay_buffer, iterations, batch_size):
    losses = []
    model.train()
    for it in range(iterations):
        # Sample replay buffer / batch
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        loss = model.elbo_loss(state).mean()
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = model.elbo_loss(next_state).mean()
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if it % 1000 == 1:
        #     model.anneal(1000)
    return np.mean(losses)

def test(model, replay_buffer, batch_size):
    model.eval()
    state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
    score = -model.elbo_loss(state)
    return score.detach().cpu().numpy()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("Hopper-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latent_dim = 4
    categorical_dim = 10  # one-of-K vector
    temp_min = 0.5
    ANNEAL_RATE = 0.00003

    #model = VAE_gumbel(state_dim, latent_dim, categorical_dim, device)
    model = VAE_state(state_dim, latent_dim, None, device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    buffer_name = "Extended-Imperfect"
    replay_buffer = ExtendedReplayBuffer(state_dim, action_dim, env.init_qpos.shape[0],
                                               env.init_qvel.shape[0], device)
    replay_buffer.load(f"./buffers/{buffer_name}_Hopper-v3_0", 100000)

    training_iters = 0
    while training_iters < 200000:
        vae_loss = train(model, optimizer, replay_buffer, iterations=5000, batch_size=100)
        state, action, next_state, reward, not_done = replay_buffer.sample(1000)
        recon_mse = model.recon_mse(state).detach().cpu().numpy().mean()

        print(f"Training iterations: {training_iters}. State VAE loss: {vae_loss:.3f}. Recon MSE: {recon_mse:.3f}")
        #print("Temperature", model.temp)
        training_iters += 5000
    score = test(model, replay_buffer, batch_size=100000)
    np.save("results/vae_score",score)
    print(np.percentile(score, 2.0))
    plt.hist(score)


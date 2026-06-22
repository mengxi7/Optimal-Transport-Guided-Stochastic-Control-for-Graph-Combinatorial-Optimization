import math

import torch
from torch_geometric.data import Batch

from .models import Net, Noise_Net


def effective_epsilon(args):
    if getattr(args, "epsilon", None) is not None:
        return args.epsilon
    return getattr(args, "noise", 0.7)


class VSampler(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.leap_net = Net(args)
        self.refine_net = Net(args)
        self.arrive_net = Net(args)
        self.noise_param = Noise_Net(args)
        self.T = args.T
        self.h = 1.0 / self.T
        self.mode = args.mode
        self.epsilon = effective_epsilon(args)
        self.delta = args.delta
        self.mask_noise = getattr(args, "noise", 0.0)
        self.mask_noise_decay = getattr(args, "noise_decay", 5.0)
        self.mask_noise_value = getattr(args, "noise_value", 0.0)

    def forward(self, data: tuple[torch.Tensor, float], batch: Batch, r=1, sigma=None, dt=None) -> torch.Tensor:
        particle, t = data
        t_input = torch.tensor([t], device=particle.device).repeat(batch.num_nodes) * self.T

        for _ in range(r):
            v = self.refine_net(particle, batch, t_input)

        v_noise = None
        if sigma is not None:
            dt = (1.0 - self.delta) / self.T if dt is None else dt
            v_noise = self.euler_maruyama_step(particle, v, t, dt)

        return v, v_noise

    def euler_maruyama_step(self, particle, y_theta, t, dt):
        denom = max(1.0 - t, self.delta)
        drift = (y_theta - particle) / denom

        if self.epsilon > 0:
            gaussian_noise = math.sqrt(2.0 * self.epsilon * dt) * torch.randn_like(particle)
        else:
            gaussian_noise = torch.zeros_like(particle)

        state = (particle + drift * dt + gaussian_noise).clamp(min=0.0, max=1.0)
        return self._apply_step_zero_noise(state, t)

    def _apply_step_zero_noise(self, state, t):
        if self.mask_noise <= 0:
            return state

        horizon = max(1.0 - self.delta, 1e-8)
        step = min(max(int(round((t / horizon) * self.T)), 0), self.T)
        step_ratio = step / max(self.T, 1)
        prob = self.mask_noise * math.exp(-self.mask_noise_decay * step_ratio)
        prob = min(max(prob, 0.0), 1.0)

        if prob <= 0:
            return state

        mask = torch.rand_like(state) < prob
        fill = torch.full_like(state, self.mask_noise_value)
        return torch.where(mask, fill, state).clamp(min=0.0, max=1.0)

    def optimize(self, data: tuple[torch.Tensor, float], batch: Batch, r=2) -> torch.Tensor:
        particle, t = data
        t_input = torch.tensor([t], device=particle.device).repeat(batch.num_nodes) * self.T

        for _ in range(r):
            v = self.arrive_net(particle, batch, t_input)

        return v

    def sample(self, batch: Batch, sigma: float):
        particle = batch.x
        t_input = torch.zeros((batch.num_nodes,), device=particle.device) * 1000
        outputs = []

        for i in range(batch.T + 1):
            particle = self.refine_net(particle, batch, t_input)
            outputs.append(particle)

            particle_n, particle_p = particle < 0.5, particle > 0.5
            noise = torch.randn_like(particle, device=particle.device) * sigma
            noise_n, noise_p = noise[particle_n], noise[particle_p]
            noise_n[noise_n < 0] = -noise_n[noise_n < 0]
            noise_p[noise_p > 0] = -noise_p[noise_p > 0]
            noise[particle_n] = noise_n
            noise[particle_p] = noise_p
            t_input = torch.ones((batch.num_nodes,), device=particle.device) * (float(i) / batch.T) * 1000
            particle = (particle + noise).clamp(min=0.0, max=1.0)

        return outputs

    def arrive(self, particle, batch):
        return self.arrive_net(particle, batch).sigmoid()


@torch.no_grad()
def inference(args, model: VSampler, batch: Batch) -> dict:
    real_buffer = [batch.x]
    particle, buffer = batch.x, []
    horizon = 1.0 - args.delta

    if args.mode == "train":
        t_list = sorted((torch.rand((args.T,), device=particle.device) * horizon).tolist())
    else:
        t_list = torch.arange(0.0, horizon, horizon / args.T, device=particle.device).tolist()

    for idx, t in enumerate(t_list):
        next_t = t_list[idx + 1] if idx + 1 < len(t_list) else horizon
        dt = max(next_t - t, 1e-8)
        buffer.append((particle.detach(), t))
        _, particle_next = model.forward((particle, t), batch, sigma=True, dt=dt)
        particle = particle_next.detach()
        real_buffer.append(particle.detach())

    return buffer, real_buffer

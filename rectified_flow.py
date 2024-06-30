import torch


class RectifiedFlow:
    def __init__(self, model=None, num_steps=1000):
        self.model = model
        self.num_steps = num_steps

    def get_train_tuple(self, z0=None, z1=None):
        t = torch.rand((z1.shape[0], 1))
        z_t = t * z1 + (1. - t) * z0
        target = z1 - z0

        return z_t, t, target

    @torch.no_grad()
    def sample_ode(self, z0=None, sample_steps=None):
        if sample_steps is None:
            sample_steps = self.num_steps
        dt = 1. / sample_steps
        z = z0.detach().clone()
        batch_size = z.shape[0]
        for i in range(sample_steps):
            t = torch.ones((batch_size, 1)) * i / sample_steps
            pred = self.model(z, t)
            z = z.detach().clone() + pred * dt

        return z

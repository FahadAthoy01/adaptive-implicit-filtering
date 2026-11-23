import math
import torch
import torch.nn as nn


class SDFNetwork(nn.Module):
    def __init__(
        self,
        d_in: int = 3,
        d_out: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_in: tuple[int, ...] = (4,),
        geometric_init: bool = True,
        bias: float = 0.5,
    ) -> None:
        super().__init__()
        self.skip_in = set(skip_in)
        self.num_layers = num_layers

        dims = [d_in] + [hidden_dim] * num_layers + [d_out]
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1] - d_in if (i + 1) in self.skip_in else dims[i + 1]
            lin = nn.Linear(in_dim, out_dim)

            if geometric_init:
                if i == len(dims) - 2:
                    nn.init.normal_(lin.weight, mean=math.sqrt(math.pi) / math.sqrt(in_dim), std=1e-4)
                    nn.init.constant_(lin.bias, -bias)
                elif i == 0:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, math.sqrt(2.0 / out_dim))
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, math.sqrt(2.0 / out_dim))

            self.layers.append(lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_in:
                h = torch.cat([h, x], dim=-1) / math.sqrt(2.0)
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        y = self.forward(x)
        grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grad

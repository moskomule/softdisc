import torch

from softdisc import sinkhorn


def test_softmin():
    input = torch.randn(4, 3, 2)
    assert sinkhorn.softmin(input, 0.1, 2).size() == torch.Size([4, 3])
    assert sinkhorn.softmin(input, 0.1, 1).size() == torch.Size([4, 2])
    # for very small eps, softmin==min
    assert torch.allclose(sinkhorn.softmin(input, 0.001, 2), input.min(dim=2)[0])


def _centerize(c: torch.Tensor,
               x: torch.Tensor,
               y: torch.Tensor) -> torch.Tensor:
    # `BxNxM` -> `BxNxM`
    return c - x.unsqueeze(-1) - y.unsqueeze(1)


def test_log_sinkhorn():
    # just check it is runnable
    x = torch.zeros(1, 3)
    x[0, 0] = 1
    y = torch.randn(1, 2)
    x[0, 1] = 1
    a = torch.ones_like(x) / 3
    b = torch.ones_like(y) / 2
    print(sinkhorn.log_sinkhorn(x, y, a, b, 0.01, 2, 0.0001))


def test_softquantizer():
    x = torch.rand(1, 3)
    q = sinkhorn.SoftQuantizer(x, num_targets=3)
    print(q.transport)
    print(q.softcdf)
    print(q.softsort)

import torch
from src.model import FashionCNN


def test_output_shape():
    out = FashionCNN()(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_can_overfit_tiny_batch():
    torch.manual_seed(0)
    m = FashionCNN(dropout=0.0)
    x, y = torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(50):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward(); opt.step()
    assert (m(x).argmax(1) == y).float().mean() == 1.0

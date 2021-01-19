import torch
import torch.nn.functional as F
from typing import Optional

class ContentLoss(torch.nn.Module):
    """
    Calculates the content loss between given features and input features.

    Args:
        content: the tensor (1, c, h, w) of features from one layer of the content image.
    """

    loss: Optional[torch.Tensor]
    content: torch.Tensor

    def __init__(self, content: torch.Tensor):        
        super().__init__()

        self.content = content.detach()
        self.loss = None

    def forward(self, x):
        self.loss = F.mse_loss(self.content, x)
        return x


class StyleLoss(torch.nn.Module):
    """
    Calculates the style loss between given features and input features.

    Args:
        style: the tensor (1, c, h, w) of features from one layer of the style image.
    """

    loss: Optional[torch.Tensor]
    style: torch.Tensor

    def __init__(self, style: torch.Tensor):
        super().__init__()

        self.style_gram = self._gram_matrix(style.detach())
        self.loss = None

    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the gram matrix of a tensor x.

        Args:
            x: tensor of shape (1, c, h, w).
        """

        b, c, h, w = x.shape
        x = x.view(c, h * w)
        result = torch.matmul(x, torch.transpose(x, 0, 1))
        result = result.div(float(c * h * w))

        return result

    def forward(self, x):
        self.loss = F.mse_loss(self.style_gram, self._gram_matrix(x))
        return x


class TotalVariationLoss(torch.nn.Module):
    """
    Calculates the total variation loss of a tensor.
    """

    loss: Optional[torch.Tensor]

    def __init__(self):
        super().__init__()

        self.loss = None

    def forward(self, x):
        b, c, h, w = x.shape

        a = torch.square(x[:, :, :h - 1, :w - 1] - x[:, :, 1:, :w - 1])
        b = torch.square(x[:, :, :h - 1, :w - 1] - x[:, :, :h - 1, 1:])
        self.loss = torch.mean(torch.pow(a + b, 1.25))
        return x
from typing import List

import torch
import torch.nn.functional as F
import math


class DynamicMergeNet(torch.nn.Module):
    def __init__(
            self,
            models: List[torch.nn.Module],
    ):
        super(DynamicMergeNet, self).__init__()
        self.models = models

    def forward(self, x: torch.Tensor):
        for model in self.models:
            x = model(x)
        return x
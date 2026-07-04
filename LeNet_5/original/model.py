import torch
import torch.nn as nn
import torch.nn.functional as F


# Activation function used in the paper
class ScaledTanh(nn.Module):
    def forward(self, x):
        # 1.7159: <Gradient-based learning applied to document recognition>
        # 1.7159 and 2/3: <Efficient BackProp> (also by Yann LeCun)
        return 1.7159 * torch.tanh(2 / 3 * x)


# Subsampling layer used in the paper
class Subsampling(nn.Module):
    def __init__(self, num_maps):
        super().__init__()
        self.coeff = nn.Parameter(torch.full((num_maps,), 0.25))
        self.bias = nn.Parameter(torch.zeros(num_maps))
        self.act = ScaledTanh()

    def forward(self, x):   # x: (N, C, H, W)
        s = F.avg_pool2d(x, 2) * 4
        s = s * self.coeff.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return self.act(s)


# C3 connection table used in the paper
C3_CONNECTIONS = [
    [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [0, 4, 5], [0, 1, 5],             # 3개 입력
    [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [0, 3, 4, 5], [0, 1, 4, 5], [0, 1, 2, 5],  # 4개 입력(연속)
    [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5],                                     # 4개 입력(비연속)
    [0, 1, 2, 3, 4, 5],                                                           # 전부
]


class C3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 16, kernel_size=5)     # 일단 완전 연결 conv
        mask = torch.zeros(16, 6, 1, 1)                 # 연결 마스크
        for out_map, in_maps in enumerate(C3_CONNECTIONS):
            for in_map in in_maps:
                mask[out_map, in_map] = 1.0
        self.register_buffer("mask", mask)              # 학습X, .to(device) 따라감

    def forward(self, x):
        # 연결 안 된 입력의 weight를 0으로 마스킹해서 conv
        return F.conv2d(x, self.conv.weight * self.mask, self.conv.bias)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(1, 6, kernel_size=5)
        self.S2 = Subsampling(6)
        self.C3 = C3()
        self.S4 = Subsampling(16)
        self.C5 = nn.Conv2d(16, 120, kernel_size=5)
        self.F6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)
        self.act = ScaledTanh()

    def forward(self, x):
        x = self.act(self.C1(x))
        x = self.S2(x)
        x = self.act(self.C3(x))
        x = self.S4(x)
        x = self.act(self.C5(x))
        x = torch.flatten(x, 1)
        x = self.act(self.F6(x))
        logits = self.out(x)
        return logits

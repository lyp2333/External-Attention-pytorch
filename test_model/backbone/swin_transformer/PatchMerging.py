import torch
from torch import nn


class PatchMerging(nn.Module):
    def __init__(self, input_patch_res, in_chans, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.patch_res = input_patch_res
        self.dim = in_chans
        self.reduction = nn.Linear(4 * self.dim, 2 * self.dim, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(4 * self.dim)
        else:
            self.norm = None

    def forward(self, input):
        # input.shape = B, Patch_num ,C
        B, patch_num, C = input.shape
        H, W = self.patch_res
        assert patch_num == self.patch_res[0] * self.patch_res[1], \
            f'输入与要求尺寸({self.patch_res[0]},{self.patch_res[1]})不符'

        assert self.patch_res[0] % 2 == 0 and self.patch_res[1] % 2 == 0, \
            f'设置的输入patch分辨率不是偶数'

        input = input.reshape(B, H, W, C)

        x1 = input[:, 0::2, 0::2, :]  # 左上 shape = B,H/2,W/2,C,以下均是
        x2 = input[:, 1::2, 0::2, :]  # 右上
        x3 = input[:, 0::2, 1::2, :]  # 左下
        x4 = input[:, 1::2, 1::2, :]  # 右下

        x = torch.cat([x1, x2, x3, x4], -1)
        x = x.reshape(B, -1, 4 * C)  # shape = B,h//2,w//2,4C

        if self.norm is not None:
            x = self.norm(x)
        output = self.reduction(x)
        return output

    def extra_repr(self) -> str:
        return f'input_patch_resolution+{self.patch_res},input_channels={self.dim}'

    def flops(self):
        H, W = self.patch_res[0], self.patch_res[1]
        ho, wo = self.patch_res[0] // 2, self.patch_res[1] // 2
        norm_flops = ho * wo * (4 * self.dim)
        reduction_flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return norm_flops + reduction_flops

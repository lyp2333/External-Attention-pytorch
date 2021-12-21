from torch import nn
from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.patch_num = self.patches_resolution[0] * self.patches_resolution[1]
        # 用于生成patch
        self.in_channels = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, input):
        B, C, H, W = input.size()

        assert H == self.img_size[0] and W == self.img_size[1], \
            f'输入图像尺寸({H},{W})与要求图像尺寸({self.img_size[0]},{self.img_size[1]})不符'
        x = self.proj(input).flatten(2).transpose(1, 2)
        if (self.norm is not None):
            output = self.norm(x)
        else:
            output = x

        return output  # output.shape=B ,Patch token 个数, C

    def extra_repr(self) -> str:
        return f'img_size={self.img_size},patch_size={self.patch_size},in_channels={self.in_channels},' \
               f'embedding_channel={self.embed_dim}'
    def flops(self):
        flops = self.in_channels * self.embed_dim * self.patches_resolution[0] * \
                self.patches_resolution[1] * self.patch_size[0] * self.patch_size[1]
        if self.norm is not None:
            flops += self.patches_resolution[0] * self.patches_resolution[1] * self.embed_dim
        return flops

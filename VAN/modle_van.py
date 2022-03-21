import torch.nn as nn
from timm.models.layers import DropPath


class AttentionModle(nn.Module):
    """LKAע��������, �����˳ߴ�Ϊ21"""

    def __init__(self, dim):
        super(AttentionModle, self).__init__()
        self.conv_0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, dilation=3, padding=9, groups=dim)
        self.conv_1 = nn.Conv2d(dim, dim, 1)

    def forward(self):
        u = x.clone()
        attn = self.conv_0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv_1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    """��LKA����ƿ���Ͳв�"""

    def __init__(self, d_model):
        super(SpatialAttention, self).__init__()

        self.conv0 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModle(d_model)
        self.conv1 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.conv0(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.conv1(x)
        x = x + shortcut
        return x


class Block(nn.Module):
    """VAN��һ��BLOCK��û��д��MLP"""

    def __init__(self, dim, drop_rate=0, layer_scale_init_value=1e-2):
        super(block, self).__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()
        x = self.attn(self.norm1)
        if self.layer_scale_1 is not None:
            x = self.layer_scale_1 * x
        x = self.drop_path(x) + shortcut
        return x


class VAN(nn.Module):
    """VAN������ܹ���û��patch embedding"""

    def __init__(self, img_size=224, in_chans=3, num_classes=11, drop_rate=0, depths=[3, 3, 5, 2],
                 dims=[32, 64, 160, 5], layer_scale_init_value=1e-2):
        super(VAN, self).__init__()
        # ʹ��modulelist���ӵ�ģ�黯
        self.downsample_layers = nn.ModuleList()
        # ����
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=4, padding=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # ���������²����㣬������һ��LIST����
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.LayerNorm(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        # �ĸ�stage�㣬Ҳ����һ��LIST����
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        """һֱ���������������ͼ"""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # (N, C, H, W) -> (N, C)ȡƽ��ֵ
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        feature = self.forward_features(x)
        out = self.head(feature)
        return out


def van_tiny(num_classes):
    model = VAN(
        depths=[8, 8, 4, 4],
        dims=[32, 64, 160, 256]
    )
    return model

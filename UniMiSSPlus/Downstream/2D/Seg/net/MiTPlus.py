import torch
import torch.nn as nn
import logging
import math
from net import MiTPlus_encoder as MiTPlus_encoder
from net.utils import trunc_normal_

logger = logging.getLogger(__name__)


class Conv2dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,norm_cfg,activation_cfg,kernel_size,stride=1,padding=0,dilation=1,bias=False):
        super(Conv2dBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x


def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN1':
        out = nn.BatchNorm1d(inplanes)
    elif norm_cfg == 'BN2':
        out = nn.BatchNorm2d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN2':
        out = nn.InstanceNorm2d(inplanes, affine=True)
    elif norm_cfg == 'LN':
        out = nn.LayerNorm(inplanes, eps=1e-6)

    return out


def Activation_layer(activation_cfg, inplace=True):

    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr2D = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, dims):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            if len(x.view(-1)) == (B * C * dims[0] * dims[1]):
                x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, dims[0], dims[1])
                x_ = self.sr2D(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            else: 
                x_ = x[:, 1::].permute(0, 2, 1).contiguous().reshape(B, C, dims[0], dims[1])
                x_ = self.sr2D(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
                x_ = torch.cat((x[:, 0:1], x_), dim=1)

            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, dims):
        x = x + self.drop_path(self.attn(self.norm1(x), dims))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed2D_de(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=224, patch_size=16, in_chans=3, embed_dim=768, is_proj1=False):
        super().__init__()
        num_patches = (img_size * patch_size) * (img_size * patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.is_proj1 = is_proj1

        self.proj = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if self.is_proj1:
            self.proj1 = Conv2dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.is_proj1:
            x = self.proj1(self.proj(x)).flatten(2).transpose(1, 2).contiguous()
        else:
            x = self.proj(x).flatten(2).transpose(1, 2).contiguous()

        return x, (H * self.patch_size, W * self.patch_size)


class MiT(nn.Module):

    def __init__(self, norm_cfg2D='BN2', activation_cfg='ReLU', img_size2D=224,
                 num_classes=None, embed_dims=[320, 256, 128, 64, 32], num_heads=[4, 2, 1], mlp_ratios=[4, 8, 8], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[2, 4, 6], is_proj1=False):

        super().__init__()
        self.MODEL_NUM_CLASSES = num_classes
        self.embed_dims = embed_dims
        self.embed_dim = embed_dims[2]

        # Encoder
        self.transformer = MiTPlus_encoder.encoder_small(norm_cfg2D=norm_cfg2D, activation_cfg=activation_cfg, img_size2D=img_size2D, is_proj1=True)

        # Decoder patchEmbed
        # 2D
        self.DecEmbed2D1 = PatchEmbed2D_de(norm_cfg=norm_cfg2D, activation_cfg=activation_cfg, img_size=img_size2D // 16,
                                        patch_size=2, in_chans=self.transformer.embed_dims[-1], embed_dim=embed_dims[0], is_proj1=is_proj1)
        self.DecEmbed2D2 = PatchEmbed2D_de(norm_cfg=norm_cfg2D, activation_cfg=activation_cfg, img_size=img_size2D // 8,
                                        patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], is_proj1=is_proj1)
        self.DecEmbed2D3 = PatchEmbed2D_de(norm_cfg=norm_cfg2D, activation_cfg=activation_cfg, img_size=img_size2D // 4,
                                        patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], is_proj1=is_proj1)

        # pos_embed
        self.DecPosEmbed2D1 = nn.Parameter(torch.zeros(1, self.transformer.patch_embed2D3.num_patches, embed_dims[0]))
        self.DecPosDrop2D1 = nn.Dropout(p=drop_rate)
        self.DecPosEmbed2D2 = nn.Parameter(torch.zeros(1, self.transformer.patch_embed2D2.num_patches, embed_dims[1]))
        self.DecPosDrop2D2 = nn.Dropout(p=drop_rate)
        self.DecPosEmbed2D3 = nn.Parameter(torch.zeros(1, self.transformer.patch_embed2D1.num_patches, embed_dims[2]))
        self.DecPosDrop2D3 = nn.Dropout(p=drop_rate)

        trunc_normal_(self.DecPosEmbed2D1, std=.02)
        trunc_normal_(self.DecPosEmbed2D2, std=.02)
        trunc_normal_(self.DecPosEmbed2D3, std=.02)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.Decblock1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.Decblock2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        cur += depths[1]
        self.Decblock3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        self.norm = norm_layer(embed_dims[2])

        self.TransposeConv2D1 = nn.ConvTranspose2d(embed_dims[2], embed_dims[3], kernel_size=2, stride=2)
        self.DeConvBlock2D1 = Conv2dBlock(embed_dims[3], embed_dims[3], norm_cfg=norm_cfg2D, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1)

        self.TransposeConv2D2 = nn.ConvTranspose2d(embed_dims[3], embed_dims[4], kernel_size=2, stride=2)
        # self.DeConvBlock2D2 = Conv2dBlock(embed_dims[4], embed_dims[4], norm_cfg=norm_cfg2D, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1)

        self.seg_conv2d = nn.Conv2d(embed_dims[4], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding2D(self, pos_embed, x, w, h):

        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return pos_embed
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w
        h0 = h
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2).contiguous(),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).contiguous().reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def forward(self, inputs):
        B = inputs.shape[0]

        ####### encoder
        x_encoder_2D, (H, W) = self.transformer(inputs)
        x_trans = x_encoder_2D[-1].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ####### decoder
        # stage 0
        x, (H, W) = self.DecEmbed2D1(x_trans)
        x = x + x_encoder_2D[-2].flatten(2).transpose(1, 2)
        x = x + self.interpolate_pos_encoding2D(self.DecPosEmbed2D1, x, W, H)
        x = self.DecPosDrop2D1(x)
        for blk in self.Decblock1:
            x = blk(x, (H, W))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 1
        x, (H, W) = self.DecEmbed2D2(x)
        x = x + x_encoder_2D[-3].flatten(2).transpose(1, 2)
        x = x + self.interpolate_pos_encoding2D(self.DecPosEmbed2D2, x, W, H)
        x = self.DecPosDrop2D2(x)
        for blk in self.Decblock2:
            x = blk(x, (H, W))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x, (H, W) = self.DecEmbed2D3(x)
        x = x + x_encoder_2D[-4].flatten(2).transpose(1, 2)
        x = x + self.interpolate_pos_encoding2D(self.DecPosEmbed2D3, x, W, H)
        x = self.DecPosDrop2D3(x)
        for blk in self.Decblock3:
            x = blk(x, (H, W))
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x = self.TransposeConv2D1(x)
        skip3 = x_encoder_2D[-5]
        x = x + skip3
        x = self.DeConvBlock2D1(x)

        # stage 4
        x = self.TransposeConv2D2(x)
        skip4 = x_encoder_2D[-6]
        x = x + skip4
        outputs = self.seg_conv2d(x)

        return outputs



def MiTPlus(norm_cfg2D='IN2', activation_cfg='LeakyReLU', img_size2D=224, is_proj1=True, **kwargs):
    model = MiT(norm_cfg2D=norm_cfg2D, activation_cfg=activation_cfg, img_size2D=img_size2D, 
                embed_dims=[320, 256, 128, 64, 32], num_heads=[4, 2, 1], depths=[1, 1, 1], 
                mlp_ratios=[4, 4, 4], sr_ratios=[2, 4, 8], is_proj1=is_proj1, **kwargs)

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of Model Params: %.2f(e6)' % (total / 1e6))

    return model
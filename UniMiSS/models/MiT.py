import torch
import torch.nn as nn
import logging
import math
from models import MiT_encoder as MiT_encoder
from utils import trunc_normal_

logger = logging.getLogger(__name__)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 modal_type='MM'):
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
            if modal_type == '2D':
                self.sr2D = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            elif modal_type == '3D':
                self.sr3D = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            elif modal_type == 'MM':
                self.sr2D = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.sr3D = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)

            self.norm = nn.LayerNorm(dim)

    def forward(self, x, dims, modal_type):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            if modal_type == '2D':
                x_ = x[:, 1::].permute(0, 2, 1).reshape(B, C, dims[0], dims[1])
                x_ = self.sr2D(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = torch.cat((x[:, 0:1], x_), dim=1)
            elif modal_type == '3D':
                x_ = x[:, 1::].permute(0, 2, 1).reshape(B, C, dims[0], dims[1], dims[2])
                x_ = self.sr3D(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = torch.cat((x[:, 0:1], x_), dim=1)

            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, modal_type='MM'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, modal_type=modal_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, dims, modal_type):
        x = x + self.drop_path(self.attn(self.norm1(x), dims, modal_type))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed2D(nn.Module):
    """ Image to 2D Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x, (H * self.patch_size, W * self.patch_size)


class PatchEmbed3D(nn.Module):
    """ Image to 3D Patch Embedding
    """

    def __init__(self, img_size=[16, 96, 96], patch_size=[16, 16, 16], in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] * patch_size[0]) * (img_size[1] * patch_size[1]) * (img_size[2] * patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.ConvTranspose3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x, (D * self.patch_size[0], H * self.patch_size[1], W * self.patch_size[1])


class MiT(nn.Module):

    def __init__(self, norm_cfg2D='BN2', norm_cfg3D='BN3', activation_cfg='ReLU', weight_std=False,
                 img_size2D=224, img_size3D=[16, 96, 96], embed_dims=[192, 64, 32], num_heads=[8, 4, 2],
                 mlp_ratios=[4, 4, 4],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[2, 2, 2], sr_ratios=[2, 2, 4], modal_type='MM', encoder='small'):

        super().__init__()
        self.embed_dim = embed_dims[2]
        self.img_size2D = img_size2D
        self.img_size3D = img_size3D

        # Encoder
        if encoder == 'tiny':
            self.transformer = MiT_encoder.encoder_tiny(norm_cfg2D=norm_cfg2D, norm_cfg3D=norm_cfg3D,
                                                  activation_cfg=activation_cfg,
                                                  weight_std=weight_std, img_size2D=img_size2D, img_size3D=img_size3D,
                                                  modal_type=modal_type)
        elif encoder == 'small':
            self.transformer = MiT_encoder.encoder_small(norm_cfg2D=norm_cfg2D, norm_cfg3D=norm_cfg3D,
                                                 activation_cfg=activation_cfg,
                                                 weight_std=weight_std, img_size2D=img_size2D, img_size3D=img_size3D,
                                                 modal_type=modal_type)
        elif encoder == 'large':
            self.transformer = MiT_encoder.encoder_large(norm_cfg2D=norm_cfg2D, norm_cfg3D=norm_cfg3D,
                                                 activation_cfg=activation_cfg,
                                                 weight_std=weight_std, img_size2D=img_size2D, img_size3D=img_size3D,
                                                 modal_type=modal_type)

        total = sum([param.nelement() for param in self.transformer.parameters()])
        print('  + Number of Transformer Params: %.2f(e6)' % (total / 1e6))

        # Decoder patchEmbed
        if modal_type == '2D':
            # 2D
            self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.DecEmbed2D0 = PatchEmbed2D(img_size=img_size2D // 2, patch_size=2, in_chans=self.transformer.embed_dims[-1],
                                            embed_dim=embed_dims[0])
            self.DecEmbed2D1 = PatchEmbed2D(img_size=img_size2D // 2, patch_size=2, in_chans=embed_dims[0],
                                            embed_dim=embed_dims[1])
            self.DecEmbed2D2 = PatchEmbed2D(img_size=img_size2D // 2, patch_size=2, in_chans=embed_dims[1],
                                            embed_dim=embed_dims[2])

            self.DecPosEmbed2D0 = nn.Parameter(
                torch.zeros(1, self.transformer.patch_embed2D3.num_patches + 1, embed_dims[0]))
            self.DecPosDrop2D0 = nn.Dropout(p=drop_rate)
            self.DecPosEmbed2D1 = nn.Parameter(
                torch.zeros(1, self.transformer.patch_embed2D2.num_patches + 1, embed_dims[1]))
            self.DecPosDrop2D1 = nn.Dropout(p=drop_rate)
            self.DecPosEmbed2D2 = nn.Parameter(
                torch.zeros(1, self.transformer.patch_embed2D1.num_patches + 1, embed_dims[2]))
            self.DecPosDrop2D2 = nn.Dropout(p=drop_rate)

            trunc_normal_(self.DecPosEmbed2D0, std=.02)
            trunc_normal_(self.DecPosEmbed2D1, std=.02)
            trunc_normal_(self.DecPosEmbed2D2, std=.02)

        elif modal_type == '3D':
            # 3D
            self.upsamplex122 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
            self.DecEmbed3D0 = PatchEmbed3D(img_size=[img_size3D[0] // 8, img_size3D[1] // 16, img_size3D[2] // 16],
                                            patch_size=[2, 2, 2], in_chans=self.transformer.embed_dims[-1], embed_dim=embed_dims[0])
            self.DecEmbed3D1 = PatchEmbed3D(img_size=[img_size3D[0] // 4, img_size3D[1] // 8, img_size3D[2] // 8],
                                            patch_size=[2, 2, 2], in_chans=embed_dims[0], embed_dim=embed_dims[1])
            self.DecEmbed3D2 = PatchEmbed3D(img_size=[img_size3D[0] // 2, img_size3D[1] // 4, img_size3D[2] // 4],
                                            patch_size=[2, 2, 2], in_chans=embed_dims[1], embed_dim=embed_dims[2])

            self.DecPosEmbed3D0 = nn.Parameter(
                torch.zeros(1, self.transformer.patch_embed3D3.num_patches + 1, embed_dims[0]))
            self.DecPosDrop3D0 = nn.Dropout(p=drop_rate)
            self.DecPosEmbed3D1 = nn.Parameter(
                torch.zeros(1, self.transformer.patch_embed3D2.num_patches + 1, embed_dims[1]))
            self.DecPosDrop3D1 = nn.Dropout(p=drop_rate)
            self.DecPosEmbed3D2 = nn.Parameter(
                torch.zeros(1, self.transformer.patch_embed3D1.num_patches + 1, embed_dims[2]))
            self.DecPosDrop3D2 = nn.Dropout(p=drop_rate)

            trunc_normal_(self.DecPosEmbed3D0, std=.02)
            trunc_normal_(self.DecPosEmbed3D1, std=.02)
            trunc_normal_(self.DecPosEmbed3D2, std=.02)

        elif modal_type == 'MM':
            # 2D
            self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.DecEmbed2D0 = PatchEmbed2D(img_size=img_size2D // 16, patch_size=2, in_chans=self.transformer.embed_dims[-1],
                                            embed_dim=embed_dims[0])
            self.DecEmbed2D1 = PatchEmbed2D(img_size=img_size2D // 8, patch_size=2, in_chans=embed_dims[0],
                                            embed_dim=embed_dims[1])
            self.DecEmbed2D2 = PatchEmbed2D(img_size=img_size2D // 4, patch_size=2, in_chans=embed_dims[1],
                                            embed_dim=embed_dims[2])

            self.DecPosEmbed2D0 = nn.Parameter(
                torch.zeros(1, self.transformer.patch_embed2D3.num_patches + 1, embed_dims[0]))
            self.DecPosDrop2D0 = nn.Dropout(p=drop_rate)
            self.DecPosEmbed2D1 = nn.Parameter(
                torch.zeros(1, self.transformer.patch_embed2D2.num_patches + 1, embed_dims[1]))
            self.DecPosDrop2D1 = nn.Dropout(p=drop_rate)
            self.DecPosEmbed2D2 = nn.Parameter(
                torch.zeros(1, self.transformer.patch_embed2D1.num_patches + 1, embed_dims[2]))
            self.DecPosDrop2D2 = nn.Dropout(p=drop_rate)

            trunc_normal_(self.DecPosEmbed2D0, std=.02)
            trunc_normal_(self.DecPosEmbed2D1, std=.02)
            trunc_normal_(self.DecPosEmbed2D2, std=.02)

            # 3D
            self.upsamplex122 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
            self.DecEmbed3D0 = PatchEmbed3D(img_size=[img_size3D[0] // 8, img_size3D[1] // 16, img_size3D[2] // 16],
                                            patch_size=[2, 2, 2], in_chans=self.transformer.embed_dims[-1], embed_dim=embed_dims[0])
            self.DecEmbed3D1 = PatchEmbed3D(img_size=[img_size3D[0] // 4, img_size3D[1] // 8, img_size3D[2] // 8],
                                            patch_size=[2, 2, 2], in_chans=embed_dims[0], embed_dim=embed_dims[1])
            self.DecEmbed3D2 = PatchEmbed3D(img_size=[img_size3D[0] // 2, img_size3D[1] // 4, img_size3D[2] // 4],
                                            patch_size=[2, 2, 2], in_chans=embed_dims[1], embed_dim=embed_dims[2])

            self.DecPosEmbed3D0 = nn.Parameter(
                torch.zeros(1, self.transformer.patch_embed3D3.num_patches + 1, embed_dims[0]))
            self.DecPosDrop3D0 = nn.Dropout(p=drop_rate)
            self.DecPosEmbed3D1 = nn.Parameter(
                torch.zeros(1, self.transformer.patch_embed3D2.num_patches + 1, embed_dims[1]))
            self.DecPosDrop3D1 = nn.Dropout(p=drop_rate)
            self.DecPosEmbed3D2 = nn.Parameter(
                torch.zeros(1, self.transformer.patch_embed3D1.num_patches + 1, embed_dims[2]))
            self.DecPosDrop3D2 = nn.Dropout(p=drop_rate)

            trunc_normal_(self.DecPosEmbed3D0, std=.02)
            trunc_normal_(self.DecPosEmbed3D1, std=.02)
            trunc_normal_(self.DecPosEmbed3D2, std=.02)

        # Decoder transformer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.Decblock0 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], modal_type=modal_type)
            for i in range(depths[0])])

        cur += depths[0]
        self.Decblock1 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], modal_type=modal_type)
            for i in range(depths[1])])

        cur += depths[1]
        self.Decblock2 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], modal_type=modal_type)
            for i in range(depths[2])])

        self.norm = norm_layer(embed_dims[2])

        self.cls_tokens0 = nn.Linear(self.transformer.embed_dims[-1], embed_dims[0])
        self.cls_tokens1 = nn.Linear(embed_dims[0], embed_dims[1])
        self.cls_tokens2 = nn.Linear(embed_dims[1], embed_dims[2])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (
        nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm2d,
        nn.InstanceNorm3d)):
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
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def interpolate_pos_encoding3D(self, pos_embed, x, d, w, h):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return pos_embed
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        d0 = d
        w0 = w
        h0 = h
        d0, w0, h0 = d0 + 0.1, w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, d, int(math.sqrt(N/d)), int(math.sqrt(N/d)), dim).permute(0, 4, 1, 2, 3),
            scale_factor=(d0 / d, w0 / math.sqrt(N/d), h0 / math.sqrt(N/d)),
            mode='trilinear',
        )
        assert int(d0) == patch_pos_embed.shape[-3] and int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward2D(self, inputs, modal_type):
        B = inputs.shape[0]

        ####### encoder
        x_encoder, (H, W) = self.transformer(inputs, modal_type)
        cls_tokens = x_encoder[-1][:, 0]
        x = x_encoder[-1][:, 1::].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ####### decoder
        # stage 0
        x, (H, W) = self.DecEmbed2D0(x)
        x = x + x_encoder[-2].flatten(2).transpose(1, 2)
        cls_tokens = self.cls_tokens0(cls_tokens)
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        x = x + self.interpolate_pos_encoding2D(self.DecPosEmbed2D0, x, W, H)
        x = self.DecPosDrop2D0(x)
        for blk in self.Decblock0:
            x = blk(x, (H, W), modal_type)
        cls_tokens = x[:, 0]
        x = x[:, 1::].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 1
        x, (H, W) = self.DecEmbed2D1(x)
        x = x + x_encoder[-3].flatten(2).transpose(1, 2)
        cls_tokens = self.cls_tokens1(cls_tokens)
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        x = x + self.interpolate_pos_encoding2D(self.DecPosEmbed2D1, x, W, H)
        x = self.DecPosDrop2D1(x)
        for blk in self.Decblock1:
            x = blk(x, (H, W), modal_type)
        cls_tokens = x[:, 0]
        x = x[:, 1::].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x, (H, W) = self.DecEmbed2D2(x)
        x = x + x_encoder[-4].flatten(2).transpose(1, 2)
        cls_tokens = self.cls_tokens2(cls_tokens)
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        x = x + self.interpolate_pos_encoding2D(self.DecPosEmbed2D2, x, W, H)
        x = self.DecPosDrop2D2(x)
        for blk in self.Decblock2:
            x = blk(x, (H, W), modal_type)

        x = self.norm(x)

        return x[:, 0]

    def forward3D(self, inputs, modal_type):
        B = inputs.shape[0]

        ####### encoder
        x_encoder, (D, H, W) = self.transformer(inputs, modal_type)
        cls_tokens = x_encoder[-1][:, 0]
        x = x_encoder[-1][:, 1::].reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        ####### decoder
        # stage 0
        x, (D, H, W) = self.DecEmbed3D0(x)
        x = x + x_encoder[-2].flatten(2).transpose(1, 2)
        cls_tokens = self.cls_tokens0(cls_tokens)
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        x = x + self.interpolate_pos_encoding3D(self.DecPosEmbed3D0, x, D, W, H)
        x = self.DecPosDrop3D0(x)
        for blk in self.Decblock0:
            x = blk(x, (D, H, W), modal_type)
        cls_tokens = x[:, 0]
        x = x[:, 1::].reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        # stage 1
        x, (D, H, W) = self.DecEmbed3D1(x)
        x = x + x_encoder[-3].flatten(2).transpose(1, 2)
        cls_tokens = self.cls_tokens1(cls_tokens)
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        x = x + self.interpolate_pos_encoding3D(self.DecPosEmbed3D1, x, D, W, H)
        x = self.DecPosDrop3D1(x)
        for blk in self.Decblock1:
            x = blk(x, (D, H, W), modal_type)
        cls_tokens = x[:, 0]
        x = x[:, 1::].reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        # stage 2
        x, (D, H, W) = self.DecEmbed3D2(x)
        x = x + x_encoder[-4].flatten(2).transpose(1, 2)
        cls_tokens = self.cls_tokens2(cls_tokens)
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        x = x + self.interpolate_pos_encoding3D(self.DecPosEmbed3D2, x, D, W, H)
        x = self.DecPosDrop3D2(x)
        for blk in self.Decblock2:
            x = blk(x, (D, H, W), modal_type)

        x = self.norm(x)

        return x[:, 0]

    def forward(self, inputs):

        if len(inputs.shape) == 4:
            modal_type = '2D'
            return self.forward2D(inputs, modal_type)
        elif len(inputs.shape) == 5:
            modal_type = '3D'
            return self.forward3D(inputs, modal_type)


# tiny
def model_tiny(norm2D='BN2', norm3D='BN3', act='ReLU', ws=False, img_size2D=224, img_size3D=[16, 96, 96],
               modal_type='MM', **kwargs):
    model = MiT(norm_cfg2D=norm2D, norm_cfg3D=norm3D, activation_cfg=act, weight_std=ws, img_size2D=img_size2D,
                       img_size3D=img_size3D,
                       embed_dims=[256, 128, 48], num_heads=[8, 4, 2], depths=[1, 1, 1], mlp_ratios=[4, 4, 4],
                       sr_ratios=[2, 4, 6], modal_type=modal_type, encoder='tiny')

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of Model Params: %.2f(e6)' % (total / 1e6))

    return model



def model_small(norm2D='BN2', norm3D='BN3', act='ReLU', ws=False, img_size2D=224, img_size3D=[16, 96, 96],
                modal_type='MM', **kwargs):
    model = MiT(norm_cfg2D=norm2D, norm_cfg3D=norm3D, activation_cfg=act, weight_std=ws, img_size2D=img_size2D,
                       img_size3D=img_size3D,
                       embed_dims=[256, 128, 48], num_heads=[8, 4, 2], depths=[3, 4, 3], mlp_ratios=[4, 4, 4],
                       sr_ratios=[2, 4, 6], modal_type=modal_type, encoder='small')

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of Model Params: %.2f(e6)' % (total / 1e6))

    return model


class Head(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.mlp(x)
        x = nn.functional.normalize(x0, dim=-1, p=2)
        x = self.last_layer(x)
        return x#, x0

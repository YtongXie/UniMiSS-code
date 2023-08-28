import math
import torch
import torch.nn as nn
from functools import partial
from nets import utils as utils
from scipy import ndimage
import numpy as np



class VisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(self, num_classes=2, norm_cfg3D='BN3', activation_cfg='ReLU', weight_std=False, img_size3D=[16, 96, 96], in_chans=1,
                 embed_dims=[64, 192, 384, 384], num_heads=[2, 4, 8], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 4, 6], sr_ratios=[1, 1, 1]):

        super().__init__()

        self.embed_dims = embed_dims

        self.patch_embed3D0 = utils.Conv3dBlock(in_chans, 32, norm_cfg=norm_cfg3D, activation_cfg=activation_cfg,
                                                   weight_std=weight_std, kernel_size=7, stride=(1, 2, 2),
                                                   padding=3)

        self.patch_embed3D1 = utils.PatchEmbed3D(norm_cfg=norm_cfg3D, activation_cfg=activation_cfg,
                                           weight_std=weight_std,
                                           img_size=[img_size3D[0] // 1, img_size3D[1] // 2, img_size3D[2] // 2],
                                           patch_size=[2, 2, 2], in_chans=32, embed_dim=embed_dims[0])
        self.patch_embed3D2 = utils.PatchEmbed3D(norm_cfg=norm_cfg3D, activation_cfg=activation_cfg,
                                           weight_std=weight_std,
                                           img_size=[img_size3D[0] // 2, img_size3D[1] // 4, img_size3D[2] // 4],
                                           patch_size=[2, 2, 2], in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3D3 = utils.PatchEmbed3D(norm_cfg=norm_cfg3D, activation_cfg=activation_cfg,
                                           weight_std=weight_std,
                                           img_size=[img_size3D[0] // 4, img_size3D[1] // 8, img_size3D[2] // 8],
                                           patch_size=[2, 2, 2], in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed3D4 = utils.PatchEmbed3D(norm_cfg=norm_cfg3D, activation_cfg=activation_cfg,
                                           weight_std=weight_std,
                                           img_size=[img_size3D[0] // 8, img_size3D[1] // 16, img_size3D[2] // 16],
                                           patch_size=[2, 2, 2], in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.pos_embed3D1 = nn.Parameter(torch.zeros(1, self.patch_embed3D1.num_patches + 1, embed_dims[0]))
        self.pos_drop3D1 = nn.Dropout(p=drop_rate)
        self.pos_embed3D2 = nn.Parameter(torch.zeros(1, self.patch_embed3D2.num_patches + 1, embed_dims[1]))
        self.pos_drop3D2 = nn.Dropout(p=drop_rate)
        self.pos_embed3D3 = nn.Parameter(torch.zeros(1, self.patch_embed3D3.num_patches + 1, embed_dims[2]))
        self.pos_drop3D3 = nn.Dropout(p=drop_rate)
        self.pos_embed3D4 = nn.Parameter(torch.zeros(1, self.patch_embed3D4.num_patches + 1, embed_dims[3]))
        self.pos_drop3D4 = nn.Dropout(p=drop_rate)

        utils.trunc_normal_(self.pos_embed3D1, std=.02)
        utils.trunc_normal_(self.pos_embed3D2, std=.02)
        utils.trunc_normal_(self.pos_embed3D3, std=.02)
        utils.trunc_normal_(self.pos_embed3D4, std=.02)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([utils.Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([utils.Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([utils.Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([utils.Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        self.cls_tokens1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_tokens2 = nn.Linear(embed_dims[0], embed_dims[1])
        self.cls_tokens3 = nn.Linear(embed_dims[1], embed_dims[2])
        self.cls_tokens4 = nn.Linear(embed_dims[2], embed_dims[3])
        utils.trunc_normal_(self.cls_tokens1, std=.02)

        # Classifier head
        self.norm_new = norm_layer(embed_dims[3])
        self.head_new = nn.Linear(embed_dims[3], num_classes)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            utils.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        d0, w0, h0 = d0 + 0.1, w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, d, int(math.sqrt(N / d)), int(math.sqrt(N / d)), dim).permute(0, 4, 1, 2, 3),
            scale_factor=(d0 / d, w0 / math.sqrt(N / d), h0 / math.sqrt(N / d)),
            mode='trilinear',
        )
        assert int(d0) == patch_pos_embed.shape[-3] and int(w0) == patch_pos_embed.shape[-2] and int(h0) == \
               patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def forward(self, x):

        B = x.shape[0]
        x = self.patch_embed3D0(x)

        # stage 1
        x, (D, H, W) = self.patch_embed3D1(x)
        cls_tokens = self.cls_tokens1.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding3D(self.pos_embed3D1, x, D, W, H)
        x = self.pos_drop3D1(x)
        for blk in self.block1:
            x = blk(x, (D, H, W))
        cls_tokens = x[:, 0]
        x = x[:, 1::].reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        # stage 2
        x, (D, H, W) = self.patch_embed3D2(x)
        cls_tokens = self.cls_tokens2(cls_tokens)
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        x = x + self.interpolate_pos_encoding3D(self.pos_embed3D2, x, D, W, H)
        x = self.pos_drop3D2(x)
        for blk in self.block2:
            x = blk(x, (D, H, W))
        cls_tokens = x[:, 0]
        x = x[:, 1::].reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        # stage 3
        x, (D, H, W) = self.patch_embed3D3(x)
        cls_tokens = self.cls_tokens3(cls_tokens)
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        x = x + self.interpolate_pos_encoding3D(self.pos_embed3D3, x, D, W, H)
        x = self.pos_drop3D3(x)
        for blk in self.block3:
            x = blk(x, (D, H, W))
        cls_tokens = x[:, 0]
        x = x[:, 1::].reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        # stage 4
        x, (D, H, W) = self.patch_embed3D4(x)
        cls_tokens = self.cls_tokens4(cls_tokens)
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        x = x + self.interpolate_pos_encoding3D(self.pos_embed3D4, x, D, W, H)
        x = self.pos_drop3D4(x)
        for blk in self.block4:
            x = blk(x, (D, H, W))

        out_cls0 = self.head_new(self.norm_new(x)[:, 0])
        out_cls1 = self.head_new(self.avg_pool(self.norm_new(x)[:, 1::].transpose(1, 2))[:, :, 0])

        return (out_cls0 + out_cls1)/2.



def model_tiny(norm_cfg3D='BN3', activation_cfg='ReLU', weight_std=False, img_size3D=[16, 96, 96], num_classes=3, pretrain=False, pretrain_path=None, **kwargs):
    model = VisionTransformer(norm_cfg3D=norm_cfg3D, activation_cfg=activation_cfg,
                              weight_std=weight_std, num_classes=num_classes, img_size3D=img_size3D,
                              embed_dims=[48, 128, 256, 512], depths=[1, 1, 1, 1],
                              num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                              sr_ratios=[6, 4, 2, 1], qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of Transformer Params: %.2f(e6)' % (total / 1e6))

    if pretrain:
        pre_type = 'student'  # teacher student
        print('*********loading from checkpoint ssl: {}'.format(pretrain_path))

        if pre_type == 'teacher':
            pre_dict_ori = torch.load(pretrain_path, map_location="cpu")[pre_type]
            pre_dict_ori = {k.replace("backbone.", ""): v for k, v in pre_dict_ori.items()}
            print('Teacher: length of pre-trained layers: %.f' % (len(pre_dict_ori)))
        else:
            pre_dict_ori = torch.load(pretrain_path, map_location="cpu")[pre_type]
            pre_dict_ori = {k.replace("module.backbone.", ""): v for k, v in pre_dict_ori.items()}
            print('Student: length of pre-trained layers: %.f' % (len(pre_dict_ori)))

        pre_dict_transformer = {k: v for k, v in pre_dict_ori.items() if 'transformer' in k}
        pre_dict_transformer = {k.replace("transformer.", ""): v for k, v in pre_dict_transformer.items()}

        model_dict = model.state_dict()
        print('length of new layers: %.f' % (len(model_dict)))
        print('before loading weights: %.16f' % (model.state_dict()['block1.0.mlp.fc1.weight'].mean()))

        # Patch_embeddings
        print('Patch_embeddings layer1 weights: %s' % (model.state_dict()['patch_embed3D1.proj.conv.weight'].mean()))
        print('Patch_embeddings layer2 weights: %s' % (model.state_dict()['patch_embed3D2.proj.conv.weight'].mean()))
        # Position_embeddings
        print('Position_embeddings weights: %s' % (model.pos_embed3D1.data.mean()))

        for k, v in pre_dict_transformer.items():
            if 'pos_embed3D' in k:
                posemb = pre_dict_transformer[k]
                posemb_new = model_dict[k]

                if posemb.size() == posemb_new.size():
                    print(k + 'layer is matched')
                    pre_dict_transformer[k] = posemb
                else:
                    ntok_new = posemb_new.size(1)
                    posemb_zoom = ndimage.zoom(posemb[0], (ntok_new / posemb.size(1), 1), order=1)
                    posemb_zoom = np.expand_dims(posemb_zoom, 0)
                    pre_dict_transformer[k] = torch.from_numpy(posemb_zoom)

        pre_dict = {k: v for k, v in pre_dict_transformer.items() if k in model_dict}
        print('length of matched layers: %.f' % (len(pre_dict)))

        # Update weigts
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)
        print('after loading weights: %.16f' % (model.state_dict()['block1.0.mlp.fc1.weight'].mean()))
        print('Patch_embeddings layer1 pretrained weights: %s' % (model.state_dict()['patch_embed3D1.proj.conv.weight'].mean()))
        print('Patch_embeddings layer2 pretrained weights: %s' % (model.state_dict()['patch_embed3D2.proj.conv.weight'].mean()))
        print('Position_embeddings pretrained weights: %.12f' % (model.pos_embed3D1.data.mean()))


    else:
        print('length of new layers: %.f' % (len(model.state_dict())))
        print('before loading weights: %.16f' % (model.state_dict()['block1.0.mlp.fc1.weight'].mean()))
        # Patch_embeddings
        print('Patch_embeddings layer1 weights: %s' % (model.state_dict()['patch_embed3D1.proj.conv.weight'].mean()))
        print('Patch_embeddings layer2 weights: %s' % (model.state_dict()['patch_embed3D2.proj.conv.weight'].mean()))
        # Position_embeddings
        print('Position_embeddings weights: %s' % (model.pos_embed3D1.data.mean()))

    return model


def model_small(norm_cfg3D='BN3', activation_cfg='ReLU', weight_std=False, img_size3D=[16, 96, 96], num_classes=3, pretrain=False, pretrain_path=None, **kwargs):
    model = VisionTransformer(norm_cfg3D=norm_cfg3D, activation_cfg=activation_cfg,
                              weight_std=weight_std, num_classes=num_classes, img_size3D=img_size3D,
                              embed_dims=[48, 128, 256, 512], depths=[2, 3, 4, 3],
                              num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                              sr_ratios=[6, 4, 2, 1], qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of Transformer Params: %.2f(e6)' % (total / 1e6))

    if pretrain:
        pre_type = 'student'  # teacher student
        print('*********loading from checkpoint ssl: {}'.format(pretrain_path))

        if pre_type == 'teacher':
            pre_dict_ori = torch.load(pretrain_path, map_location="cpu")[pre_type]
            pre_dict_ori = {k.replace("backbone.", ""): v for k, v in pre_dict_ori.items()}
            print('Teacher: length of pre-trained layers: %.f' % (len(pre_dict_ori)))
        else:
            pre_dict_ori = torch.load(pretrain_path, map_location="cpu")[pre_type]
            pre_dict_ori = {k.replace("module.backbone.", ""): v for k, v in pre_dict_ori.items()}
            print('Student: length of pre-trained layers: %.f' % (len(pre_dict_ori)))

        pre_dict_transformer = {k: v for k, v in pre_dict_ori.items() if 'transformer' in k}
        pre_dict_transformer = {k.replace("transformer.", ""): v for k, v in pre_dict_transformer.items()}

        model_dict = model.state_dict()
        print('length of new layers: %.f' % (len(model_dict)))
        print('before loading weights: %.16f' % (model.state_dict()['block1.0.mlp.fc1.weight'].mean()))

        # Patch_embeddings
        print('Patch_embeddings layer1 weights: %s' % (model.state_dict()['patch_embed3D1.proj.conv.weight'].mean()))
        print('Patch_embeddings layer2 weights: %s' % (model.state_dict()['patch_embed3D2.proj.conv.weight'].mean()))
        # Position_embeddings
        print('Position_embeddings weights: %s' % (model.pos_embed3D1.data.mean()))

        for k, v in pre_dict_transformer.items():
            if 'pos_embed3D' in k:
                posemb = pre_dict_transformer[k]
                posemb_new = model_dict[k]

                if posemb.size() == posemb_new.size():
                    print(k + 'layer is matched')
                    pre_dict_transformer[k] = posemb
                else:
                    ntok_new = posemb_new.size(1)
                    posemb_zoom = ndimage.zoom(posemb[0], (ntok_new / posemb.size(1), 1), order=1)
                    posemb_zoom = np.expand_dims(posemb_zoom, 0)
                    pre_dict_transformer[k] = torch.from_numpy(posemb_zoom)

        pre_dict = {k: v for k, v in pre_dict_transformer.items() if k in model_dict}
        print('length of matched layers: %.f' % (len(pre_dict)))

        # Update weigts
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)
        print('after loading weights: %.16f' % (model.state_dict()['block1.0.mlp.fc1.weight'].mean()))
        print('Patch_embeddings layer1 pretrained weights: %s' % (model.state_dict()['patch_embed3D1.proj.conv.weight'].mean()))
        print('Patch_embeddings layer2 pretrained weights: %s' % (model.state_dict()['patch_embed3D2.proj.conv.weight'].mean()))
        print('Position_embeddings pretrained weights: %.12f' % (model.pos_embed3D1.data.mean()))


    else:
        print('length of new layers: %.f' % (len(model.state_dict())))
        print('before loading weights: %.16f' % (model.state_dict()['block1.0.mlp.fc1.weight'].mean()))
        # Patch_embeddings
        print('Patch_embeddings layer1 weights: %s' % (model.state_dict()['patch_embed3D1.proj.conv.weight'].mean()))
        print('Patch_embeddings layer2 weights: %s' % (model.state_dict()['patch_embed3D2.proj.conv.weight'].mean()))
        # Position_embeddings
        print('Position_embeddings weights: %s' % (model.pos_embed3D1.data.mean()))

    return model

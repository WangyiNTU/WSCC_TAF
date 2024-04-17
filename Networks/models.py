# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from .swin_transformer import SwinTransformer
from efficientnet_pytorch import EfficientNet
from .vision_mamba import VisionMamba

class VisionTransformer_token(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 1)
        )
        self.output1.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        x = self.output1(x)

        return x




class VisionTransformer_gap(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6912 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.output1.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = x[:, 1:]

        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        x = F.adaptive_avg_pool1d(x, (48))
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.output1(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class Res_SE1D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(Res_SE1D, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer1D(planes, reduction)
        self.downsample = downsample
        if inplanes != planes or stride > 1:
            self.downsample = nn.Sequential(nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm1d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class VisionTransformer_cls(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6912 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 12)
        )
        # self.output1 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(6912 * 4, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(128, 12)
        # )
        self.output1.apply(self._init_weights)

        self.output2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.output2.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x_cls = x[:, 1:]
        x_off = x[:, 0]


        return x_cls, x_off

    def forward(self, x):
        x_cls_1, x_offset_1 = self.forward_features(x)
        x_cls = F.adaptive_avg_pool1d(x_cls_1, (48))
        x_cls = torch.flatten(x_cls, 1)
        x_cls = self.output1(x_cls) #* F.sigmoid(x_cls)

        x_global = F.adaptive_avg_pool1d(x_cls_1.transpose(1,2), (1))
        x_offset = torch.flatten(x_offset_1.unsqueeze(1) - x_global.transpose(1,2), 1)
        x_offset = self.output2(x_offset)
        # x = self.head(x)
        return [x_offset, x_cls]

class efficientnet_b7(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backend = EfficientNet.from_pretrained('efficientnet-b7')

        hidden_size = 2560 # for efficientnet-b7
        self.pooling_size = (1,1)
        self.output1 = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.output1.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.backend.extract_features(x)

        return x

    def forward(self, x):
        x = self.forward_features(x) # B, 2560, 12, 12
        x = F.adaptive_avg_pool2d(x, (self.pooling_size)) # B, 2560, 1
        x = x.view(x.shape[0], -1)
        x = self.output1(x)

        return x

class VisionTransformer_fgap(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        hidden_size = self.embed_dim # D: 768

        self.pooling_size = 1
        self.output1 = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.output1.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = x[:, 1:]

        return x

    def forward(self, x):
        x = self.forward_features(x) # B, 576, 768
        x = x.transpose(1,2) # B, 768, 576
        x = F.adaptive_avg_pool1d(x, (self.pooling_size)) # B, 768, 1
        x = x.view(x.shape[0], -1)
        x = self.output1(x)

        return x

class VisionTransformer_attention(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        hidden_size = self.embed_dim # D: 768

        self.post_conv = nn.Sequential(
            Res_SE1D(hidden_size, hidden_size, stride=1), # B, 768, 576
        )
        self.post_conv.apply(self._init_weights)

        self.pooling_size = 1
        self.output1 = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.output1.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = x[:, 1:]

        return x

    def forward(self, x):
        x = self.forward_features(x) # B, 576, 768
        x = x.transpose(1,2) # B, 768, 576
        x = self.post_conv(x) # B, 768, 576
        x = F.adaptive_avg_pool1d(x, (self.pooling_size)) # B, 768, 1
        x = x.view(x.shape[0], -1)
        x = self.output1(x)

        return x

@register_model
def base_patch16_384_token(pretrained=False, **kwargs):
    model = VisionTransformer_token(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        checkpoint = torch.load(
            './Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model


@register_model
def base_patch16_384_gap(pretrained=False, **kwargs):
    model = VisionTransformer_gap(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        checkpoint = torch.load(
            './Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model

@register_model
def base_patch16_384_effnet(pretrained=False, **kwargs):
    model = efficientnet_b7(**kwargs)
    if pretrained:
        pass # always pretrain is not useful
    return model

@register_model
def base_patch16_384_fgap(pretrained=False, **kwargs):
    model = VisionTransformer_fgap(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        checkpoint = torch.load(
            './Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model

@register_model
def base_patch16_384_attention(pretrained=False, **kwargs):
    model = VisionTransformer_attention(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        checkpoint = torch.load(
            './Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model

@register_model
def base_patch16_384_cls(pretrained=False, **kwargs):
    model = VisionTransformer_cls(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        checkpoint = torch.load(
            './Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model

@register_model
def base_patch16_384_swin(pretrained=False, mode = 0, **kwargs):
   # base SwinTransformer
    model = SwinTransformer(img_size=384,
                            patch_size=4,
                            in_chans=3,
                            num_classes=1,
                            embed_dim=128,
                            depths=[ 2, 2, 18, 2 ],
                            num_heads=[ 4, 8, 16, 32 ],
                            window_size=12,
                            mlp_ratio=4.0,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            drop_path_rate=0.2,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=False,
                            mode=mode)

    # Tiny SwinTransformer
    # model = SwinTransformer(img_size=224,
    #                         patch_size=4,
    #                         in_chans=3,
    #                         num_classes=1,
    #                         embed_dim=96,
    #                         depths=[2, 2, 6, 2],
    #                         num_heads=[3, 6, 12, 24],
    #                         window_size=7,
    #                         mlp_ratio=4.0,
    #                         qkv_bias=True,
    #                         qk_scale=None,
    #                         drop_rate=0.0,
    #                         drop_path_rate=0.2,
    #                         ape=False,
    #                         patch_norm=True,
    #                         use_checkpoint=False)

    if pretrained:
        checkpoint = torch.load('Networks/swin_base_patch4_window12_384.pth', map_location='cpu')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model


@register_model
def base_patch16_384_mamba(pretrained=False, mode = 0, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True,mode=mode, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load('Networks/swin_base_patch4_window12_384.pth', map_location='cpu')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load mamba pretrained")

    
    return model

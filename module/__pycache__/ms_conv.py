import torch
import torch.nn as nn
from timm.models.layers import DropPath
# from Nspikingjelly.clock_driven.neuronv1 import (
#     MultiStepLIFNode,
# )
from Nspikingjelly.clock_driven.neuronv1 import  MultiStepLIFNode

# class Erode(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.pool = nn.MaxPool3d(
#             kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
#         )

#     def forward(self, x):
#         return self.pool(x)


class MS_MLP_Conv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        spike_mode="lif",
        layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")

        self.fc2_conv = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x):
        identity = x
        #x = x.unsqueeze(1)
        #T, C, H, W = x.shape
        
        #x = x.reshape(4, 512, 14, 14)
        x = self.fc1_lif(x)
        out1_w0, out1_w1, out1_w2, out1_w3= x[0].unsqueeze(0), x[1].unsqueeze(0), x[2].unsqueeze(0), x[3].unsqueeze(0)
        out1_temp = self.fc1_conv(out1_w0)
        out1_w0 = self.fc1_bn(out1_temp)
        out1_temp=self.fc1_conv(out1_w1)
        out1_w1 = self.fc1_bn(out1_temp)
        out1_temp =self.fc1_conv(out1_w2)
        out1_w2 = self.fc1_bn(out1_temp)
        out1_temp=self.fc1_conv(out1_w3)
        out1_w3 = self.fc1_bn(out1_temp)
        out1 = torch.cat([out1_w0, out1_w1, out1_w2, out1_w3], dim=0)
        # x = self.fc1_conv(x.flatten(0, 1))
        # x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        x = self.fc2_lif(out1)
        #x = x.resahpe(4, 512, 14, 14)
        out1_w0, out1_w1, out1_w2, out1_w3= x[0].unsqueeze(0), x[1].unsqueeze(0), x[2].unsqueeze(0), x[3].unsqueeze(0)
        out1_temp = self.fc2_conv(out1_w0)
        out1_w0 = self.fc2_bn(out1_temp)
        out1_temp=self.fc2_conv(out1_w1)
        out1_w1 = self.fc2_bn(out1_temp)
        out1_temp =self.fc2_conv(out1_w2)
        out1_w2 = self.fc2_bn(out1_temp)
        out1_temp=self.fc2_conv(out1_w3)
        out1_w3 = self.fc2_bn(out1_temp)
        out1 = torch.cat([out1_w0, out1_w1, out1_w2, out1_w3], dim=0)
        # x = self.fc2_conv(x.flatten(0, 1))
        # x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        #x = out1
        x = out1 + identity
        return x


class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.attn_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
            )
        self.talking_heads = nn.Conv1d(
            num_heads, num_heads, kernel_size=1, stride=1, bias=False
        )
        self.talking_heads_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
            )
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)
        self.shortcut_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="torch"
            )
        self.mode = mode
        self.layer = layer

    def forward(self, x):
        identity = x
        #x = x.unsqueeze(1)
        #print(x.shape)
        # T, C, H, W = x.shape
        # N = H * W
        x = x.reshape(4, 512, 14, 14)
        x = self.shortcut_lif(x)
        out1_w0, out1_w1, out1_w2, out1_w3= x[0].unsqueeze(0), x[1].unsqueeze(0), x[2].unsqueeze(0), x[3].unsqueeze(0)
        out1_temp = self.q_conv(out1_w0)
        out1_w0 = self.q_bn(out1_temp)
        out1_temp=self.q_conv(out1_w1)
        out1_w1 = self.q_bn(out1_temp)
        out1_temp =self.q_conv(out1_w2)
        out1_w2 = self.q_bn(out1_temp)
        out1_temp=self.q_conv(out1_w3)
        out1_w3 = self.q_bn(out1_temp)
        q_conv_out = torch.cat([out1_w0, out1_w1, out1_w2, out1_w3], dim=0)
        #x_for_qkv = x.flatten(0, 1)
        #q_conv_out = q_conv_out.unsqueeze(1).reshape(T, B, C, H, W).contiguous()
        # q_conv_out = self.q_conv(x_for_qkv)
        # q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q_conv_out = q_conv_out.unsqueeze(1)
        q_conv_out = q_conv_out.reshape(4, 1, 512, 196)
        q_conv_out = q_conv_out.transpose(-1, -2)
        q_conv_out = q_conv_out.reshape(4, 196, 8, 64)
        q = q_conv_out.permute(0, 2, 1, 3) # 4, 196, 8, 64
        # q = (
        #     q_conv_out.flatten(3)
        #     .transpose(-1, -2)
        #     .reshape(T, B, N, self.num_heads, dim_q)
        #     .permute(0, 1, 3, 2, 4)
        #     .contiguous()
        # )
        out1_w0, out1_w1, out1_w2, out1_w3= x[0].unsqueeze(0), x[1].unsqueeze(0), x[2].unsqueeze(0), x[3].unsqueeze(0)
        out1_temp = self.k_conv(out1_w0)
        out1_w0 = self.k_bn(out1_temp)
        out1_temp=self.k_conv(out1_w1)
        out1_w1 = self.k_bn(out1_temp)
        out1_temp =self.k_conv(out1_w2)
        out1_w2 = self.k_bn(out1_temp)
        out1_temp=self.k_conv(out1_w3)
        out1_w3 = self.k_bn(out1_temp)
        k_conv_out = torch.cat([out1_w0, out1_w1, out1_w2, out1_w3], dim=0)
        # k_conv_out = self.k_conv(x_for_qkv)
        # k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        #k_conv_out = self.k_lif(k_conv_out)
        k_conv_out = self.k_lif(k_conv_out)
        k_conv_out = k_conv_out.unsqueeze(1)
        k_conv_out = k_conv_out.reshape(4, 1, 512, 196)
        k_conv_out = k_conv_out.transpose(-1, -2)
        k_conv_out = k_conv_out.reshape(4, 196, 8, 64)
        k = k_conv_out.permute(0, 2, 1, 3) # 4, 196, 8, 64
        # k = (
        #     k_conv_out.flatten(3)
        #     .transpose(-1, -2)
        #     .reshape(T, B, N, self.num_heads, torch.div(C,self.num_heads, rounding_mode='trunc').item())
        #     .permute(0, 1, 3, 2, 4)
        #     .contiguous()
        # )
        out1_w0, out1_w1, out1_w2, out1_w3= x[0].unsqueeze(0), x[1].unsqueeze(0), x[2].unsqueeze(0), x[3].unsqueeze(0)
        out1_temp = self.v_conv(out1_w0)
        out1_w0 = self.v_bn(out1_temp)
        out1_temp=self.v_conv(out1_w1)
        out1_w1 = self.v_bn(out1_temp)
        out1_temp =self.v_conv(out1_w2)
        out1_w2 = self.v_bn(out1_temp)
        out1_temp=self.v_conv(out1_w3)
        out1_w3 = self.v_bn(out1_temp)
        v_conv_out = torch.cat([out1_w0, out1_w1, out1_w2, out1_w3], dim=0)
        # v_conv_out = self.v_conv(x_for_qkv)
        # v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v_conv_out = v_conv_out.unsqueeze(1)
        v_conv_out = v_conv_out.reshape(4, 1, 512, 196)
        v_conv_out = v_conv_out.transpose(-1, -2)
        v_conv_out = v_conv_out.reshape(4, 196, 8, 64)
        v = v_conv_out.permute(0, 2, 1, 3) # 4, 196, 8, 64
        #v_conv_out = v_conv_out
        # v = (
        #     v_conv_out.flatten(3)
        #     .transpose(-1, -2)
        #     .reshape(T, B, N, self.num_heads, torch.div(C,self.num_heads, rounding_mode='trunc').item())
        #     .permute(0, 1, 3, 2, 4)
        #     .contiguous()
        # )  # T B head N C//h

        kv = k.mul(v)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        x = q.mul(kv) # 4,8,196,64
        #x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = x.transpose(2, 3).reshape(4, 512, 14, 14)
        out1_w0, out1_w1, out1_w2, out1_w3= x[0].unsqueeze(0), x[1].unsqueeze(0), x[2].unsqueeze(0), x[3].unsqueeze(0)
        out1_temp = self.proj_conv(out1_w0)
        out1_w0 = self.proj_bn(out1_temp)
        out1_temp=self.proj_conv(out1_w1)
        out1_w1 = self.proj_bn(out1_temp)
        out1_temp =self.proj_conv(out1_w2)
        out1_w2 = self.proj_bn(out1_temp)
        out1_temp=self.proj_conv(out1_w3)
        out1_w3 = self.proj_bn(out1_temp)
        out1 = torch.cat([out1_w0, out1_w1, out1_w2, out1_w3], dim=0)
        # x = (
        #     self.proj_bn(self.proj_conv(x.flatten(0, 1)))
        #     .reshape(T, B, C, H, W)
        #     .contiguous()
        # )
        #x = out1
        x = out1 + identity # 4，512，14，14
        return x


class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            mode=attn_mode,
            spike_mode=spike_mode,
            dvs=dvs,
            layer=layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            spike_mode=spike_mode,
            layer=layer,
        )

    def forward(self, x):
        x_attn = self.attn(x)
        x = self.mlp(x_attn)
        return x

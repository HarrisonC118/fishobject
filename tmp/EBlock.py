import torch
from torch import nn as nn
__all__ = ['EBlock', 'TokenWiseEBlock']


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))

        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class FreMLP(nn.Module):

    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''

    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation,
                      stride=1, groups=self.dw_channel,
                      bias=True, dilation=dilation)  # the dconv
        )

    def forward(self, input):
        return self.branch(input)


class EBlock(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        # we define the 2 branches
        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True,
                                    dilation=1) if extra_depth_wise else nn.Identity()  # optional extra dw
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        # second step

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc=c, expand=2)
        self.gamma = nn.Parameter(torch.zeros(
            (1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # 第一步：空间域特征提取
        # 1. 层归一化
        x = self.norm1(inp)

        # 2. 可选的深度卷积
        x = self.conv1(self.extra_conv(x))  # 1x1卷积扩展通道数（DW_Expand倍）

        # 3. 多分支空洞卷积
        z = 0
        for branch in self.branches:  # 不同膨胀率的并行分支
            z += branch(x)  # 分支结果相加

        # 4. 门控机制
        z = self.sg1(z)

        # 5. 简化通道注意力SCA
        x = self.sca(z) * z  # 全局平均池化+1x1卷积生成通道权重

        # 6. 通道压缩回原始维度
        x = self.conv3(x)  # 1x1卷积降维

        # 7. 残差连接
        y = inp + self.beta * x  # 可学习的β参数控制残差强度

        # 第二步：频域特征增强
        # 8. 二次归一化
        x_step2 = self.norm2(y)

        # 9. 频域MLP处理
        x_freq = self.freq(x_step2)
        # 核心操作：
        # 1) FFT变换到频域
        # 2) 仅调整振幅（process1 MLP）
        # 3) 保留原始相位
        # 4) IFFT返回空间域

        # 10. 频域-空间域特征融合
        x = y * x_freq  # 频域增强特征作为注意力权重

        # 11. 最终残差连接
        x = y + x * self.gamma  # 可学习的γ参数控制融合强度
        return x


class TokenWiseGate(nn.Module):
    """空间级门控机制，为每个空间位置生成独立的门控权重"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 使用1x1卷积生成空间门控权重
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x1, x2 = x.chunk(2, dim=1)  # 在通道维度分成两部分

        # 生成空间级门控权重：为每个空间位置(H,W)生成独立的门控
        gate_weights = self.gate_conv(x1)  # [B, C//2, H, W]

        # 应用空间级门控：每个token都有自己的门控权重
        gated_output = x2 * gate_weights  # [B, C//2, H, W]

        return gated_output


class TokenWiseEBlock(nn.Module):
    """
    升级版EBlock，使用Token-wise（空间级）门控机制
    用于替换Backbone-FPN流程中的S5编码器
    """

    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        # 基础参数设置
        self.dw_channel = DW_Expand * c

        # 第一阶段：空间域特征提取
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True,
                                    dilation=1) if extra_depth_wise else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)

        # 多分支空洞卷积
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=dilation))

        # Token-wise门控机制（替换原来的SimpleGate）
        self.token_gate = TokenWiseGate(self.dw_channel)

        # 简化通道注意力SCA
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1, groups=1, bias=True, dilation=1),
        )

        # 通道压缩
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)

        # 第二阶段：频域特征增强
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc=c, expand=2)

        # 可学习的残差权重参数
        self.gamma = nn.Parameter(torch.zeros(
            (1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # 第一步：空间域特征提取
        # 1. 层归一化
        x = self.norm1(inp)

        # 2. 可选的深度卷积 + 1x1卷积扩展通道数
        x = self.conv1(self.extra_conv(x))

        # 3. 多分支空洞卷积
        z = 0
        for branch in self.branches:
            z += branch(x)

        # 4. Token-wise门控机制（核心升级点）
        z = self.token_gate(z)  # 空间级门控，每个token独立控制

        # 5. 简化通道注意力SCA
        x = self.sca(z) * z

        # 6. 通道压缩回原始维度
        x = self.conv3(x)

        # 7. 残差连接
        y = inp + self.beta * x

        # 第二步：频域特征增强
        # 8. 二次归一化
        x_step2 = self.norm2(y)

        # 9. 频域MLP处理
        x_freq = self.freq(x_step2)

        # 10. 频域-空间域特征融合
        x = y * x_freq

        # 11. 最终残差连接
        x = y + x * self.gamma

        return x


if __name__ == '__main__':
    # 测试原始EBlock
    block = EBlock(c=8).to('cuda')
    input = torch.rand(1, 8, 4, 4).to('cuda')
    output = block(input)
    print("Original EBlock:")
    print(f"Input size: {input.size()}")
    print(f"Output size: {output.size()}")

    # 测试TokenWiseEBlock
    token_block = TokenWiseEBlock(c=8).to('cuda')
    token_output = token_block(input)
    print("\nTokenWiseEBlock:")
    print(f"Input size: {input.size()}")
    print(f"Output size: {token_output.size()}")

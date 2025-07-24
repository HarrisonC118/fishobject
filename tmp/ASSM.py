import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat

__all__ = ['ASSM']
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


"""《MambaIRv2: Attentive State Space Restoration》CVPR 2025
基于Mamba的图像修复骨干网络最近在平衡全局感知和计算效率方面展现了显著潜力。
然而，Mamba固有的因果建模限制——即每个标记仅依赖于扫描序列中的前驱标记——限制了图像中像素的充分利用，从而为图像修复带来了新的挑战。
在本研究中，我们提出了MambaIRv2，它赋予Mamba类似于视觉Transformer（ViTs）的非因果建模能力，以达到注意力状态空间修复模型。
具体而言，所提出的注意力状态空间方程允许关注超出扫描序列的范围，并通过单次扫描促进图像展开。此外，我们进一步引入了语义引导的邻近机制，以鼓励远距离但相似的像素之间的交互。
大量实验表明，我们的MambaIRv2在轻量级超分辨率（SR）任务中，即使参数减少了9.3%，其PSNR仍比SRFormer高出0.35dB，并在经典SR任务中以高达0.29dB的优势超越了HAT。
"""


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        B, L, C = x.shape
        K = 1  # mambairV2 needs noly 1 scan
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        #  our ASE here ---
        Cs = Cs.float().view(B, K, -1, L) + prompt  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y


class ASSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)  # 投影回原始维度

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        # ---- 三级 Prompt 配置 ----
        self.Tg, self.Tl, self.Ti = 32, 16, 8

        # Global / Layer 可学习 Prompt 库
        self.Mg = nn.Parameter(torch.randn(self.Tg, self.inner_rank))
        self.Ml = nn.Parameter(torch.randn(self.Tl, self.inner_rank))

        # 路由头
        self.route_g = nn.Linear(self.dim, self.Tg)   # soft
        self.route_l = nn.Linear(self.dim, self.Tl)   # soft

    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size

        # -------------- 3-level Prompt-Pool --------------
        # (1) Global 路由（软）
        logits_g = self.route_g(x)                       # (B,n,Tg)
        Rg = F.gumbel_softmax(logits_g, tau=0.7, hard=False, dim=-1)

        # (2) Layer 路由（软）
        logits_l = self.route_l(x)                       # (B,n,Tl)
        Rl = F.gumbel_softmax(logits_l, tau=0.7, hard=False, dim=-1)

        # (3) Instance Prompt：每个 batch 随机 8 个向量
        M_i = torch.randn(self.Ti, self.inner_rank, device=x.device)   # (8,r)
        idx_i = torch.randint(0, self.Ti, (B, n), device=x.device)     # (B,n)
        Ri    = F.one_hot(idx_i, num_classes=self.Ti).float()          # (B,n,Ti)

        # 汇总 Prompt（r 维）
        prompt_r = (
              torch.einsum('bnt,tr->bnr', Rg, self.Mg)     # Global
            + torch.einsum('bnt,tr->bnr', Rl, self.Ml)     # Layer
            + torch.einsum('bnt,tr->bnr', Ri, M_i)         # Instance
        )                                                  # (B,n,r)

        # 降到 d_state 维： token.weight shape = [inner_rank, d_state]
        prompt = torch.matmul(prompt_r, token.weight)      # (B,n,d_state)
        # -----------------------------------------------

        # 计算 detached_index 和排序索引
        detached_index = torch.argmax(Rl.detach(), dim=-1, keepdim=False).view(B, n)  # [B, n]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        # 将输入投影到隐藏维度
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)  # [B, hidden, H, W]
        x = x * torch.sigmoid(self.CPE(x))  # [B, hidden, H, W]
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # [B, n, hidden]

        # 使用 SGN-unfold 和 selective scan
        semantic_x = semantic_neighbor(x, x_sort_indices)  # SGN-unfold
        y = self.selectiveScan(semantic_x, prompt)  # [B, n, hidden]
        y = self.out_proj(self.out_norm(y))  # [B, n, dim]

        # 使用 SGN-fold 恢复原始顺序
        x = semantic_neighbor(y, x_sort_indices_reverse)  # [B, n, dim]

        # 返回与输入形状相同的输出
        return x


class ModifiedASSM(nn.Module):
    """
    简化版本的ASSM，专门用于RT-DETR中的小目标低置信度token处理
    使用简化的Selective_Scan，隐藏层维度降低为原始的一半
    """
    def __init__(self, dim, d_state, num_tokens=64, inner_rank=128, mlp_ratio=1.):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # 简化的Mamba params - 隐藏层维度降低为原始的一半
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand * 0.5)  # 降低为原始的一半
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim, bias=True)  # 投影回原始维度

        self.in_proj = nn.Linear(self.dim, hidden)

        # ---- 三级 Prompt 配置 ----
        self.Tg, self.Tl, self.Ti = 32, 16, 8

        # Global / Layer 可学习 Prompt 库
        self.Mg = nn.Parameter(torch.randn(self.Tg, self.inner_rank))
        self.Ml = nn.Parameter(torch.randn(self.Tl, self.inner_rank))

        # 路由头
        self.route_g = nn.Linear(self.dim, self.Tg)   # soft
        self.route_l = nn.Linear(self.dim, self.Tl)   # soft

    def forward(self, x, token):
        """
        Args:
            x: [B, num_queries, dim] - memory特征
            token: token embedding用作query输入
        Returns:
            x: [B, num_queries, dim] - 处理后的特征
        """
        B, n, C = x.shape

        # -------------- 3-level Prompt-Pool --------------
        # (1) Global 路由（软）
        logits_g = self.route_g(x)                       # (B,n,Tg)
        Rg = F.gumbel_softmax(logits_g, tau=0.7, hard=False, dim=-1)

        # (2) Layer 路由（软）
        logits_l = self.route_l(x)                       # (B,n,Tl)
        Rl = F.gumbel_softmax(logits_l, tau=0.7, hard=False, dim=-1)

        # (3) Instance Prompt：每个 batch 随机 8 个向量
        M_i = torch.randn(self.Ti, self.inner_rank, device=x.device)   # (8,r)
        idx_i = torch.randint(0, self.Ti, (B, n), device=x.device)     # (B,n)
        Ri    = F.one_hot(idx_i, num_classes=self.Ti).float()          # (B,n,Ti)

        # 汇总 Prompt（r 维）
        prompt_r = (
              torch.einsum('bnt,tr->bnr', Rg, self.Mg)     # Global
            + torch.einsum('bnt,tr->bnr', Rl, self.Ml)     # Layer
            + torch.einsum('bnt,tr->bnr', Ri, M_i)         # Instance
        )                                                  # (B,n,r)

        # 降到 d_state 维： token.weight shape = [inner_rank, d_state]
        prompt = torch.matmul(prompt_r, token.weight)      # (B,n,d_state)
        # -----------------------------------------------

        # 计算 detached_index 和排序索引
        detached_index = torch.argmax(Rl.detach(), dim=-1, keepdim=False).view(B, n)  # [B, n]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        # 将输入投影到隐藏维度
        x = self.in_proj(x)  # [B, n, hidden]

        # 使用 SGN-unfold 和 selective scan
        semantic_x = semantic_neighbor(x, x_sort_indices)  # SGN-unfold
        y = self.selectiveScan(semantic_x, prompt)  # [B, n, hidden]
        y = self.out_proj(self.out_norm(y))  # [B, n, dim]

        # 使用 SGN-fold 恢复原始顺序
        x = semantic_neighbor(y, x_sort_indices_reverse)  # [B, n, dim]

        # 返回与输入形状相同的输出
        return x

if __name__ == '__main__':
    # Define the input dimensions
    dim = 64  # 输入维度
    d_state = 32  # 状态维度
    input_resolution = (16, 16)
    num_tokens = 64  # token数量
    inner_rank = 128  # 内部rank

    # Create the model
    block = ASSM(
        dim=dim,
        d_state=d_state,
        input_resolution=input_resolution,
        num_tokens=num_tokens,
        inner_rank=inner_rank
    ).to('cuda')

    # 修改 token embedding 的维度为 (inner_rank, d_state)
    token = nn.Embedding(inner_rank, d_state).to('cuda')

    # Create random input tensor
    B = 2  # Batch size
    H, W = input_resolution
    input = torch.rand(B, H * W, dim).to('cuda')  # Input shape: [B, H*W, dim]

    # Forward pass
    output = block(input, input_resolution, token)

    # Print shapes for debugging
    print("Input size:", input.size())
    print("Token embedding size:", token.weight.size())
    print("Output size:", output.size())
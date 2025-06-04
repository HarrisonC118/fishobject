import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from .ConvSSM import PatchEmbed2D, SS2D, PatchMerging2D
from src.core import register


class ConvSSMStage(nn.Module):
    def __init__(self, dim, depth, d_state, d_conv, expand, dt_rank, dropouts, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            SS2D(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                dropout=dropouts[i] if isinstance(dropouts, list) else dropouts, # Use individual dropout rate
                **kwargs
            )
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


@register
class HierarchicalConvSSM(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],  # Corresponds to C2, C3, C4, C5 stages
                 dims=[96, 192, 384, 768], # Output channels for C2, C3, C4, C5
                 d_state=16,
                 d_conv=3,
                 expand=2,
                 dt_rank="auto",
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(1, 2, 3), # Indices of stages to output features from (0-indexed for stages list)
                 **kwargs):
        super().__init__()

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.feature_strides = [4, 8, 16, 32] # Strides for C2, C3, C4, C5 respectively

        # Patch Embedding
        self.patch_embed = PatchEmbed2D(
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.stages = nn.ModuleList()
        current_dim = embed_dim
        for i in range(self.num_stages):
            if i > 0: # Add PatchMerging before stages 1, 2, 3 (0-indexed)
                patch_merging = PatchMerging2D(dim=current_dim, norm_layer=norm_layer)
                self.stages.append(patch_merging)
                current_dim = current_dim * 2 # PatchMerging doubles the dimension
            
            stage = ConvSSMStage(
                dim=current_dim, # Should match dims[i]
                depth=depths[i],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                dropouts=dpr[sum(depths[:i]):sum(depths[:i+1])],
                **kwargs
            )
            self.stages.append(stage)
            
            # Add norm layer after each stage's output if needed for FPN
            # For RT-DETR, the neck (HybridEncoder) might handle normalization
            # For now, let's assume features are taken directly after SS2D blocks

        # Initialize weights
        self.apply(self._init_weights)
        self._out_channels = [dims[i] for i in out_indices]
        self._out_feature_strides = [self.feature_strides[i] for i in out_indices] # +1 because patch_embed is C1 (stride 4)
                                                                                    # and stages[0] is C2 (stride 4), stages[1] is C3 (stride 8) etc.
                                                                                    # More precisely, after patch_embed (stride 4), the first ConvSSMStage operates at this resolution.
                                                                                    # Then PatchMerging reduces resolution.
                                                                                    # Let's adjust out_indices and feature_strides logic to be more clear.
                                                                                    # If out_indices = (0,1,2,3) for C2,C3,C4,C5
                                                                                    # C2: after patch_embed + stage[0]
                                                                                    # C3: after stage[0] + patch_merge[0] + stage[1]
                                                                                    # C4: after stage[1] + patch_merge[1] + stage[2]
                                                                                    # C5: after stage[2] + patch_merge[2] + stage[3]
        # Let's refine the output logic. RT-DETR typically needs P3, P4, P5 (strides 8, 16, 32)
        # If embed_dim is for stride 4 (P2 level), then:
        # Stage 0 (depths[0], dims[0]): processes features at stride 4.
        # PatchMerging[0] + Stage 1 (depths[1], dims[1]): processes features at stride 8 (P3).
        # PatchMerging[1] + Stage 2 (depths[2], dims[2]): processes features at stride 16 (P4).
        # PatchMerging[2] + Stage 3 (depths[3], dims[3]): processes features at stride 32 (P5).
        # So, if out_indices refers to the output of these *merged* stages:
        # out_indices=(0,1,2) would correspond to P3, P4, P5.
        # dims should be [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8] if following typical doubling.
        # The provided `dims` parameter seems to specify the output channels of each stage *after* merging.

        self.feat_dims = [] # To store actual output channel dimensions for neck
        # Adjusting feature strides and output channels based on out_indices
        # The stages list contains [PatchEmbed, Stage0, PatchMerge0, Stage1, PatchMerge1, Stage2, PatchMerge2, Stage3]
        # Or more generally: [PatchEmbed, Stage_0, (PatchMerge_i, Stage_i+1) ... ]
        # Let's simplify: self.stages will store the main processing blocks (ConvSSMStage)
        # and self.downsamplers will store PatchMerging layers.

        del self.stages # Re-do stage definition for clarity
        self.downsamplers = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Stage 0 (operates on patch_embed output, e.g., C2 / Stride 4)
        stage0_dim = embed_dim
        self.stages.append(ConvSSMStage(
            dim=stage0_dim,
            depth=depths[0],
            d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank,
            dropouts=dpr[0:depths[0]], # Pass the list of dropouts for this stage
            **kwargs
        ))
        self.feat_dims.append(stage0_dim)

        # Subsequent stages with downsampling
        current_dim = stage0_dim
        for i in range(1, self.num_stages):
            self.downsamplers.append(PatchMerging2D(dim=current_dim, norm_layer=norm_layer))
            current_dim *= 2 # PatchMerging doubles the dimension
            self.stages.append(ConvSSMStage(
                dim=current_dim, # Should match dims[i]
                depth=depths[i],
                d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank,
                dropouts=dpr[sum(depths[:i]):sum(depths[:i+1])], # Pass the list of dropouts for this stage
                **kwargs
            ))
            self.feat_dims.append(current_dim)
        
        # Ensure `dims` parameter aligns with calculated `current_dim` at each stage if used for verification
        # For RT-DETR, we usually need features from later stages (e.g., P3, P4, P5)
        # out_indices will select from the outputs of self.stages
        # Example: if depths=[c2_depth, c3_depth, c4_depth, c5_depth]
        # self.stages[0] -> C2 features (stride 4)
        # self.stages[1] -> C3 features (stride 8)
        # self.stages[2] -> C4 features (stride 16)
        # self.stages[3] -> C5 features (stride 32)
        # If RT-DETR needs P3, P4, P5, then out_indices should be (1, 2, 3)

        self._out_channels = [self.feat_dims[i] for i in self.out_indices]
        self._out_feature_strides = [self.feature_strides[i] for i in self.out_indices] # P_i+2, so if out_indices=(1,2,3) -> P3,P4,P5 (strides 8,16,32)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x) # (B, H/4, W/4, embed_dim)
        x = x.permute(0, 3, 1, 2) # (B, embed_dim, H/4, W/4)

        outs = []
        current_feature_map = x
        for i in range(self.num_stages):
            if i == 0: # First stage (e.g., C2)
                current_feature_map = self.stages[i](current_feature_map)
            else: # Subsequent stages with downsampling (e.g., C3, C4, C5)
                # Permute for PatchMerging: (B, C, H, W) -> (B, H, W, C)
                current_feature_map = current_feature_map.permute(0, 2, 3, 1)
                current_feature_map = self.downsamplers[i-1](current_feature_map)
                # Permute back for ConvSSMStage: (B, H', W', C') -> (B, C', H', W')
                current_feature_map = current_feature_map.permute(0, 3, 1, 2)
                current_feature_map = self.stages[i](current_feature_map)
            
            if i in self.out_indices:
                outs.append(current_feature_map)
        
        return outs

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def out_feature_strides(self):
        return self._out_feature_strides


# Example usage (for testing, can be removed later)
if __name__ == '__main__':
    # Standard RT-DETR expects features from P3, P4, P5 (strides 8, 16, 32)
    # ResNet50 typically outputs [512, 1024, 2048] for these stages.
    # Let's try to match this with ConvSSM.
    # If embed_dim = 96 (for stride 4 / P2 level features)
    # Stage 0 (depths[0], embed_dim): P2, stride 4, channels 96
    # Stage 1 (depths[1], embed_dim*2): P3, stride 8, channels 192
    # Stage 2 (depths[2], embed_dim*4): P4, stride 16, channels 384
    # Stage 3 (depths[3], embed_dim*8): P5, stride 32, channels 768
    # So, if out_indices=(1,2,3), we get P3, P4, P5 features.
    # The `dims` parameter in __init__ should align with these calculated dimensions.

    backbone = HierarchicalConvSSM(
        in_chans=3,
        embed_dim=96, # Base dimension after patch embedding (P2 level)
        depths=[2, 2, 6, 2], # Number of SS2D blocks in P2, P3, P4, P5 stages respectively
        # dims are implicitly defined by embed_dim and patch merging
        d_state=16,
        d_conv=3,
        expand=2,
        out_indices=(1, 2, 3) # Output P3, P4, P5 features
    )

    dummy_input = torch.randn(1, 3, 640, 640)
    features = backbone(dummy_input)

    print("HierarchicalConvSSM initialized.")
    print(f"Output feature shapes and strides:")
    for i, feat in enumerate(features):
        print(f"  Output {i}: shape {feat.shape}, stride {backbone.out_feature_strides[i]}, channels {backbone.out_channels[i]}")

    # Expected output for out_indices=(1,2,3) with embed_dim=96:
    # P3: (1, 192, 80, 80), stride 8
    # P4: (1, 384, 40, 40), stride 16
    # P5: (1, 768, 20, 20), stride 32

    # Check if the actual output channels match the property
    assert [f.shape[1] for f in features] == backbone.out_channels, "Mismatch in output channels"

    # Test with different out_indices
    backbone_all_stages = HierarchicalConvSSM(
        embed_dim=64, depths=[1,1,1,1], out_indices=(0,1,2,3)
    )
    features_all = backbone_all_stages(torch.randn(1,3,256,256))
    print("\nFeatures from all stages (P2, P3, P4, P5):")
    for i, feat in enumerate(features_all):
        print(f"  Output {i}: shape {feat.shape}, stride {backbone_all_stages.out_feature_strides[i]}, channels {backbone_all_stages.out_channels[i]}")
    # Expected for embed_dim=64, out_indices=(0,1,2,3):
    # P2: (1, 64, 64, 64), stride 4
    # P3: (1, 128, 32, 32), stride 8
    # P4: (1, 256, 16, 16), stride 16
    # P5: (1, 512, 8, 8), stride 32

    # Check if the actual output channels match the property
    assert [f.shape[1] for f in features_all] == backbone_all_stages.out_channels, "Mismatch in output channels for all stages"
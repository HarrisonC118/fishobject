#!/usr/bin/env python3
"""
测试ASSM模块使用率的脚本
用于验证修改后的触发条件是否提高了ASSM的使用率
"""

import torch
import torch.nn.functional as F
import numpy as np

def simulate_small_target_mask(batch_size=2, num_queries=300, num_classes=80):
    """
    模拟_compute_small_target_mask函数的行为
    """
    # 模拟reference_points和score_logits
    reference_points = torch.rand(batch_size, num_queries, 2)  # [cx, cy]格式
    score_logits = torch.randn(batch_size, num_queries, num_classes)
    
    # 模拟原始条件（严格）
    areas_old = torch.ones_like(reference_points[:, :, 0]) * 0.005
    max_scores_old = torch.max(torch.softmax(score_logits, dim=-1), dim=-1)[0]
    area_mask_old = areas_old < 0.03
    score_mask_old = max_scores_old < 0.4
    small_target_mask_old = area_mask_old & score_mask_old
    
    # 模拟新条件（放宽）
    areas_new = torch.ones_like(reference_points[:, :, 0]) * 0.02
    max_scores_new = torch.max(torch.softmax(score_logits, dim=-1), dim=-1)[0]
    area_mask_new = areas_new < 0.05
    score_mask_new = max_scores_new < 0.6
    small_target_mask_new = area_mask_new | score_mask_new
    
    return small_target_mask_old, small_target_mask_new

def main():
    print("=== ASSM使用率对比测试 ===\n")
    
    # 运行多次模拟
    old_usage_rates = []
    new_usage_rates = []
    
    for i in range(10):
        old_mask, new_mask = simulate_small_target_mask()
        
        old_rate = old_mask.float().mean().item()
        new_rate = new_mask.float().mean().item()
        
        old_usage_rates.append(old_rate)
        new_usage_rates.append(new_rate)
        
        print(f"测试 {i+1:2d}: 原始条件使用率={old_rate:.3f}, 新条件使用率={new_rate:.3f}")
    
    print(f"\n=== 统计结果 ===")
    print(f"原始条件平均使用率: {np.mean(old_usage_rates):.3f} ± {np.std(old_usage_rates):.3f}")
    print(f"新条件平均使用率:   {np.mean(new_usage_rates):.3f} ± {np.std(new_usage_rates):.3f}")
    print(f"使用率提升倍数:     {np.mean(new_usage_rates)/np.mean(old_usage_rates):.2f}x")
    
    # 分析具体条件的贡献
    print(f"\n=== 条件分析 ===")
    old_mask, new_mask = simulate_small_target_mask(batch_size=4, num_queries=300)
    
    # 重新计算各个条件
    reference_points = torch.rand(4, 300, 2)
    score_logits = torch.randn(4, 300, 80)
    
    areas_new = torch.ones_like(reference_points[:, :, 0]) * 0.02
    max_scores_new = torch.max(torch.softmax(score_logits, dim=-1), dim=-1)[0]
    area_mask_new = areas_new < 0.05
    score_mask_new = max_scores_new < 0.6
    
    print(f"面积条件触发率:     {area_mask_new.float().mean().item():.3f}")
    print(f"置信度条件触发率:   {score_mask_new.float().mean().item():.3f}")
    print(f"组合条件触发率(OR): {(area_mask_new | score_mask_new).float().mean().item():.3f}")
    print(f"组合条件触发率(AND):{(area_mask_new & score_mask_new).float().mean().item():.3f}")

if __name__ == "__main__":
    main()

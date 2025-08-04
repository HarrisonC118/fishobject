"""
快速测试脚本 - 快速评估训练好的模型
使用方法:
python tools/quick_test.py --model-path output/rtdetr_r50vd_6x_coco/checkpoint_best_XXXX.pth
python tools/quick_test.py --model-path mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/best_mAP_model.pth
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import time
import torch
import json
from pathlib import Path

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def load_model_from_checkpoint(checkpoint_path, config_path=None):
    """从检查点加载模型"""
    print(f"加载模型: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 确定配置文件路径
    if config_path is None:
        config_path = "configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
    
    # 加载配置
    cfg = YAMLConfig(config_path)
    
    # 创建solver并初始化
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.setup()  # 重要：需要先setup才能访问model属性

    # 加载模型权重
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            # 标准检查点格式
            model_state = checkpoint['model']
            epoch = checkpoint.get('last_epoch', 0)
            print(f"加载epoch {epoch}的模型")
        elif 'ema' in checkpoint:
            # EMA模型
            model_state = checkpoint['ema']['module']
            print("加载EMA模型")
        else:
            # 直接的状态字典
            model_state = checkpoint
    else:
        model_state = checkpoint

    # 加载状态字典
    solver.model.load_state_dict(model_state, strict=False)
    solver.model.eval()
    
    return solver


def quick_metrics(model, device='cuda'):
    """快速计算基本指标"""
    print("\n=== 计算模型基本指标 ===")
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 快速FPS测试
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    # 测量FPS
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    num_runs = 50
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    fps = num_runs / (end_time - start_time)
    print(f"FPS (batch_size=1): {fps:.2f}")
    
    return {
        'params_M': total_params / 1e6,
        'fps': fps
    }


def quick_accuracy_test(solver):
    """快速精度测试（使用现有的验证功能）"""
    print("\n=== 快速精度测试 ===")
    
    # 使用solver的val方法进行评估
    solver.val()
    
    # 尝试读取最新的评估结果
    eval_file = solver.output_dir / "eval.pth"
    if eval_file.exists():
        eval_data = torch.load(eval_file, map_location='cpu')
        if hasattr(eval_data, 'stats'):
            stats = eval_data.stats
            print(f"mAP@0.5:0.95: {stats[0]:.4f}")
            print(f"mAP@0.5:     {stats[1]:.4f}")
            print(f"mAP@0.75:    {stats[2]:.4f}")
            return {
                'mAP': stats[0],
                'mAP_50': stats[1],
                'mAP_75': stats[2]
            }
    
    return {}


def main(args):
    """主函数"""
    print("=== 快速模型测试工具 ===")
    
    # 初始化分布式环境
    dist.init_distributed()
    
    # 加载模型
    solver = load_model_from_checkpoint(args.model_path, args.config)
    
    # 快速指标测试
    model = solver.ema.module if solver.ema else solver.model
    metrics = quick_metrics(model, device=str(solver.device))
    
    # 精度测试（可选）
    if not args.skip_accuracy:
        accuracy_metrics = quick_accuracy_test(solver)
        metrics.update(accuracy_metrics)
    
    # 打印结果
    print("\n=== 测试结果汇总 ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 保存结果
    if args.save_results:
        output_file = Path(args.model_path).parent / "quick_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='快速模型测试工具')
    parser.add_argument('--model-path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径（可选）')
    parser.add_argument('--skip-accuracy', action='store_true', help='跳过精度测试')
    parser.add_argument('--save-results', action='store_true', help='保存结果到JSON文件')
    
    args = parser.parse_args()
    main(args)

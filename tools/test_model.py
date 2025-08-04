"""
模型测试脚本 - 评估模型精度、参数量、FPS等指标
使用方法:
python tools/test_model.py --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml --resume path/to/checkpoint.pth
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import time
import torch
import torch.nn as nn
from pathlib import Path

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS
from src.solver.det_engine import evaluate
from src.data.coco import get_coco_api_from_dataset


def calculate_model_metrics(model, device='cuda', input_size=(3, 640, 640)):
    """计算模型的参数量、FLOPs和FPS等指标"""
    metrics = {}
    
    # 1. 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics['total_params'] = total_params
    metrics['trainable_params'] = trainable_params
    metrics['params_M'] = total_params / 1e6
    
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 2. 计算FLOPs (使用thop库)
    try:
        from thop import profile
        dummy_input = torch.randn(1, *input_size).to(device)
        model.eval()
        
        # 计算FLOPs
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        metrics['flops'] = flops
        metrics['gflops'] = flops / 1e9
        print(f"FLOPs: {flops:,} ({flops/1e9:.2f} GFLOPs)")
        
    except ImportError:
        print("警告: 未安装thop库，无法计算FLOPs。请运行: pip install thop")
        metrics['flops'] = 0
        metrics['gflops'] = 0
    except Exception as e:
        print(f"计算FLOPs时出错: {e}")
        metrics['flops'] = 0
        metrics['gflops'] = 0
    
    # 3. 测量FPS (batch_size=1)
    try:
        model.eval()
        dummy_input = torch.randn(1, *input_size).to(device)
        
        # 预热
        print("预热模型...")
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # 测量FPS
        print("测量FPS...")
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        num_runs = 100
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        fps = num_runs / (end_time - start_time)
        avg_latency = (end_time - start_time) / num_runs * 1000  # ms
        
        metrics['fps'] = fps
        metrics['latency_ms'] = avg_latency
        
        print(f"FPS (batch_size=1): {fps:.2f}")
        print(f"平均延迟: {avg_latency:.2f} ms")
        
    except Exception as e:
        print(f"测量FPS时出错: {e}")
        metrics['fps'] = 0
        metrics['latency_ms'] = 0
    
    # 4. 内存使用情况
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        metrics['memory_mb'] = memory_allocated
        print(f"GPU内存使用: {memory_allocated:.2f} MB")
    
    return metrics


def test_model_accuracy(solver, output_dir=None):
    """测试模型在验证集上的精度"""
    print("\n开始评估模型精度...")
    
    # 设置为评估模式
    solver.eval()
    
    # 获取COCO API
    base_ds = get_coco_api_from_dataset(solver.val_dataloader.dataset)
    
    # 选择模型（EMA或普通模型）
    module = solver.ema.module if solver.ema else solver.model
    
    # 进行评估
    test_stats, coco_evaluator = evaluate(
        module, 
        solver.criterion, 
        solver.postprocessor,
        solver.val_dataloader, 
        base_ds, 
        solver.device, 
        output_dir
    )
    
    # 提取COCO指标
    coco_metrics = {}
    if coco_evaluator is not None and 'bbox' in coco_evaluator.coco_eval:
        stats = coco_evaluator.coco_eval['bbox'].stats
        coco_metrics = {
            'mAP': stats[0],           # mAP@0.5:0.95
            'mAP_50': stats[1],        # mAP@0.5
            'mAP_75': stats[2],        # mAP@0.75
            'mAP_small': stats[3],     # mAP for small objects
            'mAP_medium': stats[4],    # mAP for medium objects
            'mAP_large': stats[5],     # mAP for large objects
            'AR_1': stats[6],          # AR with 1 detection per image
            'AR_10': stats[7],         # AR with 10 detections per image
            'AR_100': stats[8],        # AR with 100 detections per image
            'AR_small': stats[9],      # AR for small objects
            'AR_medium': stats[10],    # AR for medium objects
            'AR_large': stats[11]      # AR for large objects
        }
        
        print("\n=== COCO评估结果 ===")
        print(f"mAP@0.5:0.95: {coco_metrics['mAP']:.4f}")
        print(f"mAP@0.5:     {coco_metrics['mAP_50']:.4f}")
        print(f"mAP@0.75:    {coco_metrics['mAP_75']:.4f}")
        print(f"mAP_small:   {coco_metrics['mAP_small']:.4f}")
        print(f"mAP_medium:  {coco_metrics['mAP_medium']:.4f}")
        print(f"mAP_large:   {coco_metrics['mAP_large']:.4f}")
        print(f"AR@100:      {coco_metrics['AR_100']:.4f}")
    
    return coco_metrics, test_stats


def save_results_to_file(model_metrics, coco_metrics, output_path):
    """保存测试结果到文件"""
    results = {
        'model_metrics': model_metrics,
        'coco_metrics': coco_metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存为JSON格式
    import json
    json_path = output_path / 'test_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存为可读的文本格式
    txt_path = output_path / 'test_results.txt'
    with open(txt_path, 'w') as f:
        f.write("=== 模型测试结果 ===\n")
        f.write(f"测试时间: {results['timestamp']}\n\n")
        
        f.write("=== 模型指标 ===\n")
        f.write(f"参数量: {model_metrics.get('params_M', 0):.2f}M\n")
        f.write(f"FLOPs: {model_metrics.get('gflops', 0):.2f} GFLOPs\n")
        f.write(f"FPS (bs=1): {model_metrics.get('fps', 0):.2f}\n")
        f.write(f"延迟: {model_metrics.get('latency_ms', 0):.2f} ms\n")
        if 'memory_mb' in model_metrics:
            f.write(f"GPU内存: {model_metrics['memory_mb']:.2f} MB\n")
        
        f.write("\n=== COCO评估指标 ===\n")
        for key, value in coco_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"\n结果已保存到:")
    print(f"  JSON格式: {json_path}")
    print(f"  文本格式: {txt_path}")


def main(args):
    """主函数"""
    print("=== 模型测试工具 ===")
    print(f"配置文件: {args.config}")
    print(f"模型权重: {args.resume}")
    
    # 初始化分布式环境
    dist.init_distributed()
    
    # 加载配置
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    # 创建solver并初始化
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.setup()  # 重要：需要先setup才能访问model属性
    
    # 创建输出目录
    output_dir = Path(args.output_dir) if args.output_dir else Path('./test_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 计算模型指标（参数量、FLOPs、FPS）
    print("\n=== 计算模型指标 ===")
    model = solver.ema.module if solver.ema else solver.model
    model_metrics = calculate_model_metrics(model, device=str(solver.device))
    
    # 2. 测试模型精度
    if not args.skip_accuracy:
        coco_metrics, test_stats = test_model_accuracy(solver, output_dir)
    else:
        print("\n跳过精度测试")
        coco_metrics = {}
    
    # 3. 保存结果
    save_results_to_file(model_metrics, coco_metrics, output_dir)
    
    # 4. 打印汇总
    print("\n=== 测试完成 ===")
    print("模型性能汇总:")
    print(f"  参数量: {model_metrics.get('params_M', 0):.2f}M")
    print(f"  FLOPs: {model_metrics.get('gflops', 0):.2f} GFLOPs")
    print(f"  FPS: {model_metrics.get('fps', 0):.2f}")
    if coco_metrics:
        print(f"  mAP@0.5:0.95: {coco_metrics.get('mAP', 0):.4f}")
        print(f"  mAP@0.5: {coco_metrics.get('mAP_50', 0):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型测试工具')
    parser.add_argument('--config', '-c', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', '-r', type=str, required=True, help='模型权重路径')
    parser.add_argument('--output-dir', type=str, default='./test_results', help='结果输出目录')
    parser.add_argument('--skip-accuracy', action='store_true', help='跳过精度测试，只计算模型指标')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型')
    
    args = parser.parse_args()
    main(args)

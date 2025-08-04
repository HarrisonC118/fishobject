"""
批量测试脚本 - 一次性测试多个模型并生成比较表格
使用方法:
python tools/batch_test.py --models-dir mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import time
import torch
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def load_and_test_model(model_path: str, config_path: str) -> Dict:
    """加载并测试单个模型"""
    print(f"\n测试模型: {model_path}")
    
    try:
        # 加载检查点
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 加载配置
        cfg = YAMLConfig(config_path)
        solver = TASKS[cfg.yaml_cfg['task']](cfg)
        solver.setup()  # 重要：需要先setup才能访问model属性
        
        # 加载模型权重
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                epoch = checkpoint.get('last_epoch', 0)
                extra_info = f"epoch_{epoch}"
            elif 'ema' in checkpoint:
                model_state = checkpoint['ema']['module']
                extra_info = "ema"
            else:
                model_state = checkpoint
                extra_info = "direct"
        else:
            model_state = checkpoint
            extra_info = "direct"
        
        solver.model.load_state_dict(model_state, strict=False)
        solver.model.eval()
        
        # 计算基本指标
        model = solver.ema.module if solver.ema else solver.model
        device = str(solver.device)
        
        # 参数量
        total_params = sum(p.numel() for p in model.parameters())
        
        # FPS测试
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        
        # 预热
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        
        # 测量FPS
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        num_runs = 30  # 减少运行次数以加快批量测试
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        fps = num_runs / (end_time - start_time)
        
        # 尝试计算FLOPs
        try:
            from thop import profile
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            gflops = flops / 1e9
        except:
            gflops = 0
        
        results = {
            'model_name': Path(model_path).stem,
            'model_path': str(model_path),
            'params_M': total_params / 1e6,
            'gflops': gflops,
            'fps': fps,
            'extra_info': extra_info,
            'status': 'success'
        }
        
        print(f"  ✓ 参数量: {results['params_M']:.2f}M")
        print(f"  ✓ FPS: {results['fps']:.2f}")
        if gflops > 0:
            print(f"  ✓ GFLOPs: {results['gflops']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return {
            'model_name': Path(model_path).stem,
            'model_path': str(model_path),
            'params_M': 0,
            'gflops': 0,
            'fps': 0,
            'extra_info': 'error',
            'status': f'failed: {str(e)[:50]}...'
        }


def find_model_files(models_dir: str) -> List[str]:
    """查找模型文件"""
    models_dir = Path(models_dir)
    model_files = []
    
    # 查找.pth文件
    for pattern in ['*.pth', '*.pt']:
        model_files.extend(models_dir.glob(pattern))
    
    # 如果是目录，递归查找
    if models_dir.is_dir():
        for pattern in ['**/*.pth', '**/*.pt']:
            model_files.extend(models_dir.glob(pattern))
    
    return [str(f) for f in model_files]


def generate_comparison_table(results: List[Dict], output_dir: Path):
    """生成比较表格"""
    if not results:
        print("没有测试结果，无法生成表格")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 按FPS排序
    df = df.sort_values('fps', ascending=False)
    
    # 保存为CSV
    csv_path = output_dir / 'model_comparison.csv'
    df.to_csv(csv_path, index=False)
    
    # 生成Markdown表格
    md_path = output_dir / 'model_comparison.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 模型性能比较\n\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 成功的模型
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            f.write("## 测试结果\n\n")
            f.write("| 模型名称 | 参数量(M) | GFLOPs | FPS | 备注 |\n")
            f.write("|---------|-----------|--------|-----|------|\n")
            
            for result in successful_results:
                f.write(f"| {result['model_name']} | {result['params_M']:.2f} | {result['gflops']:.2f} | {result['fps']:.2f} | {result['extra_info']} |\n")
        
        # 失败的模型
        failed_results = [r for r in results if r['status'] != 'success']
        if failed_results:
            f.write("\n## 测试失败的模型\n\n")
            f.write("| 模型名称 | 错误信息 |\n")
            f.write("|---------|----------|\n")
            
            for result in failed_results:
                f.write(f"| {result['model_name']} | {result['status']} |\n")
    
    # 生成HTML表格
    html_path = output_dir / 'model_comparison.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>模型性能比较</title>
    <style>
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .success { background-color: #d4edda; }
        .failed { background-color: #f8d7da; }
    </style>
</head>
<body>
    <h1>模型性能比较</h1>
""")
        
        if successful_results:
            f.write("<h2>测试结果</h2>")
            f.write("<table>")
            f.write("<tr><th>模型名称</th><th>参数量(M)</th><th>GFLOPs</th><th>FPS</th><th>备注</th></tr>")
            
            for result in successful_results:
                f.write(f"<tr class='success'>")
                f.write(f"<td>{result['model_name']}</td>")
                f.write(f"<td>{result['params_M']:.2f}</td>")
                f.write(f"<td>{result['gflops']:.2f}</td>")
                f.write(f"<td>{result['fps']:.2f}</td>")
                f.write(f"<td>{result['extra_info']}</td>")
                f.write(f"</tr>")
            
            f.write("</table>")
        
        f.write("</body></html>")
    
    print(f"\n比较表格已生成:")
    print(f"  CSV: {csv_path}")
    print(f"  Markdown: {md_path}")
    print(f"  HTML: {html_path}")


def main(args):
    """主函数"""
    print("=== 批量模型测试工具 ===")
    
    # 初始化分布式环境
    dist.init_distributed()
    
    # 查找模型文件
    if args.models_dir:
        model_files = find_model_files(args.models_dir)
    else:
        model_files = args.model_paths
    
    if not model_files:
        print("未找到模型文件")
        return
    
    print(f"找到 {len(model_files)} 个模型文件:")
    for f in model_files:
        print(f"  {f}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 批量测试
    results = []
    for i, model_path in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}] 测试模型: {Path(model_path).name}")
        result = load_and_test_model(model_path, args.config)
        results.append(result)
    
    # 保存详细结果
    results_file = output_dir / 'batch_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成比较表格
    generate_comparison_table(results, output_dir)
    
    # 打印汇总
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len(results) - successful
    
    print(f"\n=== 批量测试完成 ===")
    print(f"总计: {len(results)} 个模型")
    print(f"成功: {successful} 个")
    print(f"失败: {failed} 个")
    print(f"结果保存在: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量模型测试工具')
    parser.add_argument('--models-dir', type=str, help='模型目录路径')
    parser.add_argument('--model-paths', nargs='+', help='具体的模型文件路径列表')
    parser.add_argument('--config', type=str, default='configs/rtdetr/rtdetr_r50vd_6x_coco.yml', help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default='./batch_test_results', help='结果输出目录')
    
    args = parser.parse_args()
    
    if not args.models_dir and not args.model_paths:
        print("错误: 必须指定 --models-dir 或 --model-paths")
        sys.exit(1)
    
    main(args)

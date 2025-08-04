#!/usr/bin/env python3
"""
模型测试运行脚本
根据你的训练命令自动测试模型性能
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description=""):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"执行失败: {e}")
        return False

def main():
    """主函数"""
    print("=== SSM-DETR 模型测试脚本 ===")
    print("根据你的训练命令，自动测试模型性能")
    
    # 基本路径设置
    model_name = "DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5]"
    best_models_dir = f"mainlog/{model_name}.log/best_models"
    config_file = "configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
    
    # 检查文件是否存在
    print(f"\n检查模型文件...")
    if not Path(best_models_dir).exists():
        print(f"❌ 未找到最佳模型目录: {best_models_dir}")
        print("请确保训练已完成并生成了最佳模型")
        return
    
    best_models = list(Path(best_models_dir).glob("*.pth"))
    if not best_models:
        print(f"❌ 在 {best_models_dir} 中未找到模型文件")
        return
    
    print(f"✅ 找到 {len(best_models)} 个最佳模型:")
    for model in best_models:
        print(f"   {model.name}")
    
    # 测试1: 快速测试最佳mAP模型
    best_map_model = Path(best_models_dir) / "best_mAP_model.pth"
    if best_map_model.exists():
        cmd = f"python tools/quick_test.py --model-path {best_map_model} --skip-accuracy --save-results"
        run_command(cmd, "快速测试最佳mAP模型")
    
    # 测试2: 完整测试（跳过精度以节省时间）
    cmd = f"python tools/test_model.py --config {config_file} --resume {best_map_model} --skip-accuracy"
    run_command(cmd, "完整模型指标测试")
    
    # 测试3: 批量测试所有最佳模型
    cmd = f"python tools/batch_test.py --models-dir {best_models_dir}"
    run_command(cmd, "批量测试所有最佳模型")
    
    # 显示结果文件
    print(f"\n{'='*60}")
    print("测试完成！生成的结果文件:")
    print('='*60)
    
    result_files = [
        "test_results/test_results.json",
        "test_results/test_results.txt", 
        "batch_test_results/model_comparison.md",
        "batch_test_results/model_comparison.csv",
        f"{best_models_dir}/quick_test_results.json"
    ]
    
    for file_path in result_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (未生成)")
    
    print(f"\n推荐查看:")
    print(f"1. 快速结果: cat {best_models_dir}/quick_test_results.json")
    print(f"2. 详细结果: cat test_results/test_results.txt")
    print(f"3. 比较表格: cat batch_test_results/model_comparison.md")
    
    # 显示模型性能总结
    try:
        import json
        quick_results_file = Path(best_models_dir) / "quick_test_results.json"
        if quick_results_file.exists():
            with open(quick_results_file) as f:
                results = json.load(f)
            
            print(f"\n{'='*60}")
            print("模型性能总结 (最佳mAP模型):")
            print('='*60)
            print(f"参数量: {results.get('params_M', 0):.2f}M")
            print(f"FPS (batch_size=1): {results.get('fps', 0):.2f}")
            
            # 与基准比较
            print(f"\n与RT-DETR R50基准比较:")
            print(f"基准参数量: ~42M")
            print(f"基准FPS: ~108 (T4 GPU)")
            print(f"基准mAP: ~53.1")
            
            params_ratio = results.get('params_M', 0) / 42
            fps_ratio = results.get('fps', 0) / 108
            print(f"参数量比例: {params_ratio:.2f}x")
            print(f"FPS比例: {fps_ratio:.2f}x")
            
    except Exception as e:
        print(f"无法读取结果文件: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
模型对比分析脚本
比较SSM-DETR和YOLO模型的性能差异
"""

import pandas as pd
import json
from pathlib import Path

def analyze_model_differences():
    """分析模型差异"""
    
    print("=" * 80)
    print("🔍 SSM-DETR vs YOLO 模型对比分析")
    print("=" * 80)
    
    # SSM-DETR数据 (从测试结果获取)
    ssm_detr_data = {
        'model': 'SSM-DETR (R50-ASSM)',
        'params_m': 43.17,
        'gflops': 67.98,
        'fps': 30.5,
        'architecture': 'Transformer-based',
        'backbone': 'ResNet50',
        'type': 'Research/Academic'
    }
    
    # YOLO数据 (从CSV文件获取)
    yolo_data = [
        {
            'model': 'YOLOv8n',
            'params_m': 3.01,
            'gflops': 8.1,  # 估算值
            'fps': 112.9,
            'architecture': 'CNN-based',
            'backbone': 'CSPDarknet',
            'type': 'Production/Deployment'
        },
        {
            'model': 'YOLO11n', 
            'params_m': 2.59,
            'gflops': 6.5,  # 估算值
            'fps': 72.1,
            'architecture': 'CNN-based',
            'backbone': 'Enhanced CSPDarknet',
            'type': 'Production/Deployment'
        },
        {
            'model': 'YOLO12n',
            'params_m': 2.57,
            'gflops': 6.4,  # 估算值  
            'fps': 44.4,
            'architecture': 'CNN-based',
            'backbone': 'Advanced CSPDarknet',
            'type': 'Production/Deployment'
        }
    ]
    
    # 创建对比表格
    all_models = [ssm_detr_data] + yolo_data
    df = pd.DataFrame(all_models)
    
    print("\n📊 模型参数对比:")
    print("-" * 80)
    print(f"{'模型':<20} {'参数量(M)':<12} {'GFLOPs':<10} {'FPS':<8} {'架构类型':<15}")
    print("-" * 80)
    
    for model in all_models:
        print(f"{model['model']:<20} {model['params_m']:<12.2f} {model['gflops']:<10.1f} {model['fps']:<8.1f} {model['architecture']:<15}")
    
    print("\n🔍 关键差异分析:")
    print("-" * 80)
    
    # 参数量对比
    yolo_avg_params = sum(m['params_m'] for m in yolo_data) / len(yolo_data)
    params_ratio = ssm_detr_data['params_m'] / yolo_avg_params
    print(f"1. 参数量差异:")
    print(f"   • SSM-DETR: {ssm_detr_data['params_m']:.2f}M")
    print(f"   • YOLO平均: {yolo_avg_params:.2f}M") 
    print(f"   • 差异倍数: {params_ratio:.1f}x")
    
    # FPS对比
    yolo_avg_fps = sum(m['fps'] for m in yolo_data) / len(yolo_data)
    fps_ratio = yolo_avg_fps / ssm_detr_data['fps']
    print(f"\n2. 推理速度差异:")
    print(f"   • SSM-DETR: {ssm_detr_data['fps']:.1f} FPS")
    print(f"   • YOLO平均: {yolo_avg_fps:.1f} FPS")
    print(f"   • YOLO快: {fps_ratio:.1f}x")
    
    # 计算效率对比
    ssm_efficiency = ssm_detr_data['fps'] / ssm_detr_data['params_m']
    yolo_avg_efficiency = yolo_avg_fps / yolo_avg_params
    efficiency_ratio = yolo_avg_efficiency / ssm_efficiency
    print(f"\n3. 参数效率 (FPS/M参数):")
    print(f"   • SSM-DETR: {ssm_efficiency:.2f}")
    print(f"   • YOLO平均: {yolo_avg_efficiency:.2f}")
    print(f"   • YOLO效率高: {efficiency_ratio:.1f}x")
    
    print("\n💡 为什么差异这么大?")
    print("-" * 80)
    print("1. 🏗️  架构设计目标不同:")
    print("   • SSM-DETR: 研究型模型，追求精度和新颖性")
    print("   • YOLO: 工业级模型，追求速度和部署效率")
    
    print("\n2. 🧠 核心组件差异:")
    print("   • SSM-DETR: Transformer + ResNet50 + ASSM模块")
    print("   • YOLO: 轻量级CNN + 优化的特征提取")
    
    print("\n3. 🎯 设计理念:")
    print("   • SSM-DETR: 学术研究，验证新方法")
    print("   • YOLO: 实用部署，平衡精度与速度")
    
    print("\n4. 📐 模型复杂度:")
    print("   • Transformer注意力机制参数多")
    print("   • ResNet50 backbone本身就有32M+参数")
    print("   • YOLO使用专门设计的轻量级结构")
    
    print("\n📈 性能权衡分析:")
    print("-" * 80)
    print("• 参数量: YOLO << SSM-DETR (约14倍差异)")
    print("• 推理速度: YOLO >> SSM-DETR (约2-4倍)")
    print("• 内存占用: YOLO << SSM-DETR")
    print("• 部署难度: YOLO << SSM-DETR")
    print("• 精度潜力: SSM-DETR可能更高 (需要完整评估)")
    
    print("\n🚀 建议:")
    print("-" * 80)
    print("1. 如果追求部署效率 → 选择YOLO")
    print("2. 如果进行学术研究 → 继续SSM-DETR")
    print("3. 可以考虑轻量化SSM-DETR:")
    print("   • 使用更小的backbone (如ResNet18)")
    print("   • 减少Transformer层数")
    print("   • 优化ASSM模块结构")
    
    # 保存分析结果
    analysis_results = {
        'comparison_data': all_models,
        'analysis': {
            'params_ratio': params_ratio,
            'fps_ratio': fps_ratio,
            'efficiency_ratio': efficiency_ratio
        },
        'conclusions': [
            "SSM-DETR是研究型模型，参数量大但可能精度更高",
            "YOLO是工业级模型，轻量高效适合部署",
            "两者设计目标不同，不能简单比较优劣"
        ]
    }
    
    with open('model_comparison_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n💾 分析结果已保存到: model_comparison_analysis.json")

if __name__ == "__main__":
    analyze_model_differences()

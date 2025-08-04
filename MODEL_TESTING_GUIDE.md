# 模型测试指南

本指南介绍如何在ssm_detr项目中测试训练好的模型，包括精度评估、参数量计算、FPS测试等。

## 测试方法

### 1. 使用训练脚本的测试模式

这是最标准的测试方法，使用项目原生的评估功能：

```bash
# 测试单GPU
python tools/train.py \
    --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --resume output/rtdetr_r50vd_6x_coco/checkpoint_best_XXXX.pth \
    --test-only

# 测试多GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py \
    --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --resume output/rtdetr_r50vd_6x_coco/checkpoint_best_XXXX.pth \
    --test-only
```

### 2. 使用专门的测试脚本

我们提供了两个专门的测试脚本：

#### 完整测试脚本 (`tools/test_model.py`)

提供全面的模型评估，包括：
- 参数量计算
- FLOPs计算（需要安装thop: `pip install thop`）
- FPS测试（batch_size=1）
- 完整的COCO指标评估
- 内存使用统计

```bash
# 完整测试
python tools/test_model.py \
    --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --resume output/rtdetr_r50vd_6x_coco/checkpoint_best_XXXX.pth \
    --output-dir ./test_results

# 只测试模型指标，跳过精度评估（更快）
python tools/test_model.py \
    --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --resume output/rtdetr_r50vd_6x_coco/checkpoint_best_XXXX.pth \
    --skip-accuracy
```

#### 快速测试脚本 (`tools/quick_test.py`)

提供快速的基本评估：

```bash
# 快速测试
python tools/quick_test.py \
    --model-path output/rtdetr_r50vd_6x_coco/checkpoint_best_XXXX.pth

# 测试最佳模型
python tools/quick_test.py \
    --model-path mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/best_mAP_model.pth \
    --save-results
```

## 模型路径说明

根据你的训练命令，模型保存在以下位置：

### 主要检查点
```
output/rtdetr_r50vd_6x_coco/
├── checkpoint.pth                    # 最新检查点
├── checkpoint_best_XXXX.pth         # 性能最佳的检查点
└── eval/                            # 评估结果
```

### 最佳模型（推荐使用）
```
mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/
├── best_mAP_model.pth              # mAP最佳模型
├── best_mAP_50_model.pth           # mAP@0.5最佳模型
└── best_f1_score_model.pth         # F1分数最佳模型
```

## 测试示例

### 测试你训练的模型

```bash
# 1. 测试最佳mAP模型（推荐）
python tools/test_model.py \
    --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --resume mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/best_mAP_model.pth

# 2. 测试最佳mAP@0.5模型（与YOLO比较时使用）
python tools/test_model.py \
    --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --resume mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/best_mAP_50_model.pth

# 3. 快速测试（只看基本指标）
python tools/quick_test.py \
    --model-path mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/best_mAP_model.pth \
    --skip-accuracy
```

## 输出指标说明

### 模型指标
- **参数量 (Params)**: 模型的总参数数量，单位M（百万）
- **FLOPs**: 浮点运算次数，单位GFLOPs（十亿次浮点运算）
- **FPS**: 每秒处理帧数（batch_size=1）
- **延迟 (Latency)**: 单次推理时间，单位ms
- **内存使用**: GPU内存占用，单位MB

### COCO评估指标
- **mAP@0.5:0.95**: 主要精度指标，IoU阈值0.5-0.95的平均AP
- **mAP@0.5**: IoU阈值0.5的AP（与YOLO比较常用）
- **mAP@0.75**: IoU阈值0.75的AP（高精度检测）
- **mAP_small/medium/large**: 不同尺寸物体的AP
- **AR@100**: 最多100个检测的平均召回率

## 性能基准

根据README中的模型zoo，rtdetr_r50vd的预期性能：
- **参数量**: ~42M
- **FPS**: ~108 (T4 GPU)
- **mAP@0.5:0.95**: ~53.1
- **mAP@0.5**: ~71.2

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 使用CPU测试
   python tools/test_model.py --device cpu ...
   ```

2. **找不到数据集**
   - 确保COCO数据集路径在配置文件中正确设置
   - 检查 `configs/dataset/coco_detection.yml`

3. **thop库未安装**
   ```bash
   pip install thop
   ```

4. **权限问题**
   ```bash
   chmod +x tools/test_model.py
   chmod +x tools/quick_test.py
   ```

#### 批量测试脚本 (`tools/batch_test.py`)

一次性测试多个模型并生成比较表格：

```bash
# 测试目录下的所有模型
python tools/batch_test.py \
    --models-dir mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/

# 测试指定的多个模型
python tools/batch_test.py \
    --model-paths \
        mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/best_mAP_model.pth \
        mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/best_mAP_50_model.pth \
        output/rtdetr_r50vd_6x_coco/checkpoint_best_0108.pth
```

### 测试不同的检查点

```bash
# 方法1: 使用批量测试脚本（推荐）
python tools/batch_test.py \
    --models-dir mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/

# 方法2: 手动循环测试
for model in mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/*.pth; do
    echo "Testing $model"
    python tools/quick_test.py --model-path "$model" --save-results
done
```

## 结果文件

### 单个模型测试结果
- `test_results/test_results.json`: JSON格式的详细结果
- `test_results/test_results.txt`: 人类可读的文本格式
- `quick_test_results.json`: 快速测试的结果（如果使用--save-results）

### 批量测试结果
- `batch_test_results/batch_test_results.json`: 所有模型的详细测试结果
- `batch_test_results/model_comparison.csv`: CSV格式的比较表格
- `batch_test_results/model_comparison.md`: Markdown格式的比较表格
- `batch_test_results/model_comparison.html`: HTML格式的比较表格

这些文件包含了所有的性能指标，可以用于论文写作或性能比较。

## 推荐的测试流程

1. **训练完成后立即测试**：
   ```bash
   # 快速验证训练结果
   python tools/quick_test.py \
       --model-path mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/best_mAP_model.pth
   ```

2. **详细性能评估**：
   ```bash
   # 完整的性能测试
   python tools/test_model.py \
       --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
       --resume mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/best_mAP_model.pth
   ```

3. **多模型比较**：
   ```bash
   # 批量测试所有最佳模型
   python tools/batch_test.py \
       --models-dir mainlog/DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5].log/best_models/
   ```

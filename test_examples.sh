#!/bin/bash

# 模型测试示例脚本
# 根据你的训练命令，这里提供了相应的测试命令示例

echo "=== SSM-DETR 模型测试示例 ==="
echo "请根据你的实际模型路径修改以下命令"
echo ""

# 设置基本路径
CONFIG="configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
MODEL_NAME="DETR-R50-TokenwiseEBlock-ASSM[0.05,0.5]"
BEST_MODELS_DIR="mainlog/${MODEL_NAME}.log/best_models"
OUTPUT_DIR="output/rtdetr_r50vd_6x_coco"

echo "配置文件: $CONFIG"
echo "模型名称: $MODEL_NAME"
echo "最佳模型目录: $BEST_MODELS_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查文件是否存在
echo "=== 检查模型文件 ==="
if [ -d "$BEST_MODELS_DIR" ]; then
    echo "✓ 找到最佳模型目录: $BEST_MODELS_DIR"
    ls -la "$BEST_MODELS_DIR"
else
    echo "✗ 未找到最佳模型目录: $BEST_MODELS_DIR"
fi

if [ -d "$OUTPUT_DIR" ]; then
    echo "✓ 找到输出目录: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"/*.pth 2>/dev/null || echo "  (未找到.pth文件)"
else
    echo "✗ 未找到输出目录: $OUTPUT_DIR"
fi
echo ""

# 示例命令
echo "=== 测试命令示例 ==="

echo "1. 快速测试最佳mAP模型:"
echo "python tools/quick_test.py \\"
echo "    --model-path $BEST_MODELS_DIR/best_mAP_model.pth"
echo ""

echo "2. 详细测试最佳mAP模型:"
echo "python tools/test_model.py \\"
echo "    --config $CONFIG \\"
echo "    --resume $BEST_MODELS_DIR/best_mAP_model.pth"
echo ""

echo "3. 批量测试所有最佳模型:"
echo "python tools/batch_test.py \\"
echo "    --models-dir $BEST_MODELS_DIR"
echo ""

echo "4. 使用原生测试模式:"
echo "python tools/train.py \\"
echo "    --config $CONFIG \\"
echo "    --resume $BEST_MODELS_DIR/best_mAP_model.pth \\"
echo "    --test-only"
echo ""

echo "5. 测试特定检查点:"
echo "python tools/test_model.py \\"
echo "    --config $CONFIG \\"
echo "    --resume $OUTPUT_DIR/checkpoint_best_XXXX.pth"
echo ""

# 实际运行示例（注释掉，用户可以取消注释来运行）
echo "=== 自动运行示例（取消注释来执行） ==="
echo "# 如果要自动运行测试，请取消以下行的注释:"
echo ""
echo "# 快速测试（如果模型存在）"
echo "# if [ -f \"$BEST_MODELS_DIR/best_mAP_model.pth\" ]; then"
echo "#     echo \"运行快速测试...\""
echo "#     python tools/quick_test.py --model-path \"$BEST_MODELS_DIR/best_mAP_model.pth\" --save-results"
echo "# fi"
echo ""
echo "# 批量测试（如果目录存在）"
echo "# if [ -d \"$BEST_MODELS_DIR\" ]; then"
echo "#     echo \"运行批量测试...\""
echo "#     python tools/batch_test.py --models-dir \"$BEST_MODELS_DIR\""
echo "# fi"

echo ""
echo "=== 使用说明 ==="
echo "1. 首先检查上面显示的文件路径是否正确"
echo "2. 根据实际情况修改路径"
echo "3. 选择合适的测试命令运行"
echo "4. 查看生成的结果文件"
echo ""
echo "详细说明请参考: MODEL_TESTING_GUIDE.md"

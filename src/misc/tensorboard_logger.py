"""TensorBoard Logger for RTDETR

专注于记录目标检测模型的关键性能指标，包括AP、AR、F1和其他重要指标。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError(
        "TensorBoard logging requires PyTorch with TensorBoard support. "
        "Please install tensorboard: pip install tensorboard"
    )


class TensorboardLogger:
    """专注于目标检测指标的TensorBoard记录器"""
    
    def __init__(self, log_dir=None, enabled=True):
        """初始化TensorBoard记录器
        
        Args:
            log_dir (str, optional): TensorBoard日志目录，默认为'runs/'
            enabled (bool, optional): 是否启用TensorBoard记录，默认为True
        """
        self.enabled = enabled
        self.writer = None
        
        if self.enabled:
            if log_dir is None:
                log_dir = Path("runs/")
            else:
                log_dir = Path(log_dir)
                
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard记录器已初始化，日志目录：{log_dir}")
    
    def log_metrics(self, metrics, step, prefix=""):
        """记录通用指标
        
        Args:
            metrics (dict): 指标字典
            step (int): 全局步数
            prefix (str, optional): 指标名称前缀
        """
        if not self.enabled or self.writer is None:
            return
            
        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                self.writer.add_scalar(f"{prefix}{k}", v, step)
    
    def log_detection_metrics(self, coco_eval, step):
        """记录目标检测关键指标

        Args:
            coco_eval: COCO评估器对象
            step (int): 全局步数
        """
        if not self.enabled or self.writer is None or coco_eval is None:
            return

        # 1. 记录主要AP指标（最关键的目标检测指标）
        ap_metrics = {
            'mAP': coco_eval.stats[0],  # IoU=0.5:0.95的AP
            'mAP_50': coco_eval.stats[1],  # IoU=0.5的AP
            'mAP_75': coco_eval.stats[2],  # IoU=0.75的AP
            'mAP_small': coco_eval.stats[3],  # 小物体的AP
            'mAP_medium': coco_eval.stats[4],  # 中物体的AP
            'mAP_large': coco_eval.stats[5],  # 大物体的AP
        }
        
        # 2. 记录召回率指标
        ar_metrics = {
            'AR_max_1': coco_eval.stats[6],   # 每张图像最多1个检测结果的AR
            'AR_max_10': coco_eval.stats[7],  # 每张图像最多10个检测结果的AR
            'AR_max_100': coco_eval.stats[8], # 每张图像最多100个检测结果的AR
            'AR_small': coco_eval.stats[9],   # 小物体的AR
            'AR_medium': coco_eval.stats[10], # 中物体的AR
            'AR_large': coco_eval.stats[11],  # 大物体的AR
        }
        
        # 记录AP指标（主要指标）
        for name, value in ap_metrics.items():
            self.writer.add_scalar(f'detection/AP/{name}', value, step)
            print(f"记录指标 detection/AP/{name}: {value:.4f}")
        
        # 记录AR指标
        for name, value in ar_metrics.items():
            self.writer.add_scalar(f'detection/AR/{name}', value, step)
            print(f"记录指标 detection/AR/{name}: {value:.4f}")

        # 3. 计算并记录F1分数（精确率和召回率的调和平均值）
        # 使用IoU=0.5时的AP作为精确率，计算F1
        precision_50 = ap_metrics['mAP_50']
        
        # 使用AR_max_100作为召回率
        recall = ar_metrics['AR_max_100']
        
        # 计算F1分数
        f1_score = 2 * (precision_50 * recall) / (precision_50 + recall + 1e-6)
        
        # 记录精确率、召回率和F1分数
        self.writer.add_scalar('detection/precision', precision_50, step)
        self.writer.add_scalar('detection/recall', recall, step)
        self.writer.add_scalar('detection/f1_score', f1_score, step)
        
        print(f"记录关键指标 - Precision: {precision_50:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
            
        # 4. 计算并记录每个类别的AP和F1分数
        if hasattr(coco_eval, 'eval') and 'precision' in coco_eval.eval:
            if hasattr(coco_eval.params, 'catIds'):
                precisions = coco_eval.eval['precision']
                # 对于每个类别计算AP (IoU=0.5)
                for idx, cat_id in enumerate(coco_eval.params.catIds):
                    if idx < precisions.shape[2]:
                        # 计算该类别在IoU=0.5时的AP
                        ap50 = np.mean(precisions[0, :, idx, 0, -1])
                        
                        # 获取类别名称
                        category_name = f"category_{cat_id}"
                        if hasattr(coco_eval, 'cocoGt') and hasattr(coco_eval.cocoGt, 'cats') and cat_id in coco_eval.cocoGt.cats:
                            category_name = coco_eval.cocoGt.cats[cat_id]['name']
                        
                        # 记录每个类别的AP
                        self.writer.add_scalar(f'detection/category_AP/{category_name}', ap50, step)
                        print(f"记录类别 {category_name} AP: {ap50:.4f}")
        
        # 5. 可视化PR曲线（整体和每个类别）
        if hasattr(coco_eval, 'eval') and 'precision' in coco_eval.eval:
            self._log_pr_curve(coco_eval, step)
        
        # 确保立即写入数据
        self.flush()
    
    def _log_pr_curve(self, coco_eval, step):
        """记录PR曲线
        
        Args:
            coco_eval: COCO评估器对象
            step (int): 全局步数
        """
        try:
            # 创建整体PR曲线（所有类别的平均）
            fig, ax = plt.subplots(figsize=(10, 8))
        
            # 获取PR曲线数据（对所有类别和IoU=0.5的情况）
            precisions = coco_eval.eval['precision']
            # 取IoU=0.5的精确率
            precision_at_iou50 = precisions[0, :, :, 0, :]  # [recall, category, area]
            
            # 计算所有类别的平均精确率
            mean_precision = np.mean(precision_at_iou50, axis=1)
        
            # 绘制PR曲线
            recall_thresholds = np.linspace(0, 1, mean_precision.shape[0])
            ax.plot(recall_thresholds, mean_precision, 'b-', label='mAP@IoU=0.5')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.grid(True)
            ax.legend()
            
            # 记录整体PR曲线
            self.writer.add_figure('detection/pr_curve', fig, step)
            plt.close(fig)
                
            # 为每个类别创建PR曲线（如果类别数量不多）
            if hasattr(coco_eval.params, 'catIds') and len(coco_eval.params.catIds) <= 20:
                fig, ax = plt.subplots(figsize=(12, 10))
                
                for idx, cat_id in enumerate(coco_eval.params.catIds):
                    if idx < precision_at_iou50.shape[1]:
                        # 获取该类别的PR曲线
                        cat_precision = precision_at_iou50[:, idx, -1]
                        
                        # 获取类别名称
                        category_name = f"category_{cat_id}"
                        if hasattr(coco_eval, 'cocoGt') and hasattr(coco_eval.cocoGt, 'cats') and cat_id in coco_eval.cocoGt.cats:
                            category_name = coco_eval.cocoGt.cats[cat_id]['name']
                        
                        # 绘制该类别的PR曲线
                        ax.plot(recall_thresholds, cat_precision, label=category_name)
                
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Per-Category Precision-Recall Curves (IoU=0.5)')
                ax.grid(True)
                ax.legend()
                
                # 记录每个类别的PR曲线
                self.writer.add_figure('detection/category_pr_curves', fig, step)
                plt.close(fig)
        except Exception as e:
            print(f"PR曲线绘制失败: {e}")
              
    def log_training_metrics(self, loss_dict, learning_rate, step):
        """记录训练指标
        
        Args:
            loss_dict (dict): 损失字典
            learning_rate (float): 当前学习率
            step (int): 全局步数
        """
        if not self.enabled or self.writer is None:
            return
            
        # 记录总损失和各个损失组件
        for loss_name, loss_value in loss_dict.items():
            if isinstance(loss_value, (int, float)) or (hasattr(loss_value, 'item') and callable(getattr(loss_value, 'item'))):
                loss_value = loss_value.item() if hasattr(loss_value, 'item') else loss_value
                self.writer.add_scalar(f'train/loss/{loss_name}', loss_value, step)
        
        # 记录学习率
        self.writer.add_scalar('train/learning_rate', learning_rate, step)
    
    def log_optimizer_stats(self, optimizer, step):
        """记录优化器统计信息
        
        Args:
            optimizer: PyTorch优化器
            step (int): 全局步数
        """
        if not self.enabled or self.writer is None:
            return
            
        # 记录每个参数组的学习率
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"train/lr_group_{i}", param_group['lr'], step)
    
    def log_model_info(self, model, step=0):
        """记录模型信息
        
        Args:
            model: PyTorch模型
            step (int): 全局步数
        """
        if not self.enabled or self.writer is None:
            return
            
        # 计算模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 记录模型参数信息
        self.writer.add_scalar("model/total_parameters", total_params, step)
        self.writer.add_scalar("model/trainable_parameters", trainable_params, step)
        
        # 记录模型层级结构
        self.log_model_hierarchy(model)
    
    def log_model_hierarchy(self, model):
        """记录模型的层级结构和每层的参数数量
        
        Args:
            model: PyTorch模型
        """
        if not self.enabled or self.writer is None:
            return
            
        try:
            print("记录模型层级结构...")
            # 创建Markdown格式的表格
            markdown = "# 模型层级结构\n\n"
            markdown += "| 层名称 | 类型 | 参数量 | 可训练参数量 | 输出形状 |\n"
            markdown += "| ------ | ---- | ------ | ------------ | -------- |\n"
            
            # 遍历模型的所有命名模块
            for name, module in model.named_modules():
                if name == "":  # 跳过根模块
                    continue
                    
                # 计算该模块的参数量
                params = sum(p.numel() for p in module.parameters())
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                # 获取模块类型
                module_type = module.__class__.__name__
                
                # 尝试获取输出形状（如果模块有output_shape属性）
                output_shape = getattr(module, 'output_shape', '-')
                
                # 添加到表格
                markdown += f"| {name} | {module_type} | {params:,} | {trainable_params:,} | {output_shape} |\n"
            
            # 将Markdown写入TensorBoard
            self.writer.add_text('model/hierarchy', markdown, 0)
            print("模型层级结构已记录到TensorBoard")
            
            # 创建参数分布图
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'parameters/{name}', param, 0)
            
            # 确保立即写入
            self.writer.flush()
            
        except Exception as e:
            print(f"记录模型层级结构失败: {e}")
    
    def flush(self):
        """刷新TensorBoard写入器"""
        if self.enabled and self.writer is not None:
            self.writer.flush()
    
    def close(self):
        """关闭TensorBoard写入器"""
        if self.enabled and self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
            print("TensorBoard写入器已关闭")

    def log_graph(self, model, input_shape=(1, 3, 640, 640)):
        """将模型结构写入TensorBoard的GRAPHS标签页，确保显示完整的模型结构"""
        if not self.enabled or self.writer is None:
            return
        try:
            print("开始记录模型结构到TensorBoard...")
            # 1. 记录整体模型结构
            dummy_input = torch.zeros(input_shape, device=next(model.parameters()).device)
            
            # 尝试不同的trace方法
            try:
                # 方法1：使用use_strict_trace=False（适用于有dict输出的模型）
                self.writer.add_graph(model, dummy_input, use_strict_trace=False)
                print(f"模型结构已写入TensorBoard (use_strict_trace=False)，输入shape={input_shape}")
            except Exception as e1:
                print(f"使用use_strict_trace=False记录模型结构失败: {e1}")
                try:
                    # 方法2：使用verbose=True获取更多调试信息
                    self.writer.add_graph(model, dummy_input, verbose=True)
                    print(f"模型结构已写入TensorBoard (verbose=True)，输入shape={input_shape}")
                except Exception as e2:
                    print(f"使用verbose=True记录模型结构失败: {e2}")
            
            # 2. 记录主要子模块结构（如果存在）
            if hasattr(model, 'backbone'):
                try:
                    self.writer.add_graph(model.backbone, dummy_input, use_strict_trace=False)
                    print("Backbone结构已写入TensorBoard")
                except Exception as e:
                    print(f"记录Backbone结构失败: {e}")
            
            if hasattr(model, 'neck') and hasattr(model.backbone, 'forward_features'):
                try:
                    # 获取backbone的特征输出
                    with torch.no_grad():
                        features = model.backbone.forward_features(dummy_input)
                    self.writer.add_graph(model.neck, features, use_strict_trace=False)
                    print("Neck结构已写入TensorBoard")
                except Exception as e:
                    print(f"记录Neck结构失败: {e}")
            
            if hasattr(model, 'head'):
                try:
                    # 尝试直接记录head结构
                    if hasattr(model.head, 'example_input'):
                        example_input = model.head.example_input
                        self.writer.add_graph(model.head, example_input, use_strict_trace=False)
                        print("Head结构已写入TensorBoard")
                except Exception as e:
                    print(f"记录Head结构失败: {e}")
            
            # 3. 记录文本形式的模型结构
            model_structure = str(model)
            self.writer.add_text('model/structure', model_structure.replace('\n', '  \n'), 0)
            print("模型文本结构已写入TensorBoard")
            
            # 4. 记录模型的参数数量分布
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'model/parameters/{name}', param.data, 0)
            print("模型参数分布已写入TensorBoard")
            
            # 确保立即写入
            self.writer.flush()
            print("模型结构记录完成，已写入TensorBoard")
            
        except Exception as e:
            print(f"写入模型结构到TensorBoard失败: {e}")
            import traceback
            traceback.print_exc()
"""TensorBoard Logger for RTDETR

This module provides TensorBoard logging functionality for the RTDETR project.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError(
        "TensorBoard logging requires PyTorch with TensorBoard support. "
        "Please install tensorboard: pip install tensorboard"
    )

logger = logging.getLogger(__name__)


class TensorboardLogger:
    """TensorBoard Logger for RTDETR.
    
    This class provides a wrapper around TensorBoard's SummaryWriter to log
    training and evaluation metrics during model training.
    """
    
    def __init__(self, log_dir=None, enabled=True):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir (str, optional): Directory where TensorBoard logs will be written.
                If None, logs will be written to 'runs/' directory. Defaults to None.
            enabled (bool, optional): Whether to enable TensorBoard logging. Defaults to True.
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
            print(f"TensorBoard logger initialized at {log_dir}")
    
    def log_metrics(self, metrics, step, prefix=""):
        """Log metrics to TensorBoard.
        
        Args:
            metrics (dict): Dictionary of metrics to log.
            step (int): Global step value to record.
            prefix (str, optional): Prefix for metric names. Defaults to "".
        """
        if not self.enabled or self.writer is None:
            return
            
        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                self.writer.add_scalar(f"{prefix}{k}", v, step)
    
    def log_gradient_norm(self, model, step):
        """Log gradient norms for model parameters.

        Args:
            model: PyTorch model
            step: Global step value to record
        """
        if not self.enabled or self.writer is None:
            return

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.writer.add_scalar('train/gradient_norm', total_norm, step)

    def log_optimizer_stats(self, optimizer, step):
        """Log optimizer statistics.

        Args:
            optimizer: PyTorch optimizer
            step: Global step value to record
        """
        if not self.enabled or self.writer is None:
            return

        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f'train/learning_rate/group_{i}', param_group['lr'], step)
    
    def log_evaluation_metrics(self, coco_eval, step):
        """Log COCO evaluation metrics.

        Args:
            coco_eval: COCO evaluator object
            step: Global step value to record
        """
        if not self.enabled or self.writer is None:
            return

        metrics = {
            'mAP': coco_eval.stats[0],  # AP at IoU=0.50:0.95
            'mAP_50': coco_eval.stats[1],  # AP at IoU=0.50
            'mAP_75': coco_eval.stats[2],  # AP at IoU=0.75
            'mAP_small': coco_eval.stats[3],  # AP for small objects
            'mAP_medium': coco_eval.stats[4],  # AP for medium objects
            'mAP_large': coco_eval.stats[5],  # AP for large objects
            'AR_max_1': coco_eval.stats[6],  # AR for 1 detection
            'AR_max_10': coco_eval.stats[7],  # AR for 10 detections
            'AR_max_100': coco_eval.stats[8],  # AR for 100 detections
            'AR_small': coco_eval.stats[9],  # AR for small objects
            'AR_medium': coco_eval.stats[10],  # AR for medium objects
            'AR_large': coco_eval.stats[11],  # AR for large objects
        }

        # Log all metrics with 'val/' prefix
        for name, value in metrics.items():
            self.writer.add_scalar(f'val/{name}', value, step)

        # Log per-category AP if available
        if hasattr(coco_eval, 'eval') and 'precision' in coco_eval.eval:
            precisions = coco_eval.eval['precision']
            # Take mean over IoU thresholds, recall thresholds, and area ranges
            category_ap = np.mean(precisions, axis=(0, 1, 2))
            for idx, ap in enumerate(category_ap):
                category_id = coco_eval.params.catIds[idx]
                category_name = coco_eval.cocoGt.cats[category_id]['name']
                self.writer.add_scalar(f'val/category_AP/{category_name}', ap, step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram to TensorBoard.
        
        Args:
            tag (str): Data identifier.
            values (torch.Tensor): Values to build histogram.
            step (int): Global step value to record.
        """
        if not self.enabled or self.writer is None:
            return
            
        print(f"Writing histogram with tag {tag} at step {step}")
        self.writer.add_histogram(tag, values, step)
    
    def log_model_info(self, model, step=0):
        """Log model size, parameter count, and structure to TensorBoard.
        
        Args:
            model (torch.nn.Module): The model to log information about.
            step (int, optional): Global step value to record. Defaults to 0.
        """
        if not self.enabled or self.writer is None:
            return
            
        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate model size in MB
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # Log static model info under a separate tag group
        # These metrics only need to be logged once as they don't change during training
        self.writer.add_scalar("model_static_info/total_parameters", total_params, 0)
        self.writer.add_scalar("model_static_info/trainable_parameters", trainable_params, 0)
        self.writer.add_scalar("model_static_info/model_size_MB", model_size, 0)
        
        # Log model structure
        try:
            import torch
            from torch.utils.tensorboard._pytorch_graph import graph
            
            # Determine input shape based on model type
            # RTDETR typically uses input shape (batch_size, 3, height, width)
            input_shape = (1, 3, 640, 640)
            
            # Create a dummy input tensor with the specified shape
            dummy_input = torch.zeros(input_shape, device=next(model.parameters()).device)
            
            # Add graph to TensorBoard with strict=False to handle dict outputs
            self.writer.add_graph(model, dummy_input, use_strict_trace=False)
            
            # Log model components structure
            if hasattr(model, 'backbone') and hasattr(model, 'encoder') and hasattr(model, 'decoder'):
                print("Logging detailed RTDETR model structure...")
                
                # Log backbone structure if possible
                try:
                    self.writer.add_graph(model.backbone, dummy_input, verbose=False)
                except Exception as e:
                    print(f"Could not log backbone structure: {e}")
                
                # Log encoder structure if possible
                try:
                    # Get features from backbone for encoder input
                    with torch.no_grad():
                        backbone_features = model.backbone(dummy_input)
                    self.writer.add_graph(model.encoder, backbone_features, verbose=False)
                except Exception as e:
                    print(f"Could not log encoder structure: {e}")
                
                # Log decoder structure if possible
                try:
                    # Get features from encoder for decoder input
                    with torch.no_grad():
                        backbone_features = model.backbone(dummy_input)
                        encoder_features = model.encoder(backbone_features)
                    self.writer.add_graph(model.decoder, (encoder_features, None), verbose=False)
                except Exception as e:
                    print(f"Could not log decoder structure: {e}")
            
            print(f"Logged model structure to TensorBoard with input shape {input_shape}")
        except Exception as e:
            print(f"Failed to log model structure: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"Logged model info: Size={model_size:.2f}MB, Parameters={total_params:,} (Trainable: {trainable_params:,})")
    
    def log_graph(self, model, input_shape=(1, 3, 640, 640)):
        """Log model graph to TensorBoard.
        
        Args:
            model (torch.nn.Module): The model to log graph for.
            input_shape (tuple, optional): Shape of the input tensor. Defaults to (1, 3, 640, 640).
        """
        if not self.enabled or self.writer is None:
            return
            
        try:
            import torch
            # Create a dummy input tensor with the specified shape
            dummy_input = torch.zeros(input_shape, device=next(model.parameters()).device)
            
            # Add graph to TensorBoard with strict=False to handle dict outputs
            self.writer.add_graph(model, dummy_input, use_strict_trace=False)
            print(f"Model graph logged to TensorBoard with input shape {input_shape}")
            
            # For RTDETR model, also log individual components
            if hasattr(model, 'backbone') and hasattr(model, 'encoder') and hasattr(model, 'decoder'):
                print("Logging detailed model components...")
                
                # Create a separate writer for components to avoid conflicts
                components_writer = self.writer
                
                # Log backbone structure
                try:
                    components_writer.add_graph(model.backbone, dummy_input)
                    print("Backbone graph logged to TensorBoard")
                except Exception as e:
                    print(f"Could not log backbone structure: {e}")
                
                # For more complex components, we'll just log their structure information
                self._log_module_structure(model.backbone, "backbone")
                self._log_module_structure(model.encoder, "encoder")
                self._log_module_structure(model.decoder, "decoder")
        except Exception as e:
            print(f"Failed to log model graph: {e}")
            import traceback
            traceback.print_exc()
              
    def _log_module_structure(self, module, name, max_depth=3):
        """Log the structure of a module to TensorBoard as text.
        
        Args:
            module (torch.nn.Module): The module to log structure for.
            name (str): Name of the module.
            max_depth (int, optional): Maximum depth to traverse. Defaults to 3.
        """
        if not self.enabled or self.writer is None:
            return
            
        try:
            # Generate a text representation of the module structure
            structure_text = self._get_module_structure(module, max_depth=max_depth)
            
            # Add text to TensorBoard
            self.writer.add_text(f"model_structure/{name}", structure_text)
            print(f"Module structure for {name} logged to TensorBoard")
        except Exception as e:
            print(f"Failed to log module structure for {name}: {e}")
    
    def _get_module_structure(self, module, prefix='', depth=0, max_depth=3):
        """Recursively get the structure of a module as a formatted string.
        
        Args:
            module (torch.nn.Module): The module to get structure for.
            prefix (str, optional): Prefix for the current line. Defaults to ''.
            depth (int, optional): Current depth. Defaults to 0.
            max_depth (int, optional): Maximum depth to traverse. Defaults to 3.
            
        Returns:
            str: Formatted string representation of the module structure.
        """
        if depth > max_depth:
            return prefix + "...(max depth reached)\n"
            
        result = ''
        
        # Add current module
        module_name = module.__class__.__name__
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        result += f"{prefix}**{module_name}** (Params: {num_params:,})\n"
        
        # Add children modules
        if depth < max_depth:
            for name, child in module.named_children():
                child_prefix = prefix + '  '
                result += f"{child_prefix}*{name}*: "
                result += self._get_module_structure(child, prefix + '    ', depth + 1, max_depth)
                
        return result
    
    def log_gradient_norm(self, model, step, norm_type=2):
        """记录模型梯度的范数。
        
        Args:
            model (torch.nn.Module): 要记录梯度的模型。
            step (int): 全局步骤值。
            norm_type (int, optional): 范数类型。默认为2（L2范数）。
        """
        if not self.enabled or self.writer is None:
            return
            
        import torch
        total_norm = 0.0
        param_count = 0
        
        # 计算所有参数梯度的总范数
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
                param_count += 1
                
                # 记录每个参数组的梯度范数（可选）
                # self.writer.add_scalar(f"gradients/param_{param_count}_norm", param_norm.item(), step)
        
        if param_count > 0:
            total_norm = total_norm ** (1. / norm_type)
            self.writer.add_scalar("gradients/total_norm", total_norm, step)
            print(f"Logged gradient norm: {total_norm:.4f} at step {step}")
    
    def log_evaluation_metrics(self, coco_eval, step):
        """记录COCO评估指标。
        
        Args:
            coco_eval: COCO评估器对象。
            step (int): 全局步骤值。
        """
        if not self.enabled or self.writer is None or coco_eval is None:
            return
            
        # 记录详细的COCO评估指标
        if hasattr(coco_eval, 'stats'):
            metrics = {
                'mAP': coco_eval.stats[0],  # IoU=0.5:0.95的AP
                'mAP_50': coco_eval.stats[1],  # IoU=0.5的AP
                'mAP_75': coco_eval.stats[2],  # IoU=0.75的AP
                'mAP_small': coco_eval.stats[3],  # 小物体的AP
                'mAP_medium': coco_eval.stats[4],  # 中物体的AP
                'mAP_large': coco_eval.stats[5],  # 大物体的AP
            }
            
            # 记录每个类别的AP
            if hasattr(coco_eval, 'eval') and 'precision' in coco_eval.eval:
                precisions = coco_eval.eval['precision']
                # 计算每个类别的AP (IoU=0.5)
                for category_id in range(precisions.shape[2]):
                    ap = np.mean(precisions[0, :, category_id, 0, -1])
                    self.writer.add_scalar(f"evaluation/category_AP_{category_id}", ap, step)
            
            # 记录精确率、召回率和F1分数
            if hasattr(coco_eval, 'eval') and 'precision' in coco_eval.eval and 'recall' in coco_eval.eval:
                precision = np.mean(coco_eval.eval['precision'])
                recall = np.mean(coco_eval.eval['recall'])
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
                
                metrics.update({
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                })
            
            # 记录IoU分布
            if hasattr(coco_eval, 'ious'):
                iou_values = np.array(list(coco_eval.ious.values()))
                self.writer.add_histogram('evaluation/IoU_distribution', iou_values, step)
            
            # 记录假阳性和假阴性数量
            if hasattr(coco_eval, 'eval') and 'scores' in coco_eval.eval and 'dtIds' in coco_eval.eval:
                fp = np.sum(coco_eval.eval['scores'] == 0)
                fn = len(coco_eval.eval['dtIds']) - np.sum(coco_eval.eval['scores'] > 0)
                metrics.update({
                    'false_positives': fp,
                    'false_negatives': fn
                })
            
            # 记录混淆矩阵
            if hasattr(coco_eval, 'eval') and 'confusion' in coco_eval.eval:
                confusion_matrix = coco_eval.eval['confusion']
                figure = plt.figure(figsize=(10, 10))
                plt.imshow(confusion_matrix, cmap='Blues')
                plt.colorbar()
                plt.title('Confusion Matrix')
                plt.close()
                
                self.writer.add_figure('evaluation/confusion_matrix', figure, step)
            
            # 记录所有指标
            for name, value in metrics.items():
                self.writer.add_scalar(f"evaluation/{name}", value, step)
            
            print(f"Logged enhanced COCO evaluation metrics at step {step}")
    
    def log_optimizer_stats(self, optimizer, step):
        """记录优化器统计信息，如学习率。
        
        Args:
            optimizer (torch.optim.Optimizer): 优化器对象。
            step (int): 全局步骤值。
        """
        if not self.enabled or self.writer is None:
            return
            
        # 记录每个参数组的学习率
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"optimizer/lr_group_{i}", param_group['lr'], step)
        
        # 记录第一个参数组的学习率作为主要学习率
        if optimizer.param_groups:
            self.writer.add_scalar("optimizer/learning_rate", optimizer.param_groups[0]['lr'], step)
            print(f"Logged learning rate: {optimizer.param_groups[0]['lr']:.6f} at step {step}")
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled and self.writer is not None:
            print("Flushing and closing TensorBoard writer")
            if hasattr(self.writer, 'flush'):
                self.writer.flush()
            self.writer.close()
            self.writer = None
            
    def flush(self):
        """Flush the TensorBoard writer."""
        if self.enabled and self.writer is not None:
            print("Flushing TensorBoard writer")
            self.writer.flush()
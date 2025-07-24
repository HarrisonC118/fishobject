"""TensorBoard Logger for Object Detection Models

专注于记录DETR和YOLO比较所需的关键性能指标，包括AP、AR、速度和效率指标。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import json
from collections import defaultdict

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError(
        "TensorBoard logging requires PyTorch with TensorBoard support. "
        "Please install tensorboard: pip install tensorboard"
    )


class TensorboardLogger:
    """专注于目标检测指标的TensorBoard记录器，支持DETR与YOLO比较"""
    
    def __init__(self, log_dir=None, enabled=True, model_name=None):
        """初始化TensorBoard记录器
        
        Args:
            log_dir (str, optional): TensorBoard日志目录，默认为'runs/'
            enabled (bool, optional): 是否启用TensorBoard记录，默认为True
            model_name (str, optional): 模型名称，用于生成表格和图表
        """
        self.enabled = enabled
        self.writer = None
        self.best_metrics = {}  # 记录每个指标的最佳值
        self.best_checkpoints = {}  # 记录每个指标对应的最佳检查点
        self.baseline_metrics = None  # 用于存储基线模型(如YOLO)的指标
        self.model_name = model_name or "SSM-DETR"  # 默认模型名称
        
        if self.enabled:
            if log_dir is None:
                log_dir = Path("runs/")
            else:
                log_dir = Path(log_dir)
                
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            
            # 创建最优模型保存目录
            self.best_models_dir = log_dir / "best_models"
            os.makedirs(self.best_models_dir, exist_ok=True)
            
            # 加载之前保存的最佳指标（如果存在）
            self.best_metrics_file = log_dir / "best_metrics.json"
            if self.best_metrics_file.exists():
                try:
                    with open(self.best_metrics_file, 'r') as f:
                        saved_data = json.load(f)
                        self.best_metrics = saved_data.get('metrics', {})
                        self.best_checkpoints = saved_data.get('checkpoints', {})
                        # 如果保存了模型名称，则恢复它
                        if 'model_name' in saved_data:
                            self.model_name = saved_data['model_name']
                except Exception as e:
                    print(f"无法加载之前的最佳指标: {e}")
            
            print(f"TensorBoard记录器已初始化，日志目录：{log_dir}，模型名称：{self.model_name}")
    
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
    
    def log_detection_metrics(self, coco_eval, step, model_checkpoint=None):
        """记录目标检测关键指标并跟踪最佳性能

        Args:
            coco_eval: COCO评估器对象
            step (int): 全局步数
            model_checkpoint (str, optional): 当前模型检查点路径，用于保存最佳模型
        """
        if not self.enabled or self.writer is None or coco_eval is None:
            return

        # 1. 核心评估指标 - DETR与YOLO比较必备
        ap_metrics = {
            'mAP': coco_eval.stats[0],        # IoU=0.5:0.95的AP (COCO主要指标)
            'mAP_50': coco_eval.stats[1],     # IoU=0.5的AP (YOLO主要指标)
            'mAP_75': coco_eval.stats[2],     # IoU=0.75的AP (严格评估)
            'mAP_small': coco_eval.stats[3],  # 小物体的AP
            'mAP_medium': coco_eval.stats[4], # 中物体的AP
            'mAP_large': coco_eval.stats[5],  # 大物体的AP
        }
        
        # 2. 召回率指标
        ar_metrics = {
            'AR_max_100': coco_eval.stats[8], # 每张图像最多100个检测结果的AR (主要AR指标)
            'AR_small': coco_eval.stats[9],   # 小物体的AR
            'AR_medium': coco_eval.stats[10], # 中物体的AR
            'AR_large': coco_eval.stats[11],  # 大物体的AR
        }
        
        # 3. 记录AP指标（主要指标）
        for name, value in ap_metrics.items():
            self.writer.add_scalar(f'detection/AP/{name}', value, step)
            
            # 检查是否为新的最佳值，并更新
            if model_checkpoint and (name not in self.best_metrics or value > self.best_metrics[name]):
                self.best_metrics[name] = value
                self.best_checkpoints[name] = model_checkpoint
                print(f"✓ 新的最佳 {name}: {value:.4f}，检查点: {model_checkpoint}")
                
                # 保存最佳指标文件
                self._save_best_metrics()
        
        # 4. 记录AR指标
        for name, value in ar_metrics.items():
            self.writer.add_scalar(f'detection/AR/{name}', value, step)
        
        # 5. 计算并记录F1分数（DETR和YOLO比较的重要指标）
        precision_50 = ap_metrics['mAP_50']
        recall = ar_metrics['AR_max_100']
        f1_score = 2 * (precision_50 * recall) / (precision_50 + recall + 1e-6)
        
        self.writer.add_scalar('detection/metrics/precision', precision_50, step)
        self.writer.add_scalar('detection/metrics/recall', recall, step)
        self.writer.add_scalar('detection/metrics/f1_score', f1_score, step)
        
        # 检查F1分数是否为最佳
        if model_checkpoint and ('f1_score' not in self.best_metrics or f1_score > self.best_metrics['f1_score']):
            self.best_metrics['f1_score'] = f1_score
            self.best_checkpoints['f1_score'] = model_checkpoint
            print(f"✓ 新的最佳 F1分数: {f1_score:.4f}，检查点: {model_checkpoint}")
            
            # 保存最佳指标文件
            self._save_best_metrics()
        
        # 6. 记录类别AP
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
        
        # 7. 可视化PR曲线
        if hasattr(coco_eval, 'eval') and 'precision' in coco_eval.eval:
            self._log_pr_curve(coco_eval, step)
        
        # 创建汇总表格
        summary_markdown = f"## 检测性能指标 (Epoch {step})\n\n"
        summary_markdown += "| 指标 | 当前值 | 最佳值 | 最佳检查点 |\n"
        summary_markdown += "|------|-------|-------|----------|\n"
        
        for metric_name in ['mAP', 'mAP_50', 'mAP_75', 'f1_score']:
            current_value = ap_metrics.get(metric_name, f1_score if metric_name == 'f1_score' else 0)
            best_value = self.best_metrics.get(metric_name, 0)
            best_ckpt = self.best_checkpoints.get(metric_name, "-")
            
            # 提取检查点文件名
            if isinstance(best_ckpt, Path):
                best_ckpt = best_ckpt.name
            elif isinstance(best_ckpt, str):
                best_ckpt = Path(best_ckpt).name
                
            summary_markdown += f"| {metric_name} | {current_value:.4f} | {best_value:.4f} | {best_ckpt} |\n"
        
        self.writer.add_text('detection/summary', summary_markdown, step)
        
        # 确保立即写入数据
        self.flush()
    
    def _save_best_metrics(self):
        """保存最佳指标到JSON文件"""
        if not self.best_metrics_file:
            return
            
        try:
            # 转换Path对象为字符串
            serializable_checkpoints = {}
            for k, v in self.best_checkpoints.items():
                if isinstance(v, Path):
                    serializable_checkpoints[k] = str(v)
                else:
                    serializable_checkpoints[k] = v
                    
            data = {
                'metrics': self.best_metrics,
                'checkpoints': serializable_checkpoints,
                'model_name': self.model_name  # 保存模型名称
            }
            with open(self.best_metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"保存最佳指标失败: {e}")
    
    def save_best_model(self, model_state_dict, metric_name='mAP', model_info=None):
        """保存指定指标的最佳模型
        
        Args:
            model_state_dict: 模型状态字典
            metric_name (str): 指标名称，默认为'mAP'
            model_info (dict): 模型相关信息
        """
        if not self.enabled or metric_name not in self.best_checkpoints:
            return
            
        try:
            save_path = self.best_models_dir / f"best_{metric_name}_model.pth"
            
            # 准备保存数据
            save_data = {
                'model': model_state_dict,
                'metric_value': self.best_metrics.get(metric_name, 0),
                'info': model_info or {}
            }
            
            # 保存模型
            torch.save(save_data, save_path)
            print(f"最佳 {metric_name} 模型已保存至 {save_path}")
        except Exception as e:
            print(f"保存最佳模型失败: {e}")
    
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
    
    def log_efficiency_metrics(self, metrics, step):
        """记录效率指标，用于DETR与YOLO比较
        
        Args:
            metrics (dict): 包含效率指标的字典
            step (int): 全局步数
        """
        if not self.enabled or self.writer is None:
            return
            
        # 记录FPS
        if 'fps' in metrics:
            self.writer.add_scalar('efficiency/fps', metrics['fps'], step)
            # 保存到最佳指标中，用于后续导出表格
            self.best_metrics['fps'] = metrics['fps']
            
        # 记录参数量
        if 'params' in metrics:
            self.writer.add_scalar('efficiency/params_M', metrics['params'], step)
            self.best_metrics['params'] = metrics['params']
            
        # 记录FLOPs
        if 'flops' in metrics:
            self.writer.add_scalar('efficiency/gflops', metrics['flops'] / 1e9, step)
            self.best_metrics['flops'] = metrics['flops'] / 1e9
            
        # 记录延迟
        if 'latency' in metrics:
            self.writer.add_scalar('efficiency/latency_ms', metrics['latency'] * 1000, step)
            self.best_metrics['latency'] = metrics['latency']
            
        # 记录内存使用
        if 'memory' in metrics:
            self.writer.add_scalar('efficiency/memory_MB', metrics['memory'], step)
            self.best_metrics['memory'] = metrics['memory']
        
        # 创建速度-精度权衡图
        if 'fps' in metrics and 'mAP' in self.best_metrics:
            # 在同一图表上添加新的点
            speed_vs_accuracy = {
                'FPS': metrics['fps'],
                'mAP': self.best_metrics.get('mAP', 0)
            }
            self.writer.add_scalars('comparison/speed_vs_accuracy', speed_vs_accuracy, step)

        # 保存最佳指标
        self._save_best_metrics()
            
    def compare_with_baseline(self, current_metrics, baseline_metrics, step):
        """与基线模型(如YOLO)进行比较
        
        Args:
            current_metrics (dict): 当前模型指标
            baseline_metrics (dict): 基线模型指标
            step (int): 全局步数
        """
        if not self.enabled or self.writer is None:
            return
        
        # 保存基线指标用于后续表格导出
        self.baseline_metrics = baseline_metrics
            
        # 计算相对差异
        diff_metrics = {}
        for k in current_metrics:
            if k in baseline_metrics and isinstance(current_metrics[k], (int, float)) and isinstance(baseline_metrics[k], (int, float)):
                # 计算相对改进百分比
                if baseline_metrics[k] != 0:
                    rel_diff = (current_metrics[k] - baseline_metrics[k]) / baseline_metrics[k] * 100
                    diff_metrics[f"{k}_diff"] = rel_diff
                    
        # 记录比较指标
        self.log_metrics(diff_metrics, step, prefix='comparison/diff_')
        
        # 创建比较条形图
        if len(diff_metrics) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            keys = list(diff_metrics.keys())
            values = [diff_metrics[k] for k in keys]
            
            bars = ax.bar(keys, values)
            
            # 为正负值设置不同颜色
            for i, v in enumerate(values):
                bars[i].set_color('green' if v >= 0 else 'red')
                
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_ylabel('相对改进 (%)')
            ax.set_title('与基线模型的比较')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top')
            
            self.writer.add_figure('comparison/vs_baseline', fig, step)
            plt.close(fig)
    
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
        self.writer.add_scalar("model/total_parameters_M", total_params / 1e6, step)
        self.writer.add_scalar("model/trainable_parameters_M", trainable_params / 1e6, step)
        
        # 保存到最佳指标中，用于后续导出表格
        self.best_metrics['params'] = total_params / 1e6
    
    def generate_paper_table(self, model_name, efficiency_metrics=None, detection_metrics=None, save_path=None, compare_models=None):
        """生成适用于论文的性能指标表格
        
        Args:
            model_name (str): 当前模型名称
            efficiency_metrics (dict, optional): 效率指标，包含params、flops、fps等
            detection_metrics (dict, optional): 检测指标，包含mAP等，若None则使用best_metrics
            save_path (str, optional): 保存路径，包括.md、.tex或.csv格式
            compare_models (dict, optional): 其他模型的性能指标，用于比较
        """
        if not self.enabled:
            return
        
        try:
            import pandas as pd
        except ImportError:
            print("警告: pandas 未安装，无法生成性能指标表格。请运行 'pip install pandas' 安装。")
            print("继续训练，但不会生成表格...")
            return None
            
        import os
        
        # 使用传入的检测指标或最佳指标
        if detection_metrics is None:
            detection_metrics = self.best_metrics
        
        # 如果没有传入效率指标，尝试构建空字典
        if efficiency_metrics is None:
            efficiency_metrics = {}
        
        # 构建当前模型的数据行
        data = {
            'Model': [model_name],
            '#Params (M)': [efficiency_metrics.get('params', detection_metrics.get('params', 'N/A'))],
            'GFLOPs': [efficiency_metrics.get('flops', detection_metrics.get('flops', 'N/A'))],
            'FPS (bs=1)': [efficiency_metrics.get('fps', detection_metrics.get('fps', 'N/A'))],
            'AP': [detection_metrics.get('mAP', 'N/A')],  # mAP@0.5:0.95
            'AP50': [detection_metrics.get('mAP_50', 'N/A')],
            'AP75': [detection_metrics.get('mAP_75', 'N/A')],
            'APS': [detection_metrics.get('mAP_small', 'N/A')],
            'APM': [detection_metrics.get('mAP_medium', 'N/A')],
            'APL': [detection_metrics.get('mAP_large', 'N/A')],
        }
        
        # 如果提供了比较模型，添加它们的数据行
        if compare_models:
            for model_name, metrics in compare_models.items():
                data['Model'].append(model_name)
                data['#Params (M)'].append(metrics.get('params', 'N/A'))
                data['GFLOPs'].append(metrics.get('flops', 'N/A'))
                data['FPS (bs=1)'].append(metrics.get('fps', 'N/A'))
                data['AP'].append(metrics.get('mAP', 'N/A'))
                data['AP50'].append(metrics.get('mAP_50', 'N/A'))
                data['AP75'].append(metrics.get('mAP_75', 'N/A'))
                data['APS'].append(metrics.get('mAP_small', 'N/A'))
                data['APM'].append(metrics.get('mAP_medium', 'N/A'))
                data['APL'].append(metrics.get('mAP_large', 'N/A'))
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 生成不同格式的表格
        formats = {}
        
        # 1. Markdown格式 (适合GitHub和大多数Markdown编辑器)
        formats['markdown'] = df.to_markdown(index=False)
        
        # 2. LaTeX格式 (适合学术论文)
        formats['latex'] = df.to_latex(index=False, escape=False)
        
        # 3. CSV格式 (适合数据处理)
        formats['csv'] = df.to_csv(index=False)
        
        # 4. HTML格式 (适合网页展示)
        formats['html'] = df.to_html(index=False)
        
        # 保存到文件
        if save_path:
            base_path, ext = os.path.splitext(save_path)
            if not ext:
                # 如果没有指定扩展名，保存所有格式
                for fmt, content in formats.items():
                    with open(f"{base_path}.{fmt}", "w") as f:
                        f.write(content)
            else:
                # 根据扩展名保存特定格式
                ext = ext[1:]  # 去掉点号
                if ext == 'md':
                    ext = 'markdown'
                if ext in formats:
                    with open(save_path, "w") as f:
                        f.write(formats[ext])
                else:
                    print(f"不支持的文件格式: {ext}")
        
        # 记录到TensorBoard
        try:
            # 创建Markdown表格记录到TensorBoard
            self.writer.add_text('paper/metrics_table', formats['markdown'], 0)
            print(f"性能指标表已记录到TensorBoard")
        except Exception as e:
            print(f"记录表格到TensorBoard失败: {e}")
        
        # 打印到控制台
        print("\n===== 性能指标表 (适用于论文) =====")
        print(formats['markdown'])
        print("==================================\n")
        
        return formats
    
    def export_best_metrics_table(self, save_path=None, include_baseline=False, baseline_name="YOLO"):
        """导出最佳指标的表格，适合论文使用
        
        Args:
            save_path (str, optional): 保存路径，如果为None则只打印
            include_baseline (bool): 是否包含基线模型(如YOLO)的比较
            baseline_name (str): 基线模型名称
        """
        # 从最佳指标构建检测指标字典
        detection_metrics = {
            'mAP': self.best_metrics.get('mAP', 'N/A'),
            'mAP_50': self.best_metrics.get('mAP_50', 'N/A'),
            'mAP_75': self.best_metrics.get('mAP_75', 'N/A'),
            'mAP_small': self.best_metrics.get('mAP_small', 'N/A'),
            'mAP_medium': self.best_metrics.get('mAP_medium', 'N/A'),
            'mAP_large': self.best_metrics.get('mAP_large', 'N/A')
        }
        
        # 获取效率指标
        # 尝试从保存的最佳指标中提取效率指标
        efficiency_metrics = {}
        for metric in ['params', 'flops', 'fps', 'latency']:
            if metric in self.best_metrics:
                efficiency_metrics[metric] = self.best_metrics[metric]
        
        # 如果包含基线模型
        compare_models = None
        if include_baseline and self.baseline_metrics is not None:
            compare_models = {baseline_name: self.baseline_metrics}
        
        # 生成表格
        result = self.generate_paper_table(
            self.model_name,  # 使用实例化时提供的模型名称
            efficiency_metrics=efficiency_metrics,
            detection_metrics=detection_metrics,
            save_path=save_path,
            compare_models=compare_models
        )
        
        return result
    
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
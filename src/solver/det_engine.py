"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
import time
from typing import Iterable

import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    # RTDETR开始训练
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)
    # tensorboard_logger = kwargs.get('tensorboard_logger', None) # Removed tensorboard_logger

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()  # 记录开始时间
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        # ema 
        if ema is not None:
            ema.update(model)
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        if not math.isfinite(loss_value):
            print("Loss is {}. Stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 计算并记录梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = (total_norm ** 0.5) if total_norm > 0 else 0.0 # ensure total_norm is not nan if all grads are None
        metric_logger.update(grad_norm=total_norm)

        # 统计并记录FPS
        batch_time = time.time() - start_time
        fps = samples.shape[0] / batch_time if batch_time > 0 else 0
        metric_logger.update(fps=fps)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir, **kwargs):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    
    epoch = kwargs.get('epoch', 0)

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
        # 使用 metric_logger 记录评估指标
        for iou_type in iou_types:
            if iou_type in coco_evaluator.coco_eval:
                eval_stats = coco_evaluator.coco_eval[iou_type].stats
                metric_logger.update(**{
                    f'{iou_type}_mAP': eval_stats[0],
                    f'{iou_type}_mAP_50': eval_stats[1],
                    f'{iou_type}_mAP_75': eval_stats[2],
                    f'{iou_type}_mAP_small': eval_stats[3],
                    f'{iou_type}_mAP_medium': eval_stats[4],
                    f'{iou_type}_mAP_large': eval_stats[5],
                    f'{iou_type}_AR_1': eval_stats[6],
                    f'{iou_type}_AR_10': eval_stats[7],
                    f'{iou_type}_AR_100': eval_stats[8],
                    f'{iou_type}_AR_small': eval_stats[9],
                    f'{iou_type}_AR_medium': eval_stats[10],
                    f'{iou_type}_AR_large': eval_stats[11]
                })

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # if coco_evaluator is not None:
    #     if 'bbox' in coco_evaluator.coco_eval:
    #         stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    #     if 'segm' in coco_evaluator.coco_eval:
    #         stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # if panoptic_res is not None:
    #     stats['panoptic_eval'] = panoptic_res

    # # Log evaluation metrics to TensorBoard if enabled
    # tensorboard_logger = kwargs.get('tensorboard_logger', None)
    # if tensorboard_logger is not None and tensorboard_logger.enabled:
    #     if coco_evaluator is not None:
    #         tensorboard_logger.log_evaluation_metrics(coco_evaluator.coco_eval['bbox'], epoch) # Assuming 'bbox' is primary
    #         tensorboard_logger.writer.flush()
    #         print(f"Epoch {epoch}: COCO评估指标已写入TensorBoard。")
            
    # 获取当前epoch
    epoch = kwargs.get('epoch', 0)
    
    # Log evaluation metrics to TensorBoard if enabled
    tensorboard_logger = kwargs.get('tensorboard_logger', None)
    if tensorboard_logger is not None and tensorboard_logger.enabled:
        if coco_evaluator is not None:
            print(f"Epoch {epoch}: 开始记录评估指标到TensorBoard...")
            for iou_type in iou_types:
                if iou_type in coco_evaluator.coco_eval:
                    # 使用新的log_detection_metrics函数记录详细的检测指标
                    tensorboard_logger.log_detection_metrics(coco_evaluator.coco_eval[iou_type], epoch)
            
            # 确保数据立即写入磁盘
            tensorboard_logger.flush()
            print(f"Epoch {epoch}: 目标检测评估指标已写入TensorBoard")
            
            # 打印关键指标
            if 'bbox' in coco_evaluator.coco_eval:
                stats = coco_evaluator.coco_eval['bbox'].stats
                print(f"评估结果: mAP={stats[0]:.4f}, mAP@0.5={stats[1]:.4f}, mAP@0.75={stats[2]:.4f}")
                print(f"         mAP_small={stats[3]:.4f}, mAP_medium={stats[4]:.4f}, mAP_large={stats[5]:.4f}")
                print(f"         AR@100={stats[8]:.4f}")
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in coco_evaluator.coco_eval:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in coco_evaluator.coco_eval:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator




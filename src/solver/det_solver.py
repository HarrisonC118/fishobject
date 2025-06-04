'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()
        # 又把传进来的配置文件变成args了，不理解，脱裤子放屁
        args = self.cfg 
        # 获取参数量，看看模型一共有多少参数要被训练
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        # 记录参数量到TensorBoard
        if hasattr(self, 'tensorboard_logger') and self.tensorboard_logger is not None:
            self.tensorboard_logger.log_metrics({'Params': n_parameters}, 0, prefix='model/')
        print(f'Number of trainable parameters: {n_parameters}')

        # 统计GFLOPs（需torchprofile或thop等工具，若无则占位）
        try:
            from thop import profile
            dummy_input_gflops = torch.zeros((1, 3, 640, 640), device=next(self.model.parameters()).device)
            flops, params_profile = profile(self.model, inputs=(dummy_input_gflops,), verbose=False)
            gflops = flops / 1e9
            if hasattr(self, 'tensorboard_logger') and self.tensorboard_logger is not None:
                self.tensorboard_logger.log_metrics({'GFLOPs': gflops}, 0, prefix='model/')
            print(f"Model GFLOPs: {gflops:.2f}")
        except Exception as e:
            print(f"GFLOPs calculation failed: {e}")
        
        if hasattr(self, 'tensorboard_logger') and self.tensorboard_logger is not None:
            self.tensorboard_logger.log_model_info(self.model)
        # Calculate and print other model static info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model total parameters: {total_params}")
        model_size_MB = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        print(f"Model size (MB): {model_size_MB:.2f}")

        # Log model graph to TensorBoard if tensorboard_logger is available
        if hasattr(self, 'tensorboard_logger') and self.tensorboard_logger is not None and self.tensorboard_logger.enabled:
            try:
                # Ensure dummy_input is defined for graph logging
                if 'dummy_input_gflops' in locals():
                    dummy_input_graph = dummy_input_gflops
                else:
                    dummy_input_graph = torch.zeros((1, 3, 640, 640), device=next(self.model.parameters()).device)
                
                print("Logging model graph to TensorBoard...")
                self.tensorboard_logger.log_graph(self.model, input_shape=(1, 3, 640, 640))
                self.tensorboard_logger.writer.flush() # Flush after adding graph
            except Exception as e:
                print(f"Failed to log model graph to TensorBoard: {e}")

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }
        
        # 用于跟踪最佳检查点的列表，最多保存5个
        best_checkpoints = []  # 格式: [(性能指标, epoch, 检查点路径)]

        start_time = time.time()
        # !!开始训练!!
        for epoch in range(self.last_epoch + 1, args.epoches):
            # 分布式，不用管
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            # 正式开始第一轮训练
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)
            # 更新学习率
            self.lr_scheduler.step()
            
            if self.output_dir:
                # 始终保存最新的检查点
                latest_checkpoint_path = self.output_dir / 'checkpoint.pth'
                dist.save_on_master(self.state_dict(epoch), latest_checkpoint_path)
                
                # 每隔checkpoint_step保存一个检查点
                # if (epoch + 1) % args.checkpoint_step == 0:
                #     checkpoint_path = self.output_dir / f'checkpoint{epoch:04}.pth'
                #     dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir,
                epoch=epoch, tensorboard_logger=self.tensorboard_logger
            )

            # 更新最佳统计信息
            for k in test_stats.keys():
                v = test_stats[k]
                if isinstance(v, (list, tuple)):
                    value = v[0]
                else:
                    value = v
                if k in best_stat:
                    best_stat['epoch'] = epoch if value > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], value)
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = value
            print('best_stat: ', best_stat)
            
            # 保存性能最好的检查点（基于coco_eval_bbox指标）
            if 'coco_eval_bbox' in test_stats and dist.is_main_process():
                # 使用mAP作为性能指标（COCO评估的第一个指标通常是mAP）
                performance = test_stats['coco_eval_bbox'][0] if isinstance(test_stats['coco_eval_bbox'], (list, tuple)) else test_stats['coco_eval_bbox']
                checkpoint_path = self.output_dir / f'checkpoint_best_{epoch:04}.pth'
                
                # 保存当前检查点
                dist.save_on_master(self.state_dict(epoch), checkpoint_path)
                
                # 将当前检查点添加到最佳检查点列表
                best_checkpoints.append((performance, epoch, checkpoint_path))
                
                # 按性能指标排序（降序）
                best_checkpoints.sort(reverse=True)
                
                # 只保留前5个最佳检查点
                if len(best_checkpoints) > 5:
                    # 删除性能最差的检查点文件
                    _, _, worst_checkpoint_path = best_checkpoints.pop()
                    if worst_checkpoint_path.exists() and worst_checkpoint_path != latest_checkpoint_path:
                        worst_checkpoint_path.unlink()
                
                print(f"Best checkpoints: {[(p, e) for p, e, _ in best_checkpoints]}")


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)
            
            # Log metrics to TensorBoard
            if hasattr(self, 'tensorboard_logger') and self.tensorboard_logger is not None and self.tensorboard_logger.enabled:
                # 记录训练指标
                self.tensorboard_logger.log_metrics(train_stats, epoch, prefix='train/')
                
                # 记录评估指标
                self.tensorboard_logger.log_metrics(test_stats, epoch, prefix='val/')
                
                # 记录学习率
                self.tensorboard_logger.log_optimizer_stats(self.optimizer, epoch)
                
                # 记录COCO评估指标
                if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                    self.tensorboard_logger.log_evaluation_metrics(coco_evaluator.coco_eval["bbox"], epoch)
                
                # 记录FPS
                if 'fps' in train_stats:
                    self.tensorboard_logger.log_metrics({'FPS': train_stats['fps']}, epoch, prefix='performance/')
                
                self.tensorboard_logger.writer.flush()
                print(f"Epoch {epoch}: 训练和评估指标已写入 TensorBoard。")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir,
                tensorboard_logger=self.tensorboard_logger)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return

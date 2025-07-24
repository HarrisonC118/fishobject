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
        self.train()

        # get base dataset
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        # 获取模型参数数量，优先使用dist.get_n_params函数，如果不存在则手动计算
        try:
            n_parameters = dist.get_n_params(self.model)
        except AttributeError:
            # 兼容旧版本，手动计算参数量
            model = dist.de_parallel(self.model)  # 确保是实际模型而非DDP包装
            n_parameters = sum(p.numel() for p in model.parameters())
            print(f'模型参数总量: {n_parameters:,}')
            
        print('number of params:', n_parameters)
        
        # 兼容拼写差异：epoches vs epochs
        if not hasattr(self.cfg, 'epochs') and hasattr(self.cfg, 'epoches'):
            self.cfg.epochs = self.cfg.epoches
            print(f"注意：配置文件使用了'epoches'拼写，已自动转换为'epochs'={self.cfg.epochs}")
            
        # 确保eval_interval参数存在
        if not hasattr(self.cfg, 'eval_interval'):
            self.cfg.eval_interval = 1  # 默认每个epoch评估一次
            print(f"注意：未设置'eval_interval'，默认为{self.cfg.eval_interval}")
        
        # 确保test_only参数存在    
        if not hasattr(self.cfg, 'test_only'):
            self.cfg.test_only = False
            print("注意：未设置'test_only'，默认为False")

        # 用于保存最佳检查点的记录
        best_checkpoints = []
        latest_checkpoint_path = None  # 最新的检查点路径

        # NOTE 
        # pytorch loss NaN https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # Avoid reintializing initialization cells
        # print('self.last_epoch = ', self.last_epoch)
        start_time = time.time()
        for epoch in range(self.last_epoch + 1, self.cfg.epochs + 1):
            # if self.cfg.distributed:
            #     self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                model=self.model,
                criterion=self.criterion,
                data_loader=self.train_dataloader,
                optimizer=self.optimizer,
                device=self.device,
                epoch=epoch,
                scaler=self.scaler,
                max_norm=self.cfg.clip_max_norm,
                cfg=self.cfg,  # 传递配置对象
            )
            
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # ema
            if self.ema:
                self.ema.update(self.model)  # 传递模型参数给update方法

            if self.output_dir:
                checkpoint_path = self.output_dir / f'checkpoint{epoch:04}.pth'
                latest_checkpoint_path = checkpoint_path

            # eval
            if not self.cfg.test_only and epoch % self.cfg.eval_interval == 0:
                module = self.ema.module if self.ema else self.model
                
                # 计算效率指标(FPS等)用于与YOLO比较
                efficiency_metrics = {}
                if epoch % (self.cfg.eval_interval * 5) == 0:  # 减少频率以节省资源
                    try:
                        # 测量FPS
                        with torch.no_grad():
                            dummy_input = torch.randn(1, 3, 640, 640, device=self.device)
                            
                            # 预热
                            for _ in range(10):
                                _ = module(dummy_input)
                                
                            # 计时
                            torch.cuda.synchronize()
                            start = time.time()
                            iterations = 50
                            for _ in range(iterations):
                                _ = module(dummy_input)
                            torch.cuda.synchronize()
                            end = time.time()
                            
                            # 计算FPS
                            fps = iterations / (end - start)
                            efficiency_metrics['fps'] = fps
                            
                            # 计算参数量(百万)
                            params = sum(p.numel() for p in module.parameters()) / 1e6
                            efficiency_metrics['params'] = params
                            
                            # 延迟(秒)
                            efficiency_metrics['latency'] = (end - start) / iterations
                            
                            print(f"效率指标: FPS={fps:.2f}, 参数量={params:.2f}M, 延迟={efficiency_metrics['latency']*1000:.2f}ms")
                    except Exception as e:
                        print(f"计算效率指标失败: {e}")
                
                # 从环境变量或配置中获取YOLO基线指标
                baseline_metrics = None
                if hasattr(self.cfg, 'baseline_metrics'):
                    baseline_metrics = self.cfg.baseline_metrics
                
                # 评估模型，传入当前检查点路径和效率指标
                test_stats, coco_evaluator = evaluate(
                    module, self.criterion, self.postprocessor,
                    self.val_dataloader, base_ds, self.device, self.output_dir,
                    epoch=epoch,
                    tensorboard_logger=self.tensorboard_logger,
                    checkpoint_path=checkpoint_path,
                    efficiency_metrics=efficiency_metrics,
                    baseline_metrics=baseline_metrics
                )
                
                # 保存模型并计算性能指标
                if self.output_dir and dist.is_main_process():
                    # 确定性能指标 (使用mAP@0.5:0.95作为主要指标)
                    if coco_evaluator and "bbox" in coco_evaluator.coco_eval:
                        performance = coco_evaluator.coco_eval["bbox"].stats[0]
                        
                        # 如果是F1最优的模型，保存它
                        mAP50 = coco_evaluator.coco_eval["bbox"].stats[1]  # IoU=0.5的AP
                        AR = coco_evaluator.coco_eval["bbox"].stats[8]     # AR_max_100
                        F1 = 2 * (mAP50 * AR) / (mAP50 + AR + 1e-6)
                        
                        if hasattr(self, 'tensorboard_logger') and self.tensorboard_logger is not None:
                            # 保存最佳mAP和F1模型
                            model_state = self.state_dict(epoch)
                            model_info = {
                                'epoch': epoch,
                                'stats': {
                                    'mAP': performance,
                                    'mAP_50': mAP50,
                                    'AR': AR,
                                    'F1': F1
                                }
                            }
                            
                            # 保存最佳mAP模型
                            self.tensorboard_logger.save_best_model(
                                model_state, metric_name='mAP', model_info=model_info
                            )
                            
                            # 保存最佳F1模型
                            self.tensorboard_logger.save_best_model(
                                model_state, metric_name='f1_score', model_info=model_info
                            )
                            
                            # 保存最佳mAP_50模型 (与YOLO比较最重要的指标)
                            self.tensorboard_logger.save_best_model(
                                model_state, metric_name='mAP_50', model_info=model_info
                            )
                    else:
                        performance = 0
                
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
                print(f"Epoch {epoch}: 开始记录指标到TensorBoard...")
                
                # 记录训练损失和学习率
                if 'loss' in train_stats:
                    loss_dict = {'total_loss': train_stats['loss']}
                    # 提取其他损失组件
                    for k, v in train_stats.items():
                        if k.startswith('loss_') and isinstance(v, (float, int)):
                            loss_dict[k] = v
                    
                    # 获取当前学习率
                    current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0
                    self.tensorboard_logger.log_training_metrics(loss_dict, current_lr, epoch)
                
                # 记录其他训练指标
                self.tensorboard_logger.log_metrics({k: v for k, v in train_stats.items() if not k.startswith('loss_')}, epoch, prefix='train/')
                
                # 记录评估指标
                self.tensorboard_logger.log_metrics(test_stats, epoch, prefix='val/')
                
                # 记录优化器信息
                self.tensorboard_logger.log_optimizer_stats(self.optimizer, epoch)
                
                # 记录性能指标
                if 'fps' in train_stats:
                    self.tensorboard_logger.log_metrics({'FPS': train_stats['fps']}, epoch, prefix='performance/')
                
                # 确保数据被写入磁盘
                self.tensorboard_logger.flush()
                print(f"Epoch {epoch}: 训练和评估指标已写入TensorBoard")
                
                # 打印当前F1值等关键指标
                if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                    mAP50 = coco_evaluator.coco_eval["bbox"].stats[1]  # IoU=0.5的AP
                    AR = coco_evaluator.coco_eval["bbox"].stats[8]  # AR_max_100
                    F1 = 2 * (mAP50 * AR) / (mAP50 + AR + 1e-6)
                    print(f"Epoch {epoch} 关键指标: mAP@0.5={mAP50:.4f}, AR={AR:.4f}, F1={F1:.4f}")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # 训练结束后，生成并保存性能指标表格（适用于论文）
        if self.output_dir and dist.is_main_process() and hasattr(self, 'tensorboard_logger') and self.tensorboard_logger is not None:
            print("\n正在生成性能指标表格（用于论文）...")
            # 生成基本性能表格
            table_path = self.output_dir / "performance_metrics"
            self.tensorboard_logger.export_best_metrics_table(save_path=table_path, include_baseline=True)
            
            # 生成与YOLO的详细比较表格
            # 如果配置中有YOLO相关指标，则添加到比较中
            yolo_models = {}
            if hasattr(self.cfg, 'comparison_models') and self.cfg.comparison_models:
                for model_name, metrics in self.cfg.comparison_models.items():
                    yolo_models[model_name] = metrics
                    
                # 如果有对比模型，生成详细比较表格
                if yolo_models:
                    compare_path = self.output_dir / "model_comparison"
                    # 使用配置中的模型名称，如果存在的话
                    model_name = getattr(self.cfg, 'model_name', "SSM-DETR (Ours)")
                    self.tensorboard_logger.generate_paper_table(
                        model_name,
                        detection_metrics=self.tensorboard_logger.best_metrics,
                        save_path=compare_path,
                        compare_models=yolo_models
                    )
                    print(f"与其他模型的详细比较表格已保存至 {compare_path}.*")


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir,
                tensorboard_logger=self.tensorboard_logger)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
            
            # 验证后生成性能表格
            if dist.is_main_process() and hasattr(self, 'tensorboard_logger') and self.tensorboard_logger is not None:
                table_path = self.output_dir / "val_performance_metrics"
                self.tensorboard_logger.export_best_metrics_table(save_path=table_path)
        
        return

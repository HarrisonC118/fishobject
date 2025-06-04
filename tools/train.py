"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import src.misc.tensorboard_logger as tb_logger


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    # Initialize TensorBoard Logger before creating solver
    # This will be used by the solver to log metrics during training
    cfg.tensorboard_logger = tb_logger.TensorboardLogger(log_dir=args.log_dir)
    # 实例化DetSolver这个类，并且把cfg放进得到的solver对象里面
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()
        
        # Close the TensorBoard logger after training
        cfg.tensorboard_logger.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for TensorBoard logs')
    args = parser.parse_args()

    main(args)

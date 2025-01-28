import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), '..', '..'))

import argparse


# Combine hyper parameters from *.yaml into arguments
def combine_hyper_cfg_into_args(args, cfg_path):
    from hyper_config.file_config import get_config
    cfg = get_config(cfg_path)
    args.spatial_mode = cfg.spatial_mode
    args.hyper_head = cfg.hyper_head
    args.hyper_multi_merge_mode = cfg.hyper_multi_merge_mode
    args.joint_label = cfg.joint_label

    return args


# Combine distill parameters from *.yaml into arguments
def combine_distill_cfg_into_args(args, cfg_path):
    from distill_config.file_config import get_config
    import os

    cfg = get_config(cfg_path)
    assert os.path.exists(cfg.teacher_ckpt), 'In distillation stage, checkpoint of teacher model must be not None!!'

    args.tea_ckpt = cfg.teacher_ckpt
    args.middle_ids_t = cfg.middle_ids_t
    args.middle_ids_s = cfg.middle_ids_s
    args.common_kd_alpha = cfg.common_kd_alpha
    args.middle_kd_alpha = cfg.middle_kd_alpha
    args.t_dep = cfg.t_dep
    args.s_dep = args.dep = cfg.s_dep
    args.t_cs = cfg.t_cs
    args.s_cs = args.cs = cfg.s_cs

    # check middle ids if use middle distillation
    if len(args.middle_ids_t) > 0 or len(args.middle_ids_s) > 0:
        assert max(args.middle_ids_s) <= (args.s_dep - 1), 'Student layer indicies out of range!!'
        assert max(args.middle_ids_t) <= (args.t_dep - 1), 'Teacher layer indicies out of range!!'

    # These attributes are for distillation method2
    if hasattr(cfg, 't_f'):
        args.t_f = cfg.t_f
        args.number_of_frames = args.s_f = cfg.s_f
        # for distillation4.py
        args.stride = args.number_of_frames
        # for distillation5.py
        # args.stride = args.t_f
    else:
        args.t_f = args.s_f = args.number_of_frames
        # for distillation4.py
        args.stride = args.number_of_frames
        # for distillation5.py
        # args.stride = args.t_f

    # choose corresponding time downsample strategy automatically
    if args.number_of_frames == 243:
        args.tds = 2
    else:
        args.tds = 3

    # resume settings if exists
    if cfg.distiller_ckpt is not None or cfg.student_ckpt is not None:
        assert cfg.student_ckpt is not None and cfg.distiller_ckpt is not None, 'Checkpoint for student should be existsed'
        args.distiller_ckpt = cfg.distiller_ckpt
        args.stu_ckpt = cfg.student_ckpt
        args.resume = True

    # Add spatial mean for temproal token distillation if need
    if hasattr(cfg, 'spatial_mean'):
        args.spatial_mean = cfg.spatial_mean

    # Add adjustive knowledge distillation
    if hasattr(cfg, 'adjustive_middle'):
        args.adjustive_middle = cfg.adjustive_middle

    # Add generation for last layer distillation
    if hasattr(cfg, 'generation'):
        args.generation = cfg.generation
        # If exists generation mask, set generation mask temporal downsample mask
        if args.generation:
            args.generation_tds = cfg.generation_tds
    return args


# tmp for lr
def fix_lr_by_cs(args):
    if not args.coverlr:
        if args.cs == 512:
            args.lr = args.learning_rate = 0.000080
        elif args.cs == 256:
            args.lr = args.learning_rate = 0.000150
        elif args.cs == 128:
            args.lr = args.learning_rate = 0.000300
        else:
            raise NotImplementedError
    else:
        args.learning_rate = args.lr
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    ##########################################################
    # Important!!
    parser.add_argument('--version', default=1, type=int,
                        help='In the truly hyper, the right version should be settled to 2')
    # Quick debug for MPI-INF-3DHP
    parser.add_argument('--debug', action='store_true', help='For quick debug and go total process')
    # Abladation Study(For HyperGraph)
    parser.add_argument('--j2e', type=bool, default=True, help='Component for joint-to-hyperedge attention')
    parser.add_argument('--j2e_bias', type=bool, default=True, help='Component for joint-to-hyperedgeattn bias')
    parser.add_argument('--j2e_main', type=bool, default=True, help='Main compoent for joint-to-hyperedge attention')
    parser.add_argument('--j2j_adjacency', type=bool, default=True,
                        help='Learnable matrix for joint2joint attention of common spatial attention')
    # Experimental
    #################################################################################################################
    parser.add_argument('-l', '--log', default='log/default', type=str, metavar='PATH',
                        help='log file directory')
    parser.add_argument('-seed', default=0, type=int,
                        help='The random seed you need setting manually. It is recommend by 11451')
    parser.add_argument('--tds', default=1, type=int,
                        help='Time downsample for data trick proposed P-STMO, which author think it will expand receptive filed in time dimension.')

    # Trick for seed
    parser.add_argument('--respective_seed', action='store_true', help='Set seed for different libaray.')
    parser.add_argument('--torch_seed', default=7661715180000649815, type=int)
    parser.add_argument('--torch_cuda_seed', default=3281257025019913, type=int)
    parser.add_argument('--random_seed', default=7661715180000649815, type=int)
    parser.add_argument('--numpy_seed', default=2147483648, type=int)
    # Batch eval
    #################################################################################################################
    parser.add_argument('-batch_eval', action='store_true',
                        help='If start batch eval,then you should determine this parameter.')
    parser.add_argument('-conda_env', default=None,
                        help='To determine the detailed virtual enviroment of conda.')
    parser.add_argument('-eval_py', default='../run.py', help='To determine the evaluate py path file.')
    #################################################################################################################
    # HyperGraph Parameter
    parser.add_argument('--hyper_cfg', type=str, default=None,
                        help='In order to comebine the hyper config, its config is saved in a yaml file.')
    parser.add_argument('--hyper_head', default=1, type=int, help='hypermatrix count')
    parser.add_argument('--hyper_multi_merge_mode', default='none', type=str,
                        choices=['none', 'mean', 'max', 'weight_mean'],
                        help='multi merge way when it comes to multi hyper matrix')
    parser.add_argument('--joint_label', type=list,
                        default=None,
                        help='joint label in hyper edge to build hyper matrix correctly. if not none, right type is List[List[int]]')
    parser.add_argument('--spatial_mode', default='common', type=str,
                        choices=['common', 'hyper', 'hyper_frame_seperation', 'hyper_dependent_common'],
                        help='Due to in the thesis, the spatial input shape is [N*T,V,C], so we provide two version for HyperSpatialAttention about common hyper [N,T,V,C] and hyper frame seperation')
    parser.add_argument('--hyper_relation_bias', default='none', type=str, choices=['none', 'uniform'],
                        help="In the init version of Hyperformer, the relation bias is Identity, here, we can choose inject graph structure for attention in 'uniform' mode.")
    #################################################################################################################
    # For hrnet 16 joints experiments result
    parser.add_argument('--hrdet', action='store_true',
                        help='If the input 2d is hrnet, it should be dealt correctly.Actually, when you discover the input inferred by hrnet merely have 16 joints rather than 17 joints.')
    #################################################################################################################
    # Model parameter that current need change
    parser.add_argument('-cs', default=512, type=int,
                        help='channel size of model, only for trasformer. Attention: in the distillation stage, this parameter will be overwriten by channels of student model!!')
    parser.add_argument('-dep', default=8, type=int,
                        help='depth of model. Attention: in the distillation stage, this parameter will be overwriten by depth of student model!!')
    parser.add_argument('-pos_type', type=str, choices=['common', 'spd''mix_spd'], default='common',
                        help='postional embedding type for spatial transformer encoder')
    parser.add_argument('-f', '--number-of-frames', default='243', type=int, metavar='N',
                        help='how many frames used as input')
    #################################################################################################################
    # Strategy parameter
    parser.add_argument('--mpjpe', action='store_true', help='If true, the loss will be mpjpe, else will be wmpjpe.')
    #################################################################################################################
    # Distill arguments
    parser.add_argument('-distill', action='store_true', default=False,
                        help='If True, model will in distillation stage.')
    parser.add_argument('--distill_cfg', type=str, default=None, help='*.yaml files of distillation')
    parser.add_argument('--tea_ckpt', type=str, default=None, help='Complete path of teacher model')
    parser.add_argument('--distiller_ckpt', type=str, default=None,
                        help='The complete path of ckpt file for distiller to channel align')
    parser.add_argument('--stu_ckpt', type=str, default=None, help='The complete path of ckpt file for student model.')
    parser.add_argument('--middle_ids_t', type=list, default=[],
                        help='For feature distillation layer ids in the teacher model.')
    parser.add_argument('--middle_ids_s', type=list, default=[],
                        help='For feature distillation layer ids in the student model.')
    parser.add_argument('--common_kd_alpha', type=float, default=0.5,
                        help='For common knowledge ditillation alpha parameter in the distillation')
    parser.add_argument('--middle_kd_alpha', type=float, default=0.003,
                        help='If use middle layer distillation, it is corresponding weight parameter')
    parser.add_argument('--t_dep', type=int, default=512, help='Depth of teacher model')
    parser.add_argument('--s_dep', type=int, default=512, help='Depth of student model.It will overwrite dep argument')
    parser.add_argument('--t_cs', type=int, default=512, help='Channel of teacher model')
    parser.add_argument('--s_cs', type=int, default=512, help='Channel of student model.It will overwrite cs argument')
    parser.add_argument('--t_f', type=int, default=243, help='T frame receptive field of teacher model')
    parser.add_argument('--s_f', type=int, default=243,
                        help='T frame receptive field of student model.It will overwrite f argument')
    parser.add_argument('--spatial_mean', type=bool, default=False,
                        help='For temporal token distillation, we recommand to mean in spatial dimension first')
    parser.add_argument('--adjustive_middle', type=bool, default=False,
                        help='For middle distillation, if true, it will use adjustive distillation like PF-AFN')
    parser.add_argument('--generation', type=bool, default=False,
                        help='Like VitKD propoesed for last layer distillation.')
    parser.add_argument('--generation_tds', type=int, default=2,
                        help='To decide the correct mask temporal downsample in last layer. Actually, '
                             'the chosen downsampled stands for key frame.')
    #################################################################################################################
    # Action recognition task parrameter
    parser.add_argument('--action_label_map', type=str, default='',
                        help='If existed, it will load label map from corresponding txt, each line means a action class tag.')
    #################################################################################################################
    # Deployment to onnx
    parser.add_argument('--deployment', type=str, default='',
                        help='For total path of *.onnx file, which is more convenient for inference in cpu device.')
    #################################################################################################################
    # Attention visualization
    parser.add_argument('--attn-save', action='store_true', default=False,
                        help='if need attn map file, it will save in corresponding dir')
    parser.add_argument('--attn-dir', default='vis_data', type=str, help='the corresponding attn save dir basename')
    #################################################################################################################
    # General arguments
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME',
                        help='target dataset')
    ##########################################################
    # Init from MotionBERT
    parser.add_argument("--config", type=str, default="configs/pose3d/train_h36m.yaml", help="Path to the config file.")
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH',
                        help='pretrained checkpoint directory')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME',
                        help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-freq', '--print_freq', default=1000,
                        help='This is for pretraining on pose3d and finetuning on Action Recognition.')
    args = parser.parse_args()
    ##########################################################
    ##########################################################
    ##########################################################

    # Combine Hyper parameter
    if args.hyper_cfg is not None:
        args = combine_hyper_cfg_into_args(args, args.hyper_cfg)
    #######################################################################
    # Combine Distill parameter
    if args.distill_cfg is not None:
        args = combine_distill_cfg_into_args(args, args.distill_cfg)
    #######################################################################

    return args

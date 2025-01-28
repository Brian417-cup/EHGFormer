# Please run one of following commands:
# Please ensure your current path is under the root of our project!!!
# python deployment/deployment_action.py --config configs/action/train_NTU60_xsub_small.yaml --hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml --hyper_relation_bias uniform --version 2 --evaluate checkpoint/action/ntu_60/xsub/best_epoch_small.bin --deployment deployment/output/action/ntu60/action.onnx
# python deployment/deployment_action.py --config configs/action/train_NTU120_xsub_small.yaml --hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml --hyper_relation_bias uniform --version 2 --evaluate checkpoint/action/ntu_120/xsub/best_epoch_small.bin --deployment deployment/output/action/ntu120/action.onnx

import os
import os.path as osp
import sys

sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), '..'))

import torch
from torch import nn
import numpy as np
import torch.onnx as onnx
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone, load_pretrained_weights
from lib.utils.load_parameter import parse_args
from lib.model.model_action import ActionNet

max_person_num = 2


def reload_certain_task_attribute(opts, args):
    '''
    Reload som key parameter that has been defined in total config and update into arguments.
    :param opts: from cmd
    :param args: from config
    :return:
    '''
    assert args.dim_feat == args.dim_rep, 'Channel of regression head and encoders should be same!!'
    opts.dataset = args.dataset
    opts.cs = args.dim_rep
    opts.dep = args.depth
    opts.number_of_frames = args.maxlen


def deploy_with_config(args, opts):
    '''

    :param args: from config file
    :param opts: from cmd
    :return:
    '''
    print('#####################################')
    print('input command')
    print(f"python {' '.join(sys.argv)}")

    print('#####################################')
    print('config attribute')
    print(args)

    print('#####################################')
    print('cmd attribute')
    print(opts)
    print('#####################################')

    model_backbone = load_backbone(opts)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes,
                      dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim,
                      num_joints=args.num_joints)
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params / 1000000, 'Million')

    assert opts.evaluate != '', "In evaluation stage, corresponding checkpoint path shouln't be None !!"
    chk_filename = opts.evaluate
    print('Loading checkpoint', chk_filename)

    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    load_pretrained_weights(model, checkpoint['model'])

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()

    assert opts.deployment != '', "In deployment stage, corresponding output path for *.onnx file shouln't be None !!"

    # During deployment stage, it shoud be in a single gpu rather than multi-gpu
    input_rand_2d = torch.rand([1, max_person_num, args.maxlen, args.num_joints, 3], requires_grad=True)
    if torch.cuda.is_available():
        input_rand_2d = input_rand_2d.cuda()

    os.makedirs(osp.dirname(opts.deployment), exist_ok=True)

    dynamic_axes = {'pose2d': {0: 'batch_size'},
                    'action_label': {0: 'batch_size'}}

    onnx.export(model.module, (input_rand_2d), opts.deployment,
                # export_params=True, verbose=False,
                input_names=['pose2d'],
                output_names=['action_label'], opset_version=12, dynamic_axes=dynamic_axes)

    print('Finish deployment!!')


def test_onnx_ouput(args, deployment_path: str = 'output.onnx'):
    print('Testing deployemnt result...')
    import onnx
    model = onnx.load(deployment_path)
    print("Model inputs:")
    for input in model.graph.input:
        print("  Name:", input.name)
        print("  Shape:", input.type.tensor_type.shape.dim)
        print("  Type:", input.type.tensor_type.elem_type)
        print()

    print("Model outputs:")
    for output in model.graph.output:
        print("  Name:", output.name)
        print("  Shape:", output.type.tensor_type.shape.dim)
        print("  Type:", output.type.tensor_type.elem_type)
        print()

    if len(model.graph.input) == 1:
        input_depth = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value  # 假设深度值在第一个维度上
        print("Input depth:", input_depth)
    else:
        print("Model has multiple inputs. Please specify which one represents the depth.")

    import onnxruntime

    print("Supported onnxruntime version: ", onnxruntime.__version__)
    print("Supported Opset versions: ", onnxruntime.get_available_providers())

    ort_session = onnxruntime.InferenceSession(deployment_path, providers=["CPUExecutionProvider"], )
    out_label = ort_session.run(['action_label'], {
        'pose2d': np.random.rand(1, max_person_num, args.maxlen, args.num_joints, 3).astype(np.float32)
    })


if __name__ == "__main__":
    # opts from cmd, args from cfg
    opts = parse_args()
    args = get_config(opts.config)
    reload_certain_task_attribute(opts, args)

    # export onnx file
    deploy_with_config(args, opts)

    # test deployment for a random tensor
    test_onnx_ouput(args, opts.deployment)

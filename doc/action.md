# Skeleton-based Action Recognition

**Attention: All commands are excuted under the root of the porject.**

# Data

Download [`ntu60_hrnet.pkl`](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl)
and  [`ntu120_hrnet.pkl`](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_hrnet.pkl)
to  [`data/action/`](data/action/).

# Useful Tips

1. We use the data from 2D, but corresponding channel number is 3, the 3rd channel is the confidence of coordination. It
   has been metioned by [MotionBERT Issue](https://github.com/Walter0807/MotionBERT/issues/93).

# Accuracy Results:

## NTU 60

|                      |       manual1          |      manual2           |
| -------------------- | --------------------- | --------------------- |
| X-Sub              |       89.87            |                   |
| X-View             |                   |                   |
| X-Sub (finetuning) |          91.31             |                       |
| X-View (finetuning) |                       |                       |

## NTU 120

|                      |       manual1          |      manual2           |
| -------------------- | --------------------- | --------------------- |
| X-Sub              |                   |                   |
| X-View             |                   |                   |
| X-Sub (finetuning) |                       |                       |
| X-View (finetuning) |                       |                       |

# Train

**From scratch**

## NTU 60

```shell
# Corss-subject
python train_action.py \
--config configs/action/train_NTU60_xsub_small.yaml \
--checkpoint checkpoint/action/train_NTU60_xsub_small \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
<-r your_checkpoint_path_that_need_resume_for_resume_train>

# Cross-view
python train_action.py \
--config configs/action/train_NTU60_xview_small.yaml \
--checkpoint checkpoint/action/train_NTU60_xview_small\
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
<-r your_checkpoint_path_that_need_resume_for_resume_train>

```

## NTU 120

```shell
# Corss-subject
python train_action.py \
--config configs/action/train_NTU120_xsub_small.yaml \
--checkpoint checkpoint/action/train_NTU120_xsub_small \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
<-r your_checkpoint_path_that_need_resume_for_resume_train>

# Cross-view
python train_action.py \
--config configs/action/train_NTU120_xview_small.yaml \
--checkpoint checkpoint/action/train_NTU120_xview_small\
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
<-r your_checkpoint_path_that_need_resume_for_resume_train>

```

**Finetune from pertrained checkpoint**

## NTU 60

```shell
# Cross-subject
python train_action.py \
--config configs/action/ft_NTU60_xsub_small.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--checkpoint checkpoint/action/ft_NTU60_xsub

# Cross-view
python train_action.py \
--config configs/action/ft_NTU60_xview_small.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--checkpoint checkpoint/action/ft_NTU60_xview
```

Then, you should go into in [`config` file named **ft_xxx** under certain directory](../configs/action), and modify
their `backbone_ckpt`'s value for concrete checkpoint path.

# Evaluation

## NTU 60

```shell
# Cross-subject
python evaluate_action.py \
--config configs/action/train_NTU60_xsub_small.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--evaluate <Your complete checkpoint path for total network>

# Cross-view
python evaluate_action.py \
--config configs/action/train_NTU60_xview_small.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--evaluate <Your complete checkpoint path for total network>
```

# Heatmap Visualization

## NTU 60

```shell
python vis_action_attn.py \
--config configs/action/train_NTU60_xsub_small.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--evaluate <your complete checkpoint path,example here is : checkpoint/action/ntu_60/xsub/best_epoch_small.bin>
```

## NTU 120

```shell
python vis_action_attn.py \
--config configs/action/train_NTU120_xsub_small.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--evaluate <your complete checkpoint path,example here is : checkpoint/action/ntu_60/xsub/best_epoch_small.bin>
```

And the final heatmap result will be saved under [certain directory](vis_data)

# Deployment

Here we only take .onnx file for simple deployment on CPU device:

## NTU 60

```shell
# Cross-subject
python deployment/deployment_action.py \
--config configs/action/train_NTU60_xsub_small.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--evaluate <Your complete checkpoint path for total network> \
--deployment <Your complete output path for *.onnx file result>


# Cross-view
python deployment/deployment_action.py \
--config configs/action/train_NTU60_xview_small.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--evaluate <Your complete checkpoint path for total network> \
--deployment <Your complete output path for *.onnx file result>

```

Then, you can use exported *.onnx file under certain directory, and you can use it to visualize for wild video. Details
see in [inference document](inference.md).

The final *.onnx file information about input and output shapes are:

```shell
input: {
  pose2d: [N,T,V,C],
},
output{
  action_label: [N,num_action_class],
}

```
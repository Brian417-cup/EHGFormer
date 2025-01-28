# Pose3D

**Attention: All commands are excuted under the root of the porject.**

# Data

### Way1: generate by yourself

1. Download the finetuned Stacked Hourglass detections and our preprocessed H3.6M
   data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to your custom data_dir, which
   should be same with the `dt_root` in step 2.

> Note that the preprocessed data is only intended for reproducing our results more easily. If you want to use the dataset, please register to the [Human3.6m website](http://vision.imar.ro/human3.6m/) and download the dataset in its original format. Please refer to [LCN](https://github.com/CHUNYUWANG/lcn-pose#data) for how we prepare the H3.6M data.

2.Slice the motion clips (len=243, stride=243,tds=2)

   ```bash
   python tools/convert_h36m_mixste_style.py
   ```

> You should change the parameter about 'dt_root' in the instance code of 'DataReaderH36M' for your concerete directory path and default save dir is under [target directory](../data/motion3d).

### Way2: only download something into target directory(recommend)

> Currently, we have offered converted data under [target directory](../data/motion3d/H36M-SH), you should only move h36m_sh_conf_cam_source_final.pkl.pkl from [link](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) under target directory too.

# Details about data (Especially for each shape of component)

**source data h36m_sh_conf_cam_source_final.pkl**

```shell
{
  'train': 
  {
    'joint_2d': [T,V=17,C=2] ,
    'confidence': [T,V=17,C=1],
    'joint3d_image': [T,V=17,C=3],
    'camera_name': [T], // This may be about camera number
    'source': [T] // type is str of List, each is s_<subject_number>_act_<action_number>_cam_<camera_number>
  },
  
  'test': 
  {
      'joint_2d': [T,V=17,C=2],
      'confidence': [T,V=17,C=1],
      'joint3d_image': [T,V=17,C=3],
      'joint_2.5d_image': [T,V=17,C=3],
      '2.5d_factor': [T],
      'camera_name': [T],
      'action': [T], // List, each is the concrete action name,
      'source': [T] // type is str of List, each is s_<subject_number>_act_<action_number>_cam_<camera_number>
  }
}
```

**dealt split xx.pkl file**

```shell
{
  'data_input': [T,V,C]
  'data_label': [T,V,C]
}
```

# MPJPE Results:

It have some differences in paper.

|                      | partition1 | partition2 |
| -------------------- | --------------------- | --------------------- |
| static               | 38.8                  | 38.8                  |
| dynamic              | 38.3                  | 38.2                  |
| dynamic+distillation |   39.9                    | 39.8 |

# Useful Tips

1. We use the data from 2D, but corresponding channel number is 3, the 3rd channel is the confidence of coordination. It
   has been metioned by [MotionBERT Issue](https://github.com/Walter0807/MotionBERT/issues/93).

# Train

**Attention: Here we offer global(dynamic) version and non-global(static) version, the difference between them is the
root joint. Details can been seen in [link](https://github.com/Walter0807/MotionBERT/issues/107).**

**From scratch**

```shell
python train.py \
--config configs/pose3d/train_h36m<_small><_global>.yaml \
--checkpoint checkpoint/pose3d/h36m<_small><_global> \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
<-r your_checkpoint_path_that_need_resume_for_resume_train>
```

**Distillation**

```shell
python train_distill.py \
--config configs/pose3d/train_h36m_small<_global>_distill.yaml \
--checkpoint checkpoint/pose3d/h36m_small<_global>_distill \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--distill_cfg distill_config/distillation_config.yaml \
--version 2
```

**Finetune**

# Evaluation (Not used currently)

```shell
python evaluate.py \
--config configs/pose3d/train_h36m<_small><_global>.yaml \
--checkpoint checkpoint/pose3d/h36m<_small> \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--evaluate <Your complete checkpoint path for total network>
```

# Heatmap Visualization

```shell
# non-distillation
python vis_attn.py \
--config configs/pose3d/train_h36m<_global>.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--evaluate <your complete checkpoint path,example here is : checkpoint/pose3d/h36m/best_epoch.bin>
# distillation
python vis_attn.py \
--config configs/pose3d/train_h36m_small<_global>_distill.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--evaluate <your complete checkpoint path,example here is : checkpoint/pose3d/h36m/best_epoch.bin>
```

And the final heatmap result will be saved under [certain directory](vis_data).

# Deployment

Here we only take .onnx file for simple deployment on CPU device:

```shell
# non-distillation
python deployment/deployment_pose3d.py \
--config configs/pose3d/train_h36m<_global>.yaml \
--checkpoint checkpoint/pose3d/h36m<_global> \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--evaluate <Your complete checkpoint path for total network> \
--deployment <Your complete output path for *.onnx file result>

# distillation
python deployment/deployment_pose3d.py \
--config configs/pose3d/train_h36m_small<_global>_distill.yaml \
--checkpoint checkpoint/pose3d/h36m_small<_global>_distill \
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
  pose2d: [N,T,V,C=2+1],
},

output{
  pose3d: [N,T,V,C=3],
}

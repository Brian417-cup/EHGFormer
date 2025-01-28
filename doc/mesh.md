# Human Mesh Recovery

**Attention: We only implement our module as `refine` for image-based mesh
recovery [HybrIK](https://github.com/Jeff-sjtu/HybrIK)!!**

## Data

1. Download the datasets [here](https://1drv.ms/f/s!AvAdh0LSjEOlfy-hqlHxdVMZxWM) and put them to  `data/mesh/`. We use
   Human3.6M, COCO, and PW3D for training and testing. Descriptions of the joint regressors could be found
   in [SPIN](https://github.com/nkolot/SPIN/tree/master/data).
2. Download the SMPL model(`basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`) from [SMPLify](https://smplify.is.tue.mpg.de/),
   put it to `data/mesh/`, and rename it as `SMPL_NEUTRAL.pkl`

## Train

**Train from scratch:**

```bash
# with 3DPW
python train_mesh_refine.py \
--config configs/mesh/train_pw3d.yaml \
--checkpoint checkpoint/mesh/train_pw3d_refine \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
<-r your_checkpoint_path_that_need_resume_for_resume_train>

# H36M
python train_mesh_refine.py \
--config configs/mesh/train_h36m.yaml \
--checkpoint checkpoint/mesh/train_h36m_refine \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
<-r your_checkpoint_path_that_need_resume_for_resume_train>
```

**Finetune from a pretrained model:**

```bash
# with 3DPW
python train_mesh_refine.py \
--config configs/mesh/ft_pw3d.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/mesh/ft_pw3d_refine \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
<-r your_checkpoint_path_that_need_resume_for_resume_train>

# H36M
python train_mesh_refine.py \
--config configs/mesh/ft_h36m.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/mesh/ft_h36m_refine \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
<-r your_checkpoint_path_that_need_resume_for_resume_train>
```

Then, you should go into in [`config` file named **ft_xxx** under certain directory](../configs/mesh), and modify
their `backbone_ckpt`'s value for concrete checkpoint path.

## Evaluation

```bash
# with 3DPW
python train_mesh_refine.py \
--config configs/mesh/train_pw3d.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--checkpoint checkpoint/mesh/evaluate_3dpw_refine \
--evaluate <Your complete checkpoint path for total network>

# H36M
python train_mesh_refine.py \
--config configs/mesh/train_h36m.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--checkpoint checkpoint/mesh/evaluate_h36m_refine \
--evaluate <Your complete checkpoint path for total network>
```

## Deployment

### Deployment for HybrIK

We
use [target model in naive HybrIK](https://github.com/Jeff-sjtu/HybrIK/blob/9b8681dcf3c902dd5dacc01520ba04982990e1e2/hybrik/models/HRNetWithCam.py#L53)
, the input and output shapes for deployment are:

```shell
input:{
  input_image: [N,H,W,C],
  bbox: [N,4], # detection result in xyxy format
  image_center: [N,2]
},
output:{
      pred_shape  [N, 10],
      pred_theta_mats: [N, 216],
      pred_uvd_jts: [N, 87],
      pred_xyz_jts_29: [N, 87],
      pred_xyz_jts_24: [N, 72],
      pred_xyz_jts_24_struct: [N, 72],
      pred_xyz_jts_17: [N, 51],
      pred_vertices: [N, 6890, 3],
      cam_scale: [N, 1],
      cam_root: [N, 3],
      transl: [N, 3],
      pred_camera: [N, 1]
}
```

**Tips: Here is the [key answer](https://github.com/Jeff-sjtu/HybrIK/issues/101) about difference between `transl`
and `cam_root`.**

### Deployment for Refine module

Here we only take .onnx file for simple deployment on CPU device:

```shell
# H36M
python deployment/deployment_mesh.py \
--config configs/mesh/train_h36m.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--evaluate <Your complete checkpoint path for total network> \
--deployment <Your complete output path for *.onnx file result>


# PW3D
python deployment/deployment_mesh.py \
--config configs/mesh/train_pw3d.yaml \
--hyper_cfg hyper_config/method2/manual_joint_label<1 or 2>.yaml \
--hyper_relation_bias uniform \
--version 2 \
--evaluate <Your complete checkpoint path for total network> \
--deployment <Your complete output path for *.onnx file result>
```

The final *.onnx file information about input and output shapes are:

```shell
input: {
  pose2d: [N,T,V,C],
  hybrik_theta: [N,T,24*3],
  hybrik_shape: [N,T,10]
},
output{
  rot_mat: [N,T,V,3x3 rotation matrix],
  shape: [N,T,10],
  vertices: [N,T,6890,3],
  keypoint_3d: [N,T,V,C=3]
}

```

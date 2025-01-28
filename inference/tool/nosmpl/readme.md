# NoSMPL

<p align="center">
  <img src="https://jihulab.com/godly/fger/-/raw/c61e2e1d45afb82f48ced2bbcf6d6baa63f8f9a0/images/2023/04/5_22_41_32_37_1680705278.gif" />
</p>

> A tool can visualize SMPL in an easier way. All you need is just `pip install nosmpl`.

An enchanced and accelerated SMPL operation which commonly used in 3D human mesh generation. It takes a poses, shapes, cam_trans as inputs, outputs a high-dimensional 3D mesh verts.

However, SMPL codes and models are so messy out there, they have a lot of codes do calculation, some of them can not be easily deployed or accerlarated. So we have `nosmpl` here, it provides:

- build on smplx, but with onnx support;
- can be inference via onnx;
- we also demantrated some using scenarios infer with `nosmpl` but without any model, only onnx.

This packages provides:

- [ ] Highly optimized pytorch acceleration with FP16 infer enabled;
- [x] Supported ONNX export and infer via ort, so that it might able used into TensorRT or OpenVINO on cpu;
- [x] Support STAR, next generation of SMPL.
- [x] Provide commonly used geoemtry built-in support without torchgeometry or kornia.

STAR model download from: https://star.is.tue.mpg.de


## SMPL ONNX Model Downloads

I have exported 2 models, include `SMPL-H` and `SMPL`, which can cover most using scenarios:

- `smpl`: [link](https://github.com/jinfagang/nosmpl/releases/download/v1.1/smpl.onnx)
- `smpl-h`: [link](https://github.com/jinfagang/nosmpl/releases/download/v1.0/smplh_sim_w_orien.onnx)

They can also be found at github release.

For usage, you can take examples like `examples/demo_smplh_onnx.py`.


## Quick Start

Now you can using `nosmpl` to visualize smpl with just few line of codes without **download any SMPL file**:

```python
from nosmpl.smpl_onnx import SMPLOnnxRuntime
import numpy as np


smpl = SMPLOnnxRuntime()

body = np.random.randn(1, 23, 3).astype(np.float32)
global_orient = np.random.randn(1, 1, 3).astype(np.float32)
outputs = smpl.forward(body, global_orient)
print(outputs)
# you can visualize the verts with Open3D now.
```

So your predicted 3d pose such as SPIN, HMR, PARE etc, grap your model ouput, and through this `nosmpl` func, you will get your SMPL vertices!


## Updates

- **`2023.02.28`**: An SMPL-H ONNX model released! Now You can using ONNXRuntime to get a 3D SMPL Mesh from a pose!
- **`2022.05.16`**: Now added `human_prior` inside `nosmpl`, you don't need install that lib anymore, or install torchgeometry either:
  ```python
  from nosmpl.vpose.tools.model_loader import load_vposer
  self.vposer, _ = load_vposer(VPOSER_PATH, vp_model="snapshot")
  ```
  then you can load vpose to use.
- **`2022.05.10`**: Add BHV reader, you can now read and write bvh file:

  ```python
  from nosmpl.parsers import bvh_io
  import sys


  animation = bvh_io.load(sys.argv[1])
  print(animation.names)
  print(animation.frametime)
  print(animation.parent)
  print(animation.offsets)
  print(animation.shape)
  ```

- **`2022.05.07`**: Added a visualization for Human36m GT, you can using like this to visualize h36m data now:

  ```
  import nosmpl.datasets.h36m_data_utils as data_utils
  from nosmpl.datasets.h36m_vis import h36m_vis_on_gt_file
  import sys

  if __name__ == "__main__":
      h36m_vis_on_gt_file(sys.argv[1])
  ```

  Just send a h36m txt annotation file, and you can see the animation result. Also, you can using `from nosmpl.datasets.h36m_vis import h36m_load_gt_3d_data` to load 3d data in 3D space.

- **`2022.03.03`**: I add some `box_transform` code into `nosmpl`, no we can get box_scale info when recover cropped img predicted 3d vertices back to original image. This is helpful when you project 3d vertices back to original image when using `realrender`.
  the usage like:
  ```
  from nosmpl.box_trans import get_box_scale_info, convert_vertices_to_ori_img
  box_scale_o2n, box_topleft, _ = get_box_scale_info(img, bboxes)
  frame_verts = convert_vertices_to_ori_img(
            frame_verts, s, t, box_scale_o2n, box_topleft
        )
  ```
- **`2022.03.05`**: More to go.

## Features

The most exciting feature in `nosmpl` is **you don't need download any SMPL files anymore**, you just need to download my exported `SMPL.onnx` or `SMPLH.onnx`, then you can using numpy to generate a Mesh!!!

`nosmpl` also provided a script to visualize it~!

```python

import onnxruntime as rt
import torch
import numpy as np
from nosmpl.vis.vis_o3d import vis_mesh_o3d


def gen():
    sess = rt.InferenceSession("smplh_sim.onnx")

    for i in range(5):
        body_pose = (
            torch.randn([1, 63], dtype=torch.float32).clamp(0, 0.4).cpu().numpy()
        )
        left_hand_pose = (
            torch.randn([1, 6], dtype=torch.float32).clamp(0, 0.4).cpu().numpy()
        )
        right_hand_pose = (
            torch.randn([1, 6], dtype=torch.float32).clamp(0, 0.4).cpu().numpy()
        )

        outputs = sess.run(
            None, {"body": body_pose, "lhand": left_hand_pose, "rhand": right_hand_pose}
        )

        vertices, joints, faces = outputs
        vertices = vertices[0].squeeze()
        joints = joints[0].squeeze()

        faces = faces.astype(np.int32)
        vis_mesh_o3d(vertices, faces)


if __name__ == "__main__":
    gen()
```

You will see a mesh with your pose, generated:

![](https://s1.ax1x.com/2023/03/01/ppim6EV.png)

As you can see, we are using a single ONNX model, by some randome poses, you can generated a visualized mesh.

**this is useful when you wanna test your predict pose is right or not!**

If you using this in your project, your code will be decrease 190%, if it helps, consider cite `nosmpl` in your project!

More details you can join our Metaverse Wechat group for discussion! QQ join link:

## Examples

an example to call `nosmlp`:

```python
from nosmpl.smpl import SMPL

smpl = SMPL(smplModelPath, extra_regressor='extra_data/body_module/data_from_spin/J_regressor_extra.npy').to(device)

# get your betas and rotmat
pred_vertices, pred_joints_3d, faces = smpl(
                    pred_betas, pred_rotmat
                )

# note that we returned faces in SMPL model, you can use for visualization
# joints3d will add extra joints if you have extra_regressor like in SPIN or VIBE

```

The output shape of onnx model like:

```
                    basicModel_neutral_lbs_10_207_0_v1.0.0.onnx Detail
╭───────────────┬────────────────────────────┬──────────────────────────┬────────────────╮
│ Name          │ Shape                      │ Input/Output             │ Dtype          │
├───────────────┼────────────────────────────┼──────────────────────────┼────────────────┤
│ 0             │ [1, 10]                    │ input                    │ float32        │
│ 1             │ [1, 24, 3, 3]              │ input                    │ float32        │
│ verts         │ [-1, -1, -1]               │ output                   │ float32        │
│ joints        │ [-1, -1, -1]               │ output                   │ float32        │
│ faces         │ [13776, 3]                 │ output                   │ int32          │
╰───────────────┴────────────────────────────┴──────────────────────────┴────────────────╯
                             Table generated by onnxexplorer
```

## Notes

1. About quaternion

the `aa2quat` function, will converts quaternion in `wxyz` as default order. This is **different** from scipy. It's consistent as mostly 3d software such as Blender or UE.

## Results

Some pipelines build with `nosmpl` support.

![](https://s4.ax1x.com/2022/02/20/HLGD00.gif)

## Copyrights

Copyrights belongs to Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) and Lucas Jin

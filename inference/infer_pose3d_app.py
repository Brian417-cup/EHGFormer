import os.path

from infer_pose3d import main as infer_pose3d_main, combine_cfg_and_args, get_vid_info
import os.path as osp
import gradio as gr

root_cfg_dir = ''

cfg = combine_cfg_and_args()


def video_identity(video_path):
    # current epoch inference
    cfg.video_path = video_path
    _, vid_fps, vid_size = get_vid_info(video_path)
    infer_pose3d_main(cfg, vid_size, vid_fps)
    output_path = osp.join('output_3dhpe', osp.basename(video_path).split('.')[0] + '_3dhpe.mp4')

    return output_path


demo = gr.Interface(fn=video_identity,
                    inputs=[gr.Video()],
                    outputs=['video'],
                    )

if __name__ == "__main__":
    demo.launch()

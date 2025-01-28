"""
utils on visualize SMPL mesh in Open3D

"""
import time
import numpy as np
import os

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    Vector3dVector = o3d.utility.Vector3dVector
    Vector3iVector = o3d.utility.Vector3iVector
    Vector2iVector = o3d.utility.Vector2iVector
    TriangleMesh = o3d.geometry.TriangleMesh
except Exception as e:
    print(e)
    print("run pip install open3d for vis.")
    o3d = None


def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    mesh.compute_vertex_normals()
    if colors is not None:
        colors = np.array(colors)
        # mesh.vertex_colors = Vector3dVector(colors)
        mesh.paint_uniform_color(colors)
    else:
        r_c = np.random.random(3)
        mesh.paint_uniform_color(r_c)
    return mesh


def vis_mesh_o3d(vertices, faces):
    mesh = create_mesh(vertices, faces)
    min_y = -mesh.get_min_bound()[1]
    mesh.translate([0, min_y, 0])
    o3d.visualization.draw_geometries([mesh])


def vis_mesh_o3d_loop(vertices, faces):
    mesh = create_mesh(vertices, faces)
    min_y = -mesh.get_min_bound()[1]
    mesh.translate([0, min_y, 0])
    o3d.visualization.draw_geometries([mesh])


class Open3DVisualizer:
    def __init__(self, save_img_folder=None, fps=35, frame_width=256, frame_height=256, enable_axis=False,
                 visible_window=False) -> None:
        self.cam_parameter = o3d.camera.PinholeCameraParameters()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("NoSMPL Open3D Visualizer", visible=visible_window,
                               width=frame_width, height=frame_height)
        self.is_visible = visible_window

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        if enable_axis:
            self.vis.add_geometry(coordinate_frame)

        self.geometry_crt = None
        self.fps = fps
        self.idx = 0

        self.save_img_folder = save_img_folder
        if save_img_folder:
            os.makedirs(self.save_img_folder, exist_ok=True)

    def update(self, vertices, faces, trans=None, R_along_axis=(0, 0, 0), waitKey=1,
               extrinsics_matrix=None, intrinsics_matrix=None):
        # set new camera view field and bind into visualizer if extrinsics matrix and intrinsics matrix are given
        if extrinsics_matrix is not None and intrinsics_matrix is not None:
            self.cam_parameter.extrinsic = extrinsics_matrix
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.intrinsic_matrix = intrinsics_matrix
            self.cam_parameter.intrinsic = intrinsic

            ctr: o3d.visualization.ViewControl = self.vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(self.cam_parameter)

        mesh = create_mesh(
            vertices, faces, colors=[82.0 / 255, 217.0 / 255, 118.0 / 255]
        )
        # if not self.geometry_crt:
        #     self.geometry_crt = mesh

        # min_y = -mesh.get_min_bound()[1]
        # mesh.translate([0, min_y, 0])

        R = mesh.get_rotation_matrix_from_xyz(R_along_axis)
        mesh.rotate(R, center=(0, 0, 0))

        if trans:
            mesh.translate(trans)

        if waitKey == 0:
            self.vis.clear_geometries()
            self.vis.add_geometry(mesh)
            self.vis.run()
        else:
            self.vis.clear_geometries()
            self.vis.add_geometry(mesh)
            # self.vis.update_geometry(mesh)
            self.vis.poll_events()
            self.vis.update_renderer()
            if self.save_img_folder:
                self.vis.capture_screen_image(
                    os.path.join(self.save_img_folder, "temp_%04d.png" % self.idx)
                )
            self.idx += 1
            if self.is_visible:
                time.sleep(1 / self.fps)

    def release(self):
        self.vis.destroy_window()


if __name__ == '__main__':
    from pickle import dump, load

    ToGLCamera = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    FromGLGamera = np.linalg.inv(ToGLCamera)


    def model_matrix_to_extrinsic_matrix(model_matrix):
        return np.linalg.inv(model_matrix @ FromGLGamera)


    def create_camera_intrinsic_from_size(width=1024, height=768, hfov=60.0, vfov=60.0):
        fx = (width / 2.0) / np.tan(np.radians(hfov) / 2)
        fy = (height / 2.0) / np.tan(np.radians(vfov) / 2)
        # fx = fy # not sure why, but it looks like fx should be governed/limited by fy
        return np.array(
            [[fx, 0, width / 2.0],
             [0, fy, height / 2.0],
             [0, 0, 1]])


    def save_view(vis, fname='saved_view.pkl'):
        try:
            model_matrix = np.asarray(vis.scene.camera.get_model_matrix())
            extrinsic = model_matrix_to_extrinsic_matrix(model_matrix)
            width, height = vis.size.width, vis.size.height
            intrinsic = create_camera_intrinsic_from_size(width, height)
            saved_view = dict(extrinsic=extrinsic, intrinsic=intrinsic, width=width, height=height)
            with open(fname, 'wb') as pickle_file:
                dump(saved_view, pickle_file)
        except Exception as e:
            print(e)


    def load_view(vis, fname="saved_view.pkl"):
        try:
            with open(fname, 'rb') as pickle_file:
                saved_view = load(pickle_file)
            vis.setup_camera(saved_view['intrinsic'], saved_view['extrinsic'], saved_view['width'],
                             saved_view['height'])
            # Looks like the ground plane gets messed up, no idea how to fix
        except Exception as e:
            print("Can't find file", e)


    gui.Application.instance.initialize()
    vis = o3d.visualization.O3DVisualizer("Demo to Load a Camera Viewpoint for O3DVisualizer", 1024, 768)
    gui.Application.instance.add_window(vis)
    vis.point_size = 8
    vis.show_axes = True
    # Add saving and loading view
    vis.add_action("Save Camera View", save_view)
    vis.add_action("Load Camera View", load_view)

    # Create Random Geometry
    pcd = o3d.io.read_point_cloud('D:\W_M_NIUST\WM_Project\MyProject/scene0423_00_crop_0.ply')
    # pc = np.random.randn(100, 3) * 0.5
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    vis.add_geometry("Random Point Cloud", pcd)
    vis.reset_camera_to_default()

    gui.Application.instance.run()

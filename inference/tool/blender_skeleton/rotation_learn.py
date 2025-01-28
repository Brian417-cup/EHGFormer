import numpy as np

v1 = np.array([1, 1, 1])
x = np.array([1, 0, 0])
y = np.array([0, 1, 0])
z = np.array([0, 0, 1])

v1_norm = np.linalg.norm(v1)
a, b, c = v1[0] / v1_norm, v1[1] / v1_norm, v1[2] / v1_norm
print(a, b, c)
a, b, c = np.rad2deg(np.arccos(a)), np.rad2deg(np.arccos(b)), np.rad2deg(np.arccos(c))
print(a, b, c)


def rotation_angles(vector):
    # 将向量单位化
    vector_normalized = vector / np.linalg.norm(vector)

    # 计算向量与 x、y、z 轴的夹角
    angle_x = np.arccos(np.dot(vector_normalized, [1, 0, 0]))
    angle_y = np.arccos(np.dot(vector_normalized, [0, 1, 0]))
    angle_z = np.arccos(np.dot(vector_normalized, [0, 0, 1]))

    # 将弧度转换为角度
    angle_x_deg = np.degrees(angle_x)
    angle_y_deg = np.degrees(angle_y)
    angle_z_deg = np.degrees(angle_z)

    return angle_x_deg, angle_y_deg, angle_z_deg


# 假设有一个向量
my_vector = np.array([-1, -1, 1])  # 这里举例使用 [1, 1, 1] 作为向量

# 计算沿 x、y、z 轴的旋转角度
rotation_x, rotation_y, rotation_z = rotation_angles(my_vector)

print("沿 x 轴旋转角度：", rotation_x)
print("沿 y 轴旋转角度：", rotation_y)
print("沿 z 轴旋转角度：", rotation_z)
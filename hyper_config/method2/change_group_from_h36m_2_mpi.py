import numpy as np

# group_list = [0, 2, 3, 3, 2, 3, 3, 4, 1, 2, 2, 2, 2, 0, 2, 2, 0]
# group_list = [0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
group_list = [0, 1, 1, 1, 2, 2, 2, 0, 0, 3, 3, 4, 4, 4, 5, 5, 5]
h36m_2_mpi = {
    0: 14,
    1: 11,
    2: 12,
    3: 13,
    4: 8,
    5: 9,
    6: 10,
    7: 15,
    8: 1,
    9: 16,
    10: 0,
    11: 2,
    12: 3,
    13: 4,
    14: 5,
    15: 6,
    16: 7
}
mpi_group_list = np.zeros(17, dtype=np.int).tolist()
result = "["
for i, data in enumerate(group_list):
    mpi_group_list[h36m_2_mpi[i]] = data

print(mpi_group_list)

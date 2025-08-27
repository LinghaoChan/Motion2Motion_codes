# BSD License

# For fairmotion software

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

#  * Neither the name Facebook nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np


def distance_between_points(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def distance_from_plane(a, b, c, p, threshold):
    ba = np.array(b) - np.array(a)
    ca = np.array(c) - np.array(a)
    cross = np.cross(ca, ba)

    pa = np.array(p) - np.array(a)
    return np.dot(cross, pa) / np.linalg.norm(cross) > threshold


def distance_from_plane_normal(n1, n2, a, p, threshold):
    normal = np.array(n2) - np.array(n1)
    pa = np.array(p) - np.array(a)
    return np.dot(normal, pa) / np.linalg.norm(normal) > threshold


def angle_within_range(j1, j2, k1, k2, range):
    j = np.array(j2) - np.array(j1)
    k = np.array(k2) - np.array(k1)

    angle = np.arccos(np.dot(j, k) / (np.linalg.norm(j) * np.linalg.norm(k)))
    angle = np.degrees(angle)

    if angle > range[0] and angle < range[1]:
        return True
    else:
        return False


def velocity_direction_above_threshold(
    j1, j1_prev, j2, j2_prev, p, p_prev, threshold, time_per_frame=1 / 120.0
):
    velocity = (
        np.array(p) - np.array(j1) - (np.array(p_prev) - np.array(j1_prev))
    )
    direction = np.array(j2) - np.array(j1)

    velocity_along_direction = np.dot(velocity, direction) / np.linalg.norm(
        direction
    )
    velocity_along_direction = velocity_along_direction / time_per_frame
    return velocity_along_direction > threshold


def velocity_direction_above_threshold_normal(
    j1, j1_prev, j2, j3, p, p_prev, threshold, time_per_frame=1 / 120.0
):
    velocity = (
        np.array(p) - np.array(j1) - (np.array(p_prev) - np.array(j1_prev))
    )
    j31 = np.array(j3) - np.array(j1)
    j21 = np.array(j2) - np.array(j1)
    direction = np.cross(j31, j21)

    velocity_along_direction = np.dot(velocity, direction) / np.linalg.norm(
        direction
    )
    velocity_along_direction = velocity_along_direction / time_per_frame
    return velocity_along_direction > threshold


def velocity_above_threshold(p, p_prev, threshold, time_per_frame=1 / 120.0):
    velocity = np.linalg.norm(np.array(p) - np.array(p_prev)) / time_per_frame
    return velocity > threshold


def calc_average_velocity(positions, i, joint_idx, sliding_window, frame_time):
    current_window = 0
    average_velocity = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (
            positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        )
        current_window += 1
    return np.linalg.norm(average_velocity / (current_window * frame_time))


def calc_average_acceleration(
    positions, i, joint_idx, sliding_window, frame_time
):
    seq_len = len(positions)
    total_acc = np.zeros(3, dtype=np.float64)
    count = 0

    for offset in range(-sliding_window, sliding_window + 1):
        prev_idx = i + offset - 1
        curr_idx = i + offset
        next_idx = i + offset + 1

        # Make sure indices are in bounds
        if prev_idx < 0 or next_idx >= seq_len:
            continue

        # Compute displacement vectors and force them to ndarray
        disp1 = np.asarray(
            positions[curr_idx][joint_idx] - positions[prev_idx][joint_idx],
            dtype=np.float64
        )
        disp2 = np.asarray(
            positions[next_idx][joint_idx] - positions[curr_idx][joint_idx],
            dtype=np.float64
        )

        # Velocity estimates
        v1 = disp1 / frame_time
        v2 = disp2 / frame_time

        # Acceleration estimate
        acc = (v2 - v1) / frame_time

        total_acc += acc
        count += 1

    if count == 0:
        return 0.0

    # Average over the window and return magnitude
    avg_acc = total_acc / count
    return np.linalg.norm(avg_acc)


def calc_average_velocity_horizontal(
    positions, i, joint_idx, sliding_window, frame_time, up_vec="z"
):
    seq_len = len(positions)
    # Sum of displacements in 3D
    average_disp = np.zeros(3, dtype=np.float64)
    count = 0

    for offset in range(-sliding_window, sliding_window + 1):
        prev_idx = i + offset - 1
        curr_idx = i + offset
        if prev_idx < 0 or curr_idx >= seq_len:
            continue

        # 1) 计算两帧关节位置差
        disp = positions[curr_idx][joint_idx] - positions[prev_idx][joint_idx]
        # 2) 强制转为 NumPy 数组，防止 disp 是 list / torch.Tensor / pandas.TimeStamp 等
        disp = np.asarray(disp, dtype=np.float64)

        average_disp += disp
        count += 1

    if count == 0:
        return 0.0

    # 根据 up_vec 提取水平面分量
    if up_vec == "y":
        horiz_disp = np.array([average_disp[0], average_disp[2]], dtype=np.float64)
    elif up_vec == "z":
        horiz_disp = np.array([average_disp[0], average_disp[1]], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported up_vec '{up_vec}'; use 'y' or 'z'.")

    # 平均位移 / 时间 → 平均速度
    avg_velocity = horiz_disp / (count * frame_time)

    # 返回标量大小
    return np.linalg.norm(avg_velocity)

def calc_average_velocity_vertical(
    positions, i, joint_idx, sliding_window, frame_time, up_vec
):
    seq_len = len(positions)
    # Accumulate 3D displacement
    total_disp = np.zeros(3, dtype=np.float64)
    count = 0

    for offset in range(-sliding_window, sliding_window + 1):
        prev_idx = i + offset - 1
        curr_idx = i + offset
        if prev_idx < 0 or curr_idx >= seq_len:
            continue

        # Compute displacement and ensure it's a NumPy array
        disp = np.asarray(
            positions[curr_idx][joint_idx] - positions[prev_idx][joint_idx],
            dtype=np.float64
        )
        total_disp += disp
        count += 1

    if count == 0:
        return 0.0

    # Extract the vertical component based on up_vec
    if up_vec == "y":
        vert_disp = np.array([total_disp[1]], dtype=np.float64)
    elif up_vec == "z":
        vert_disp = np.array([total_disp[2]], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported up_vec '{up_vec}'; use 'y' or 'z'.")

    # Convert displacement to velocity
    avg_velocity = vert_disp / (count * frame_time)

    # Return scalar magnitude
    return np.linalg.norm(avg_velocity)
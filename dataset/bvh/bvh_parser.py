import torch
import numpy as np
import dataset.bvh.bvh_io as bvh_io
from utils.kinematics import ForwardKinematicsJoint
from utils.transforms import quat2repr6d
from utils.contact import foot_contact
from dataset.bvh.Quaternions import Quaternions
from dataset.bvh.bvh_writer import WriterWrapper


class Skeleton:
    def __init__(
        self, names, parent, offsets, joint_reduction=True, skeleton_conf=None
    ):
        self._names = names
        self.original_parent = parent
        self._offsets = offsets
        self._parent = None
        self._ee_id = None
        self.contact_names = []

        for i, name in enumerate(self._names):
            if ":" in name:
                self._names[i] = name[name.find(":") + 1 :]

        if joint_reduction or skeleton_conf is not None:
            assert (
                skeleton_conf is not None
            ), "skeleton_conf can not be None if you use joint reduction"
            corps_names = skeleton_conf["corps_names"]
            self.contact_names = skeleton_conf["contact_names"]
            self.contact_threshold = skeleton_conf["contact_threshold"]

            self.contact_id = []
            for i in self.contact_names:
                self.contact_id.append(corps_names.index(i))
        else:
            self.skeleton_type = -1
            corps_names = self._names

        self.details = (
            []
        )  # joints that does not belong to the corps (we are not interested in them)
        for i, name in enumerate(self._names):
            if name not in corps_names:
                self.details.append(i)

        self.corps = []
        self.simplified_name = []
        self.simplify_map = {}
        self.inverse_simplify_map = {}

        # Repermute the skeleton id according to the databse
        for name in corps_names:
            for j in range(len(self._names)):
                # if one joint is spine and the other is spine0, it will cause error. therefore, we add some random string to the end of the name in bvhs.
                if name in self._names[j]:
                    self.corps.append(j)
                    break
        if len(self.corps) != len(corps_names):
            print(self._names)
            for i in self.corps:
                print(self._names[i], end=" ")
            print(self.corps, self.skeleton_type, len(self.corps), sep="\n")
            raise Exception("Problem in this skeleton")

        self.joint_num_simplify = len(self.corps)
        for i, j in enumerate(self.corps):
            self.simplify_map[j] = i
            self.inverse_simplify_map[i] = j
            self.simplified_name.append(self._names[j])
        self.inverse_simplify_map[0] = -1
        for i in range(len(self._names)):
            if i in self.details:
                self.simplify_map[i] = -1

    @property
    def parent(self):
        if self._parent is None:
            self._parent = self.original_parent[self.corps].copy()
            for i in range(self._parent.shape[0]):
                if i >= 1:
                    self._parent[i] = self.simplify_map[self._parent[i]]
            self._parent = tuple(self._parent)
        return self._parent

    @property
    def offsets(self):
        return torch.tensor(self._offsets[self.corps], dtype=torch.float)

    @property
    def names(self):
        return self.simplified_name

    @property
    def ee_id(self):
        raise Exception("Abaddoned")


class BVH_file:
    def __init__(
        self,
        file_path,
        skeleton_conf=None,
        requires_contact=False,
        joint_reduction=True,
        auto_scale=True,
        scale=None,
    ):
        self.anim = bvh_io.load(file_path)
        self._names = self.anim.names
        self.frametime = self.anim.frametime
        if requires_contact or joint_reduction:
            assert (
                skeleton_conf is not None
            ), "Please provide a skeleton configuration for contact or joint reduction"
        self.skeleton = Skeleton(
            self.anim.names,
            self.anim.parent,
            self.anim.offsets,
            joint_reduction,
            skeleton_conf,
        )

        # Downsample to 30 fps for our application
        if self.frametime < 0.0084:
            self.frametime *= 2
            self.anim.positions = self.anim.positions[::2]
            self.anim.rotations = self.anim.rotations[::2]
        # if self.frametime < 0.017:
        #     self.frametime *= 2
        #     self.anim.positions = self.anim.positions[::2]
        #     self.anim.rotations = self.anim.rotations[::2]

        self.requires_contact = requires_contact

        if requires_contact:
            self.contact_names = self.skeleton.contact_names
        else:
            self.contact_names = []

        self.fk = ForwardKinematicsJoint(self.skeleton.parent, self.skeleton.offsets)
        self.writer = WriterWrapper(
            self.skeleton.parent, self.skeleton.offsets, self.frametime
        )

        self.auto_scale = auto_scale
        if auto_scale:
            if scale is None:
                self.scale = 1.0 / np.ceil(self.skeleton.offsets.max().cpu().numpy())
                # print(f'rescale the skeleton with scale: {self.scale}')
            else:
                self.scale = scale
                # print(f'rescale the skeleton with other scale: {self.scale}')
            self.rescale(self.scale)
        else:
            self.scale = 1.0

        self.FK_motion = self.joint_position()

        if self.requires_contact:
            gl_pos = self.joint_position()
            self.contact_label = foot_contact(
                gl_pos[:, self.skeleton.contact_id],
                threshold=self.skeleton.contact_threshold,
            )
            self.gl_pos = gl_pos

    def local_pos(self):
        gl_pos = self.joint_position()
        local_pos = gl_pos - gl_pos[:, 0:1, :]
        return local_pos[:, 1:]

    def rescale(self, ratio):
        self.anim.offsets *= ratio
        self.anim.positions *= ratio

    def to_tensor(self, repr="euler", rot_only=False):
        if repr not in ["euler", "quat", "quaternion", "repr6d"]:
            raise Exception("Unknown rotation representation")
        positions = self.get_position()
        rotations = self.get_rotation(repr=repr)

        if rot_only:
            return rotations.reshape(rotations.shape[0], -1)

        if self.requires_contact:
            virtual_contact = torch.zeros_like(
                rotations[:, : len(self.skeleton.contact_id)]
            )
            virtual_contact[..., 0] = self.contact_label
            rotations = torch.cat([rotations, virtual_contact], dim=1)

        rotations = rotations.reshape(rotations.shape[0], -1)
        return torch.cat((rotations, positions), dim=-1)

    def joint_position(self):
        positions = torch.tensor(self.anim.positions[:, 0, :], dtype=torch.float)
        rotations = self.anim.rotations[:, self.skeleton.corps, :]
        rotations = Quaternions.from_euler(np.radians(rotations)).qs
        rotations = torch.tensor(rotations, dtype=torch.float)
        j_loc = self.fk.forward(rotations, positions)
        return j_loc

    def get_rotation(self, repr="quat"):
        rotations = self.anim.rotations[:, self.skeleton.corps, :]
        if repr == "quaternion" or repr == "quat" or repr == "repr6d":
            rotations = Quaternions.from_euler(np.radians(rotations)).qs
            rotations = torch.tensor(rotations, dtype=torch.float)
        if repr == "repr6d":
            rotations = quat2repr6d(rotations)
        if repr == "euler":
            rotations = torch.tensor(rotations, dtype=torch.float)
        return rotations

    def get_position(self):
        return torch.tensor(self.anim.positions[:, 0, :], dtype=torch.float)

    def dfs(self, x, vis, dist):
        fa = self.skeleton.parent
        vis[x] = 1
        for y in range(len(fa)):
            if (fa[y] == x or fa[x] == y) and vis[y] == 0:
                dist[y] = dist[x] + 1
                self.dfs(y, vis, dist)

    def get_neighbor(self, threshold, enforce_contact=False):
        fa = self.skeleton.parent
        neighbor_list = []
        for x in range(0, len(fa)):
            vis = [0 for _ in range(len(fa))]
            dist = [0 for _ in range(len(fa))]
            self.dfs(x, vis, dist)
            neighbor = []
            for j in range(0, len(fa)):
                if dist[j] <= threshold:
                    neighbor.append(j)
            neighbor_list.append(neighbor)

        contact_list = []
        if self.requires_contact:
            for i, p_id in enumerate(self.skeleton.contact_id):
                v_id = len(neighbor_list)
                neighbor_list[p_id].append(v_id)
                neighbor_list.append(neighbor_list[p_id])
                contact_list.append(v_id)

        root_neighbor = neighbor_list[0]
        id_root = len(neighbor_list)

        if enforce_contact:
            root_neighbor = root_neighbor + contact_list
            for j in contact_list:
                neighbor_list[j] = list(set(neighbor_list[j]))

        root_neighbor = list(set(root_neighbor))
        for j in root_neighbor:
            neighbor_list[j].append(id_root)
        root_neighbor.append(id_root)
        neighbor_list.append(root_neighbor)  # Neighbor for root position
        return neighbor_list

    def get_node_index(self, node_names):
        """
        Input:
        node_names: list of strings, the names of the nodes.

        Output:
        node_index: list of integers, the indices of the nodes in the skeleton.
        """

        def find_partial_match(target, lst):
            for index, item in enumerate(lst):
                if item[:-5] == target:
                    return index
            import pdb; pdb.set_trace()
            print("Warning: Can't find the node:", target)
            return -1

        node_index = []
        for name in node_names:
            node_index.append(find_partial_match(name, self.skeleton._names))
        return node_index

    def get_matching_mask(self, node_names, repr, rot_only=False):
        """
        Input:
        node_indexs: list of integers, the indices of the nodes in the skeleton.
        repr: string, rotation representation, support ['quat', 'repr6d', 'euler']
        rot_only: bool, whether to return only the rotation part.
        """
        node_indexs = self.get_node_index(node_names)

        position_mask = torch.zeros_like(self.get_position()[0])
        rotation_mask = torch.zeros_like(self.get_rotation(repr=repr))
        rotation_mask = rotation_mask.reshape(rotation_mask.shape[0], -1)[0]

        num_of_rotations = rotation_mask.shape[0]

        mask_map = []

        if 0 in node_indexs:
            position_mask = torch.ones_like(position_mask)

        if repr == "quat":
            n_rot = 4
        elif repr == "repr6d":
            n_rot = 6
        elif repr == "euler":
            n_rot = 3

        for idx in node_indexs:
            rotation_mask[idx * n_rot : (idx + 1) * n_rot] = 1
            mask_map.append(
                {
                    "from": idx * n_rot,
                    "to": (idx + 1) * n_rot,
                }
            )
        if not rot_only and 0 in node_indexs:
            mask_map.append(
                {
                    "from": num_of_rotations,
                    "to": num_of_rotations + position_mask.shape[0],
                }
            )

        if rot_only:
            return rotation_mask, mask_map
        else:
            return torch.cat((rotation_mask, position_mask), dim=-1), mask_map

    @staticmethod
    def get_zero_6D_rot():
        return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def get_zero_6D_rot_zero_pos(self):
        N = len(self._names)
        zeros_root_pos = torch.zeros(3)
        zeros_6D_rot = self.get_zero_6D_rot().repeat(N)
        return torch.cat((zeros_6D_rot, zeros_root_pos), dim=-1)

    def get_motion_feature_with_name(self, node_names, repr, rot_only=False):
        """
        Input:
        node_indexs: list of integers, the indices of the nodes in the skeleton.
        repr: string, rotation representation, support ['quat', 'repr6d', 'euler']
        rot_only: bool, whether to return only the rotation part.
        """
        # print(node_names)
        node_indexs = self.get_node_index(node_names)

        position_mask = torch.zeros_like(self.get_position()[0])
        rotation_mask = torch.zeros_like(self.get_rotation(repr=repr))
        rotation_mask = rotation_mask.reshape(rotation_mask.shape[0], -1)[0]

        num_of_rotations = rotation_mask.shape[0]

        mask_map = []

        if 0 in node_indexs:
            position_mask = torch.ones_like(position_mask)

        if repr == "quat":
            n_rot = 4
        elif repr == "repr6d":
            n_rot = 6
        elif repr == "euler":
            n_rot = 3

        for idx in node_indexs:
            rotation_mask[idx * n_rot : (idx + 1) * n_rot] = 1
            mask_map.append(
                [
                    {
                        "from": idx * n_rot,
                        "to": (idx + 1) * n_rot,
                    }
                ]
            )
            if not rot_only and idx == 0:
                mask_map.append(
                    [
                        {
                            "from": idx * n_rot,
                            "to": (idx + 1) * n_rot,
                        },
                        {
                            "from": num_of_rotations,
                            "to": num_of_rotations + position_mask.shape[0],
                        },
                    ]
                )

        return mask_map

    def get_contact_position(self):
        FK_motion = self.FK_motion
        contact_position = FK_motion[:, self.skeleton.contact_id]

        return contact_position

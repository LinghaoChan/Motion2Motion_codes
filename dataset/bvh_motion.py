import os
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F
from dataset.motion import MotionData
from .bvh.bvh_parser import BVH_file
from utils.joint_names import KeyJoints


## Some skeleton configurations
crab_dance_corps_names = [
    "ORG_Hips",
    "ORG_BN_Bip01_Pelvis",
    "DEF_BN_Eye_L_01",
    "DEF_BN_Eye_L_02",
    "DEF_BN_Eye_L_03",
    "DEF_BN_Eye_L_03_end",
    "DEF_BN_Eye_R_01",
    "DEF_BN_Eye_R_02",
    "DEF_BN_Eye_R_03",
    "DEF_BN_Eye_R_03_end",
    "DEF_BN_Leg_L_11",
    "DEF_BN_Leg_L_12",
    "DEF_BN_Leg_L_13",
    "DEF_BN_Leg_L_14",
    "DEF_BN_Leg_L_15",
    "DEF_BN_Leg_L_15_end",
    "DEF_BN_Leg_R_11",
    "DEF_BN_Leg_R_12",
    "DEF_BN_Leg_R_13",
    "DEF_BN_Leg_R_14",
    "DEF_BN_Leg_R_15",
    "DEF_BN_Leg_R_15_end",
    "DEF_BN_leg_L_01",
    "DEF_BN_leg_L_02",
    "DEF_BN_leg_L_03",
    "DEF_BN_leg_L_04",
    "DEF_BN_leg_L_05",
    "DEF_BN_leg_L_05_end",
    "DEF_BN_leg_L_06",
    "DEF_BN_Leg_L_07",
    "DEF_BN_Leg_L_08",
    "DEF_BN_Leg_L_09",
    "DEF_BN_Leg_L_10",
    "DEF_BN_Leg_L_10_end",
    "DEF_BN_leg_R_01",
    "DEF_BN_leg_R_02",
    "DEF_BN_leg_R_03",
    "DEF_BN_leg_R_04",
    "DEF_BN_leg_R_05",
    "DEF_BN_leg_R_05_end",
    "DEF_BN_leg_R_06",
    "DEF_BN_Leg_R_07",
    "DEF_BN_Leg_R_08",
    "DEF_BN_Leg_R_09",
    "DEF_BN_Leg_R_10",
    "DEF_BN_Leg_R_10_end",
    "DEF_BN_Bip01_Pelvis",
    "DEF_BN_Bip01_Pelvis_end",
    "DEF_BN_Arm_L_01",
    "DEF_BN_Arm_L_02",
    "DEF_BN_Arm_L_03",
    "DEF_BN_Arm_L_03_end",
    "DEF_BN_Arm_R_01",
    "DEF_BN_Arm_R_02",
    "DEF_BN_Arm_R_03",
    "DEF_BN_Arm_R_03_end",
]


# import utils.confs as confs

# skeleton_confs = confs.skeleton_confs


class BVHMotion:
    def __init__(
        self,
        bvh_file,
        skeleton_name=None,
        repr="quat",
        use_velo=True,
        keep_up_pos=False,
        up_axis="Y_UP",
        padding_last=False,
        requires_contact=False,
        joint_reduction=False,
        using_mask=False,
        scale=None,
        mapping_file=None,
    ):
        """
        BVHMotion constructor
        Args:
            bvh_file         : string, bvh_file path to load from
            skelton_name     : string, name of predefined skeleton, used when joint_reduction==True or contact==True
            repr             : string, rotation representation, support ['quat', 'repr6d', 'euler']
            use_velo         : book, whether to transform the joints positions to velocities
            keep_up_pos      : bool, whether to keep y position when converting to velocity
            up_axis          : string, string, up axis of the motion data
            padding_last     : bool, whether to pad the last position
            requires_contact : bool, whether to concatenate contact information
            joint_reduction  : bool, whether to reduce the joint number
            using_mask       : bool, whether to use mask for motion matching
        """
        self.bvh_file = bvh_file
        self.skeleton_name = skeleton_name
        if skeleton_name is not None:
            assert (
                skeleton_name in skeleton_confs
            ), f"{skeleton_name} not found, please add a skeleton configuration."
        self.requires_contact = requires_contact
        self.joint_reduction = joint_reduction

        if scale is None:
            self.source = False
        else:
            self.source = True

        self.raw_data = BVH_file(
            bvh_file,
            skeleton_confs[skeleton_name] if skeleton_name is not None else None,
            requires_contact,
            joint_reduction,
            auto_scale=True,
            scale=scale,
        )

        self.motion_data = MotionData(
            self.raw_data.to_tensor(repr=repr).permute(1, 0).unsqueeze(0),
            repr=repr,
            use_velo=use_velo,
            keep_up_pos=keep_up_pos,
            up_axis=up_axis,
            padding_last=padding_last,
            contact_id=self.raw_data.skeleton.contact_id if requires_contact else None,
        )
        if using_mask:
            matching_points = KeyJoints(mapping_file, self.source)
            matching_names = matching_points.joints
            matching_mask, masking_map = self.raw_data.get_matching_mask(
                matching_names, repr=repr
            )
            self.motion_data.loading_matching_mask(matching_mask, masking_map)
        else:
            self.motion_data.loading_matching_mask(None, None)

    @property
    def repr(self):
        return self.motion_data.repr

    @property
    def use_velo(self):
        return self.motion_data.use_velo

    @property
    def keep_up_pos(self):
        return self.motion_data.keep_up_pos

    @property
    def padding_last(self):
        return self.motion_data.padding_last

    @property
    def concat_id(self):
        return self.motion_data.contact_id

    @property
    def n_pad(self):
        return self.motion_data.n_pad

    @property
    def n_contact(self):
        return self.motion_data.n_contact

    @property
    def n_rot(self):
        return self.motion_data.n_rot

    def sample(self, size=None, slerp=False):
        """
        Sample motion data, support slerp
        """
        return self.motion_data.sample(size, slerp)

    def write(self, filename, data):
        """
        Parse motion data into position, velocity and contact(if exists)
        data should be []
        No batch support here!!!
        """
        assert (
            len(data.shape) == 3
        ), "The data format should be [batch_size x n_channels x n_frames]"

        if self.n_pad:
            data = data.clone()[:, : -self.n_pad]
        if self.use_velo:
            data = self.motion_data.to_position(data)
        data = data.squeeze().permute(1, 0)
        pos = data[..., -3:]
        rot = data[..., :-3].reshape(data.shape[0], -1, self.n_rot)
        if self.requires_contact:
            contact = rot[..., -self.n_contact :, 0]
            rot = rot[..., : -self.n_contact, :]
        else:
            contact = None

        if contact is not None:
            np.save(filename + ".contact", contact.detach().cpu().numpy())

        # rescale the output
        self.raw_data.rescale(1.0 / self.raw_data.scale)
        pos *= 1.0 / self.raw_data.scale
        self.raw_data.writer.write(
            filename, rot, pos, names=self.raw_data.skeleton.names, repr=self.repr
        )


def load_multiple_dataset(name_list, **kargs):
    with open(name_list, "r") as f:
        names = [line.strip() for line in f.readlines()]
    datasets = []
    for f in names:
        kargs["bvh_file"] = osp.join(osp.dirname(name_list), f)
        datasets.append(BVHMotion(**kargs))
    return datasets


def load_from_path(path, **kargs):
    names = [f for f in os.listdir(path) if f.endswith(".bvh")]

    datasets = []
    for f in names:
        kargs["bvh_file"] = osp.join(path, f)
        datasets.append(BVHMotion(**kargs))
    return datasets

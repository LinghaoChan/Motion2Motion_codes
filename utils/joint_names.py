import json
import torch


class KeyJoints(object):
    def __init__(self, joints=None, source: bool = False):
        self.source = source
        with open(joints) as f:
            joints = json.load(f)
        if type(joints) == list:
            self.joints = joints
        elif type(joints) == dict:
            if source:
                self.joints = [item["source"] for item in joints["mapping"]]
            else:
                self.joints = [item["target"] for item in joints["mapping"]]


def projection_masking(target_map, source_map, len_target, len_source):
    """
    This is a function to create a projection mask based on the source and target mappings.
    Please refer to the Equation 1 in the paper for more details.

    Args:
    source_map: source mapping, list of dict
    target_map: target mapping, list of dict

    Returns:
    projection_mask: the projection mask
    """
    projection_mask = torch.zeros(len_target, len_source)

    for i in range(len(target_map)):
        source_from = source_map[i]["from"]
        source_to = source_map[i]["to"]
        target_from = target_map[i]["from"]
        target_to = target_map[i]["to"]
        enum_source = enumerate(range(source_from, source_to))
        enum_target = range(target_from, target_to)

        for index_t in enum_target:
            _, index_s = next(enum_source)
            projection_mask[index_t, index_s] = 1

    return projection_mask

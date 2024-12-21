import numpy as np
from typing import List, Tuple, Dict, Set
from torch import Tensor
from copy import deepcopy

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from hydra_stretch_msgs.msg import Mask, Masks, HydraVisionPacket

from hydra_seg_ros.utils import viz


def form_label_msg(
    orig_img: Tensor,
    masks: Tensor,
    colors: List[Tuple[int, int, int]],
    bridge: CvBridge,
) -> Image:
    masked_img = viz.masks(orig_img, masks, colors, alpha=1, blacked_out_rest=True)
    label_msg = bridge.cv2_to_imgmsg(masked_img, encoding="rgb8")
    return label_msg


def form_masks_msg(
    class_ids: List[int],
    masks_tensor: Tensor,
    bridge: CvBridge,
) -> Tuple[Masks, Dict]:
    assert len(class_ids) == len(
        masks_tensor
    ), "Need equal number of masks and Ids to form Masks ROS message"
    masks_msg = Masks()
    masks_msg.masks = []
    local_id_to_class = {}
    instance_cnt = 1
    for id, mask in zip(class_ids, masks_tensor):
        m_msg = Mask()
        m_msg.data = bridge.cv2_to_imgmsg(
            mask.detach().cpu().numpy().astype(np.uint8), encoding="mono8"
        )
        m_msg.class_id = int(id)
        masks_msg.masks.append(m_msg)
        # {local_id -> class}
        m_msg.local_id = instance_cnt
        local_id_to_class[id] = instance_cnt
        instance_cnt += 1
    return masks_msg, local_id_to_class


def pack_vision_msgs(
    cam_info_msg: CameraInfo,
    color_msg: Image,
    depth_msg: Image,
    label_msg: Image,
    masks_msg: Masks,
) -> Tuple[CameraInfo, HydraVisionPacket]:
    # TODO: Check the dimensions and type
    vision_packet_msg = HydraVisionPacket()

    now = rospy.Time.now()

    vision_packet_msg.color = deepcopy(color_msg)
    vision_packet_msg.depth = deepcopy(depth_msg)
    vision_packet_msg.label = deepcopy(label_msg)
    if masks_msg:
        vision_packet_msg.masks = masks_msg

    # Sync all messages' headers with depth
    vision_packet_msg.depth.header.stamp = now
    vision_packet_msg.color.header = depth_msg.header
    vision_packet_msg.label.header = depth_msg.header
    vision_packet_msg.masks.header = depth_msg.header
    if masks_msg.masks:
        for i, _ in enumerate(masks_msg.masks):
            masks_msg.masks[i].data.header = depth_msg.header

    cam_info_msg.header.stamp = now
    return cam_info_msg, vision_packet_msg

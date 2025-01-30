import cv2
import numpy as np
from typing import Optional, List, Tuple
from torch import Tensor
from copy import deepcopy

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from hydra_stretch_msgs.msg import Mask, Masks, HydraVisionPacket

from hydra_seg_ros.utils import viz


def form_label_msg(
    orig_img: np.ndarray,
    masks: Tensor,
    colors: List[List[int]],
    bridge: CvBridge,
) -> Image:
    masked_img = viz.masks(orig_img, masks, colors, alpha=1, blacked_out_rest=True)
    label_msg = bridge.cv2_to_imgmsg(masked_img, encoding="rgb8")
    return label_msg


def form_masks_msg(
    class_ids: List[int],
    masks_tensor: Tensor,
    bridge: CvBridge,
    height: int,
    width: int,
) -> Masks:
    assert len(class_ids) == len(
        masks_tensor
    ), "Need equal number of masks and Ids to form Masks ROS message"
    masks_msg = Masks()
    masks_msg.masks = []
    for id, mask in zip(class_ids, masks_tensor):
        m_msg = Mask()
        mask_cv = mask.detach().cpu().numpy().astype(np.uint8)
        m_msg.data = bridge.cv2_to_imgmsg(
            cv2.resize(mask_cv, (width, height)), encoding="mono8"
        )
        m_msg.class_id = int(id)
        masks_msg.masks.append(m_msg)
    return masks_msg


def pack_vision_msgs(
    map_view_id: int,
    cam_info_msg: CameraInfo,
    color_msg: Image,
    depth_msg: Image,
    label_msg: Image,
    masks_msg: Optional[Masks],
) -> Tuple[CameraInfo, HydraVisionPacket]:
    
    vision_packet_msg = HydraVisionPacket()

    now = rospy.Time.now()

    vision_packet_msg.map_view_id = map_view_id
    vision_packet_msg.color = deepcopy(color_msg)
    vision_packet_msg.depth = deepcopy(depth_msg)
    vision_packet_msg.label = deepcopy(label_msg)
    # Sync all messages' headers with depth
    vision_packet_msg.depth.header.stamp = now
    vision_packet_msg.color.header = depth_msg.header
    vision_packet_msg.label.header = depth_msg.header
    vision_packet_msg.masks.header = depth_msg.header
    cam_info_msg.header.stamp = now

    # Assign masks_msg if available
    if masks_msg:
        if masks_msg.masks:
            for i, _ in enumerate(masks_msg.masks):
                masks_msg.masks[i].data.header = depth_msg.header
        vision_packet_msg.masks = masks_msg

    return cam_info_msg, vision_packet_msg

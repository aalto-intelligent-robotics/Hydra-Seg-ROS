import message_filters
import rospy
from cv_bridge import CvBridge
from pathlib import Path
import numpy as np
import torch

from ultralytics import YOLO

from sensor_msgs.msg import Image, CameraInfo
from hydra_stretch_msgs.msg import HydraVisionPacket

from hydra_seg_ros.utils import labels, ros_utils


class YoloRosNode:
    def __init__(self):
        rospy.init_node("yolo_ros_node")
        self.init_ros()
        self.model = YOLO(self.model_path, verbose=False)
        rospy.loginfo("Starting YoloRosNode.")
        if self.label_space_name == "kitchen":
            self.label_space = labels.KITCHEN_LABEL_SPACE
        elif self.label_space_name == "flat":
            self.label_space = labels.FLAT_LABEL_SPACE
        else:
            raise NotImplementedError(f"{self.label_space_name} is not implemented")

    def init_ros(self):
        self.model_path = str(
            rospy.get_param(
                "~model_path", Path.home() / "models/yolo/yolo11l-seg.engine"
            )
        )
        self.conf = rospy.get_param("~conf", 0.7)
        self.label_space_name = rospy.get_param("~label_space", "kitchen")
        self.viz_label = rospy.get_param("~viz_label", "true")

        self.cam_info_sub = message_filters.Subscriber("~cam_info", CameraInfo)
        self.color_sub = message_filters.Subscriber("~colors", Image)
        self.depth_sub = message_filters.Subscriber("~depth", Image)

        if self.viz_label:
            self.label_pub = rospy.Publisher("~label", Image, queue_size=10)
        self.cam_info_pub = rospy.Publisher("~camera_info", CameraInfo, queue_size=10)
        self.vision_packet_pub = rospy.Publisher(
            "~vision_packet", HydraVisionPacket, queue_size=10
        )

        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.cam_info_sub, self.color_sub, self.depth_sub],
            10,
            0.1,
            allow_headerless=False,
        )
        self.synchronizer.registerCallback(self.vision_callback)
        self.bridge = CvBridge()

    def vision_callback(
        self, cam_info_msg: CameraInfo, color_msg: Image, depth_msg: Image
    ):
        color_cv = self.bridge.imgmsg_to_cv2(color_msg)
        height, width, _ = color_cv.shape
        pred = self.model(color_cv, conf=self.conf)
        class_idcs = pred[0].boxes.cls.cpu().numpy().astype(np.uint8)
        masks = torch.tensor([])
        if pred[0].masks is not None:
            masks = pred[0].masks.data
        valid_class_idcs = []
        valid_masks = torch.tensor([]).to(masks.device)
        for class_id, mask in zip(class_idcs, masks):
            if class_id in self.label_space:
                valid_class_idcs.append(class_id)
                valid_masks = torch.cat([valid_masks, mask[None, :, :]], dim=0)

        masks_msg, _ = ros_utils.form_masks_msg(
            valid_class_idcs, valid_masks, height, width, self.bridge
        )
        label_msg = ros_utils.form_label_msg(
            pred[0].orig_img,
            valid_masks,
            # 0 is background, instance count from 1
            [labels.COCO_COLORS[x] for x in range(1, len(valid_masks) + 1)],
            self.bridge,
        )

        cam_info_msg_pub, vision_packet_msg = ros_utils.pack_vision_msgs(
            cam_info_msg, color_msg, depth_msg, label_msg, masks_msg
        )
        self.cam_info_pub.publish(cam_info_msg_pub)
        self.vision_packet_pub.publish(vision_packet_msg)
        if self.viz_label:
            self.label_pub.publish(vision_packet_msg.label)


def main():
    _ = YoloRosNode()
    rospy.spin()

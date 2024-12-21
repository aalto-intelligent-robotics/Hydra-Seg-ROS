import message_filters
import rospy
from cv_bridge import CvBridge
from pathlib import Path
import numpy as np
import torch
import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from ultralytics.engine.results import Masks

from sensor_msgs.msg import Image, CameraInfo
from hydra_stretch_msgs.msg import Mask, Masks, HydraVisionPacket

from hydra_seg_ros.utils import viz, labels, ros_utils


class YoloRosNode:
    def __init__(self):
        rospy.init_node("yolo_ros_node")
        self.init_ros()
        rospy.loginfo("Starting YoloRosNode.")

    def init_ros(self):
        self.model_path = rospy.get_param(
            "~model_path", Path.home() / "models/yolo/yolo11l-seg.engine"
        )
        self.model = YOLO(self.model_path, verbose=False)
        self.conf = rospy.get_param("~conf", 0.7)
        self.depth_threshold = rospy.get_param("~depth_threshold", -1)
        self.label_space = rospy.get_param("~label_space", "coco")
        self.viz_label = rospy.get_param("~viz_label", "true")
        self.color_mesh_by_label = rospy.get_param("~color_mesh_by_label", "false")

        self.cam_info_sub = message_filters.Subscriber("~cam_info", CameraInfo)
        self.color_sub = message_filters.Subscriber("~colors", Image)
        self.depth_sub = message_filters.Subscriber("~depth", Image)

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
        depth_cv = self.bridge.imgmsg_to_cv2(depth_msg)
        height, width, _ = color_cv.shape
        pred = self.model(color_cv, conf=self.conf)
        class_idcs = pred[0].boxes.cls.cpu().numpy().astype(np.uint8)
        masks = torch.tensor([])
        if pred[0].masks is not None:
            masks = pred[0].masks.data
        if self.depth_threshold != -1 and depth_cv is not None:
            masks_np = masks.detach().cpu().numpy()
            masks = torch.tensor(
                [
                    filter_depth(mask, depth_cv, self.depth_threshold)
                    for mask in masks_np
                ]
            )
        label_msg = ros_utils.form_label_msg(
            pred[0].orig_img,
            masks,
            [labels.COCO_COLORS[x] for x in class_idcs],
            self.bridge,
        )
        if self.color_mesh_by_label:
            color_msg = label_msg
        masks_msg = ros_utils.form_masks_msg(class_idcs, masks, self.bridge, height, width)

        # color_cv = cv2.resize(color_cv, (640, 480))
        # depth_cv = cv2.resize(depth_cv, (640, 480))
        # color_msg = self.bridge.cv2_to_imgmsg(color_cv, encoding="rgb8")
        # depth_msg = self.bridge.cv2_to_imgmsg(depth_cv)
        cam_info_msg_pub, vision_packet_msg = ros_utils.pack_vision_msgs(
            cam_info_msg, color_msg, depth_msg, label_msg, masks_msg, self.bridge
        )
        # cam_info_msg_pub.height = 480
        # cam_info_msg_pub.width = 640
        self.cam_info_pub.publish(cam_info_msg_pub)
        self.vision_packet_pub.publish(vision_packet_msg)
        if self.viz_label:
            self.label_pub.publish(vision_packet_msg.label)


def main():
    yolo_ros_node = YoloRosNode()
    rospy.spin()

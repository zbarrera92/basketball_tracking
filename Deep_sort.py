import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, Load_Custom_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names, detect_video
from yolov3.configs import *
from yolov3.yolov3 import Create_Yolov3
import time

from object_tracker import Object_tracking
from slice_video import create_time_stamp_windows, video_slicer

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

yolo_custom = Load_Custom_Yolo_model()
yolo = Load_Yolo_model()
video_path = "./basketball-videos/made-basket-tracker/netsbuckspart1.mp4"
save_path = "./basketball-videos/made-basket-tracker/netsbuckspart1_tracked_1.mp4"

video_info = Object_tracking(yolo, video_path, save_path, input_size=YOLO_INPUT_SIZE, show=False,
                              iou_threshold=0.1, rectangle_colors=(255,0,0), Track_only = ["person"], custom_yolo=yolo_custom)

# Grab the dict that keeps te amount of baskets made by each team
baskets_dict = video_info["baskets_dict"]
baskets_dict

# Grab the list of frames where a basket was scored
basket_frame_list = video_info["basket_frame_list"]
basket_frame_list

# use the previous frames to create "frame windows" to feed into our video clipper
time_stamp_windows = create_time_stamp_windows(basket_frame_list, 5, 2, 30)
time_stamp_windows

video_path = "./basketball-videos/made-basket-tracker/netsbuckspart1_tracked_1.mp4"
save_path = "./basketball-videos/made-basket-tracker/netsbuckspart1_sliced.mp4"

# slice up our video
video_slicer(video_path, save_path, time_stamp_windows["start_frames"], time_stamp_windows["end_frames"])
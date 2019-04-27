import os
import cv2
from PIL import Image
from identify.identify_obj import identify_obj
from detection.detection_obj import detection_obj
from tracking.tracking_obj import tracking_obj

if __name__ == '__main__':

    video_path = './videos/BlurCar4/img'
    detection_net_path = './pretrained/VGGnet_fast_rcnn_iter_70000.h5'
    tracking_net_path = './pretrained/model.pth'
    examplar_path = 'examplar.png'
    img_files = os.listdir(video_path)
    img_files.sort()

    # detection
    target_boxs = detection_obj(os.path.join(video_path, img_files[0]), detection_net_path)

    # identify
    box = identify_obj(examplar_path, os.path.join(video_path, img_files[0]), target_boxs)
    box = [box[0], box[1], box[2]-box[0], box[3] - box[1]]

    # tracking
    tracking_obj(video_path, tracking_net_path, box)
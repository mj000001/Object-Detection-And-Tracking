import cv2
from PIL import Image
import numpy as np
from detection.faster_rcnn import network
from detection.faster_rcnn.faster_rcnn import FasterRCNN
from detection.faster_rcnn.utils.timer import Timer
import os

def detection_obj(im_file, model_file):
    image = cv2.imread(im_file)
    detector = FasterRCNN()
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')

    t = Timer()
    t.tic()
    dets, scores, classes = detector.detect(image, 0.7)
    runtime = t.toc()
    print('detection stage spend: {}s'.format(runtime))

    return dets
    # crop ROIs and save (test phash) dets : [left_x, left_y, right_x, right_y]
    # for i, det in enumerate(dets):
    #     det = tuple(int(x) for x in det)
    #     cropped = image[det[1]:det[3], det[0]:det[2]]
    #     cv2.imwrite(os.path.join('save_obj','pic'+str(i)+'.png'), cropped)

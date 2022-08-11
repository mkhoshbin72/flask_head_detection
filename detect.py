import argparse
import time
from pathlib import Path

import cv2
import torch
from numpy import random
import numpy as np
import math

from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path


def blur_head(x, img):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    img[c1[1]:c2[1], c1[0]:c2[0]] = cv2.blur(img[c1[1]:c2[1], c1[0]:c2[0]], (100,100))


def img_pad(image, dw, dh):

    h,w,_ = image.shape
    image_rat = w/h

    if image_rat < 1:
        # VERTICAL IMAGE
        new_h = dh
        new_w = new_h * image_rat

        image = cv2.resize(image, (int(np.ceil(new_w)), int(np.ceil(new_h))))
        top, bottom = 0,0
        pad = abs(dw - new_w)
        left, right = pad/2 , pad/2

    if image_rat >= 1:
        # HORIZONTAL IMAGE
        new_w = dw
        new_h = new_w / image_rat

        image = cv2.resize(image, (int(np.ceil(new_w)), int(np.ceil(new_h))))
        left, right = 0,0
        pad = abs(dh - new_h)
        top, bottom = pad/2 , pad/2


    pad_image = cv2.copyMakeBorder(image, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT)
    pad_image = cv2.resize(pad_image, (dw, dh))

    return pad_image

def load_model(weights, device='cpu'):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def detect(image, model):

    weights = '/home/mkh/Files/yolov7_my/head_re.pt'
    device = 'cpu'
    conf_thres = 0.25
    iou_thres = 0.6
    classes = ['head']
    imgsz = 640

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    
    im0 = image.copy()
    img = img_pad(image, imgsz, imgsz)
    img = np.moveaxis(img, -1, 0)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                blur_head(xyxy, im0)

    return im0


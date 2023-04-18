# -*-coding:utf-8-*-
import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from xml.etree.ElementTree import Element, SubElement, ElementTree
import numpy as np
import platform as pf
import psutil
import PIL
import pandas as pd
import seaborn as sns
import cv2
import json
import base64

def indent(elem, level=0):  #
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def ToF(file, cat):
    if cat == '00000000':
        output = "N"
    #elif str(file).split('_')[2] == cat:
    #    output = "T"
    else:
        output = "F"
    
    return output
    
def detect(opt, buf, save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt, save_xml = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt, opt.save_xml
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'], strict=False)
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'],
                               strict=False)  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    
    #import base64
#import json
#import cv2
#import numpy as np

#response = json.loads(open('./0.json', 'r').read())
#string = response['img']
#jpg_original = base64.b64decode(string)
#jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
#img = cv2.imdecode(jpg_as_np, flags=1)
#cv2.imwrite('./0.jpg', img)
    #


    jpg_as_np = np.frombuffer(buf, dtype=np.uint8)
    im0s = cv2.imdecode(jpg_as_np, 1)  # BGR
    # Padded resize
    img = letterbox(im0s, new_shape=320)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Get names and colors
    names = load_classes(opt.names)
    #img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    #_ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    img = torch.from_numpy(img).to(device) #이미지 인식
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = torch_utils.time_synchronized()
    pred = model(img, augment=opt.augment)[0]
    t2 = torch_utils.time_synchronized()

    # to float
    if half:
        pred = pred.float()

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    ret = []
    # Process detections
    for i, det in enumerate(pred):  # detections for image i
        s, im0 = '', im0s

        if det is not None and len(det):
            # Rescale boxes from imgsz to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            count = 0

            lst = []
            # Print results
            for c in det[:, -1].unique():
                n = int((det[:, -1] == c).sum())  # detections per class
                s += '%g %s, ' % (n, names[int(c)])  # add to string
                lst.append((n, names[int(c)]))
            
            ret.append(lst)

    return ret


def create(b64str):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-403cls.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/403food.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/best_403food_e200b150v2.pt', help='weights path')
    parser.add_argument('--source', type=str, default='sample', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save-xml', action='store_true', help='save results to *.xml')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(len(os.listdir(opt.source)))

    data = base64.b64decode(b64str)
    with torch.no_grad():
        return detect(opt, data)

def lambda_handler(event, context):

    data = json.loads(event['body'])["img"]
    ret = create(data)

    return {
            'headers': { "Content-Type": "text/json" },
            'statusCode': 200,
            'body': ret
        }

#if __name__ == '__main__':
#    from pathlib import Path
#    ff = Path('sample/a.png').read_bytes()
#    b64str = base64.b64encode(ff)
#
#    ret = create(b64str)
#
#    for i in range(len(ret)):
#        print(*ret[i])
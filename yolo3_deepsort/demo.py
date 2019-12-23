#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from tools.plot_utils import draw_one_box as draw_box
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def track_video(yolo,video_handle = 'model_data/Crossroad.mp4', det_only = False):
    '''Track objects in a video
    Parameters:
    -----------
        yolo: a class define YOLO algorithm
        video_handle: 0 or video path
    Return:
    -------
        None
    '''
   # Definition of the parameters
    nms_max_overlap = 1.0
    max_distance=0.3
    lambda0 = 0.1
    nn_budget=None

   # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    # metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric_mode="cosine",max_cosine_distance=max_distance,
                      lambda0 = lambda0,nn_budget=nn_budget)

    writeVideo_flag = False
    video_capture = cv2.VideoCapture(video_handle)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output/output.avi', fourcc, 15, (w, h))
        list_file = open('output/detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        image = Image.fromarray(frame[...,::-1]) # bgr to rgb,CV to PIL
        boxes,classes,scores = yolo.detect_image(image)# detect

        # detect and track
        features = encoder(frame,boxes)
        detections = [Detection(bbox, score, feature,class_)
                        for bbox,score,feature,class_ in zip(boxes,scores,features,classes)]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # detect only, not to track
        if det_only:
            for bbox,cat in zip(boxes,classes):
                color = yolo.colors[yolo.class_names.index(cat)]
                bbox = np.array(bbox)
                bbox[2:] = bbox[:2] + bbox[2:]#tlwh to tlbr
                image = draw_box(image,bbox,' ',cat,color)
        else:
            for i,track in enumerate(tracker.tracks):
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                image = draw_box(image,bbox,track.track_id,track.object_class,yolo.colors[yolo.class_names.index(track.object_class)])
        img_show = np.asarray(image)
        cv2.imshow('demo', img_show)

        if writeVideo_flag:
            # save a frame
            out.write(img_show)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    yolo = YOLO(model_path = 'model_data/trained_weights_coco.h5',
                classes_path = 'model_data/classes_name.txt',
                weights_only = True,
                score = 0.3,
                iou = 0.3)
    video_handle = 'model_data/Crossroad.mp4'
    track_video(yolo,video_handle,det_only = True)

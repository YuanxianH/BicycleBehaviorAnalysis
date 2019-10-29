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

# from deep_sort import preprocessing
# from deep_sort import nn_matching
from deep_sort.detection import Detection,NMS
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def main(yolo):

   # Definition of the parameters
    nms_max_overlap = 1.0
    max_cosine_distance=0.3
    nn_budget=None

   # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    # metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric_mode="cosine",max_cosine_distance=max_cosine_distance,nn_budget=nn_budget)

    writeVideo_flag = False
    video_capture = cv2.VideoCapture(0)

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
        boxs,classes,scores = yolo.detect_image(image)# detect

        features = encoder(frame,boxs)

        detections = [Detection(bbox, score, feature,class_)
                        for bbox,score,feature,class_ in zip(boxs,scores,features,classes)]
        detections = NMS(detections,nms_max_overlap = nms_max_overlap)# non-max suppression

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for i,track in enumerate(tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            cv2.putText(frame,classes[i],(int(bbox[0])+10, int(bbox[1])),0, 5e-3 * 200, (0,0,255),2)


        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        cv2.imshow('demo', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
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
    main(YOLO())

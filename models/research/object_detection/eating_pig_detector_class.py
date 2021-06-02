######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import sys
import feeder_delimiter

#Grabscreen stuff
from grabscreen import grab_screen
import pyautogui

#excel report stuff
import xlsxwriter as excel

points = False

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

import numpy as np
import time

from sort import *



# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 2

#feeder color
FEEDER_COLOR = (66, 244, 98)

DRAW_RECTS = True
DRAW_FEEDER = True
DRAW_PIG_IDS = True

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

import threading

class PigDetector():
    def __init__(self):
        self.finished = False
        self.processing_progress = 0
        self.processed_frames_counter = 0
        self.total_frames = 0
        self.last_detection_boxes = []
        self.current_detection_boxes = []
        self.current_detection_pigs = []
        self.last_detection_pigs = []
        self.write_on_video =  False
        self.paused = False
        self.started =  False
        self.generate_excel_table = True
        self.eating_list = []
        self.disappeared_list_pigs = []
        self.disappeared_list_boxes = []
        self.eating_feeder_pig = []
        self.eating_feeder_pig_count = []
        self.multiTracker = None
        self.tracked_pigs = []
        self.tracked_boxes = []
        self.frame_step = 5
        self.scale_factor = 0.5
                # Load the Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                
                self.serialized_graph = fid.read()
                
                self.od_graph_def.ParseFromString(self.serialized_graph)
                
                tf.import_graph_def(self.od_graph_def, name='')
                

            self.sess = tf.Session(config=config,graph=self.detection_graph)

        self.mot_tracker = Sort() #create instance of the SORT tracker
            

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                


    
    def detect_eating_pig_(self,frame,feeders, draw_feeder = True):
        
        
        for i, f in enumerate(feeders):
            
            frame = self.draw_feeder_area(frame,feeders[i],FEEDER_COLOR,i+1,draw_feeder = draw_feeder)
            boxes_image = np.zeros(frame.shape, np.uint8)
            boxes_image = cv2.cvtColor(boxes_image, cv2.COLOR_BGR2GRAY)
            for j, b in enumerate(self.current_detection_boxes):
                
                height = np.size(frame, 0)
                width = np.size(frame, 1)
                
                left, top, right, bottom = b[0], b[1], b[2], b[3]
                
                pig_rect = [(left, top), (left, bottom), (right, bottom),(right, top)]

                intersect_value = self._calculate_intersection(frame,feeders[i],pig_rect)

                #frame  = cv2.fillPoly(frame, np.array([pig_rect]), (255,255,255))
                #cv2.rectangle(frame,(left, top),(right, bottom),(0,255,0),2)
                
                if(intersect_value>3000*self.scale_factor):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame,"Feeding...",(b[0],b[1]), font, 1,(0,255,255),2,cv2.LINE_AA)
                    pig_idx = self.current_detection_pigs[j]
                    coord = str(i+1) + str(pig_idx)
                    if coord in self.eating_feeder_pig:
                        self.eating_feeder_pig_count[self.eating_feeder_pig.index(coord)] += (1/self.FPS)*self.frame_step
                        print('Table:')
                        print(self.eating_feeder_pig_count[self.eating_feeder_pig.index(coord)])
                    else:
                        self.eating_feeder_pig.append(coord)
                        self.eating_feeder_pig_count.append((1/self.FPS)*self.frame_step)
                    #print(self.eating_feeder_pig)
                    #print(self.eating_feeder_pig_count)
                    #boxes_image = cv2.bitwise_or(boxes_image,box_image)
        
        return frame

    def _calculate_intersection(self,frame,feeder_poly,pig_rect):
        feeder_image = np.zeros(frame.shape, np.uint8)
        cv2.fillPoly(feeder_image, np.array([feeder_poly]), (255,255,255))
        feeder_image = cv2.cvtColor(feeder_image, cv2.COLOR_BGR2GRAY)
        
        box_image = np.zeros(frame.shape, np.uint8)
        cv2.fillPoly(box_image, np.array([pig_rect]), (255,255,255))
        box_image = cv2.cvtColor(box_image, cv2.COLOR_BGR2GRAY)
        
        and_image = cv2.bitwise_and(feeder_image,box_image)
        return cv2.countNonZero(and_image)
        
        
    def draw_feeder_area(self,frame,feeder,color,n_feeder= None, draw_feeder= True):
        ALPHA = 0.2
        canvasBlack = np.ones(frame.shape, np.uint8)
        # of a filled polygon
        
        cv2.fillPoly(canvasBlack, np.array([feeder]), color)
        if(draw_feeder):
            frame = cv2.addWeighted(canvasBlack,ALPHA,frame, 1-ALPHA,10)
        if(n_feeder is not None):
            cX,cY = self.find_center_of_polygon(canvasBlack)
            cv2.putText(frame,str(n_feeder),(cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

                
    def generate_excel_report(self):
        workbook = excel.Workbook(self.videoPath[0:-4]+'_report.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write(0,0,'Comedouro/Porco')
        #worksheet.write(0,1,'Tempo estimado de alimentação (s)')
        
        
        for i in range(len(self.feeders)):
            worksheet.write(i+1,0,i+1)
            dislen = 1
            curlen = 1
            if(len(self.disappeared_list_pigs)>0):
                dislen = max(self.disappeared_list_pigs)
            if(len(self.last_detection_pigs)>0):
                curlen = max(self.last_detection_pigs)
            nie = max(curlen,dislen)
            for j in range(nie):
                worksheet.write(0,j+1,j+1)
                val = 0.0
                if((str(i+1)+str(j+1)) in self.eating_feeder_pig):
                    idx = self.eating_feeder_pig.index(str(i+1)+str(j+1))
                    val = self.eating_feeder_pig_count[idx]
                worksheet.write(i+1,j+1,round(val,2))
                
        workbook.close()

    def find_center_of_polygon(self,image):
        #cv2.imshow("mask",image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        #cv2.imshow("mask",thresh)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        # loop over the contours
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            return cX, cY     

    def _randomly_enumerate_pigs(self,frame,boxes,scores):
        for j, b in enumerate(boxes[0]):
            if scores[0][j] >= 0.8:
                height = np.size(frame, 0)
                width = np.size(frame, 1)
                
                left, right, top, bottom = int(boxes[0][j][1]*width), int(boxes[0][j][3]*width), int(boxes[0][j][0]*height), int(boxes[0][j][2]*height)
                
                pig_rect = [(left, top), (left, bottom), (right, bottom),(right, top)]

                self.current_detection_boxes.append((left,top,right,bottom))
                self.last_detection_boxes.append((left,top,right,bottom)) 
                self.current_detection_pigs.append(j+1)
                self.last_detection_pigs.append(j+1)

    def enumerate_pigs(self,frame,boxes,scores):
        # for every detection, try to find a corresponding id
        for j, b in enumerate(boxes[0]):
            if scores[0][j] >= 0.8:
                height = np.size(frame, 0)
                width = np.size(frame, 1)
                left, right, top, bottom = int(boxes[0][j][1]*width), int(boxes[0][j][3]*width), int(boxes[0][j][0]*height), int(boxes[0][j][2]*height)
                pig_rect = [(left, top), (left, bottom), (right, bottom),(right, top)]

                rect = (left,top,right,bottom)
                #cv2.rectangle(frame,(left,top),(right,bottom),(255,255,0))
                #try to calculate comparing to last detection overlapping area
                pig_id = self._calculate_corresponding_pig_id(rect)
                # if suitable pig_id is found, ok
                
                if(pig_id>0):
                    self.current_detection_boxes.append(rect)
                    self.current_detection_pigs.append(pig_id)
                #otherwise, try to match with disappeared ones
                else:
                    #print("pig_id:",pig_id)
                    #means a new pig is detected (not compatible with any last detection bo)
                    pig_id = self.find_lowest_new_pig_id()
                    self.current_detection_boxes.append(rect)
                    self.current_detection_pigs.append(pig_id)
         ######################################### AQUI ###########################################               
        #checks for any last detection that was not detected nor matched on the new detection
        # and adds it to the self.disappeared_list_pigs and self.disappeared_list_boxes
        #self.find_and_store_disappeared_pigs()
        #put those disappearing ones on a tracker 
        #self.add_disappeared_pigs_to_tracker(frame)
        
        #copy the disappeared to the current anyways :\ (for printing
        #and tracking purposes, of course)
##        for j,p in enumerate(self.disappeared_list_pigs):
##            self.current_detection_boxes.append(
##                self.disappeared_list_boxes[
##                    self.disappeared_list_pigs.index(p)])
##            self.current_detection_pigs.append(p)
        
        #clean up disappered list!
        self.disappeared_list_pigs = []
        self.disappeared_list_boxes = []

        
        
        
                #if(self.check_for_disappearing_pigs()):
                    
                    #self.disappeared_list_pigs.append(self.find_disappeared_pig())
                    #self.disappeared_list_boxes.append(self.find_disappeared_pig())

    def find_and_store_disappeared_pigs(self):
        #print(self.last_detection_pigs)
        #print(self.current_detection_pigs)
        disappeared_pigs = set(self.last_detection_pigs) - set(self.current_detection_pigs + self.tracked_pigs)
        disappeared_pigs = list(disappeared_pigs)
        
        for i in range(len(disappeared_pigs)):
            self.disappeared_list_pigs.append(disappeared_pigs[i])
            self.disappeared_list_boxes.append(
                self.last_detection_boxes[
                    self.last_detection_pigs.index(
                        disappeared_pigs[i])])
        
        #print(self.disappeared_list_pigs)
        #print(self.disappeared_list_boxes)
    
    def find_lowest_new_pig_id(self):
        pig_id = -1
        if(len(self.current_detection_pigs)>0):
            pig_id =  max(self.current_detection_pigs)
            if(len(self.tracked_pigs)>0):
                pig_id = max(pig_id,max(self.tracked_pigs))
        elif(len(self.tracked_pigs)>0):
            pig_id = max(self.tracked_pigs)

        return pig_id+1

        
    def _calculate_corresponding_pig_id(self,rect):
        overlapping_areas = []
        
        for j,b in enumerate(self.last_detection_boxes):
            a = self.calculate_overlapping_rect_area(rect,b)
            d = self.calculate_center_distance(rect,b)
            
            if(a>0):
                overlapping_areas.append((a,self.last_detection_pigs[j],b,d))
                #print("added to dispute f last:",self.last_detection_pigs[j]," d:",d)
        #also, check the tracked ones        
        for j,b in enumerate(self.tracked_boxes):
            a = self.calculate_overlapping_rect_area(rect,b)
            d = self.calculate_center_distance(rect,b)
            if(a>0):
                overlapping_areas.append((a,self.tracked_pigs[j],b,d))
                #print("added to dispute f tracked:",self.tracked_pigs[j]," d:",d)
        #print("----------------------------------------")
        
        # if any match
        if(len(overlapping_areas)>0):
            #find the best match!
            #(a_max,pig_id,elected_rect) = max(overlapping_areas,key=lambda item:item[0])
            (a_max,pig_id,elected_rect,d_min) = min(overlapping_areas,key=lambda item:item[3])
            already_detected = 0
            prev_box = None
            # check if pig already matched, if so, choose the best one
            if(pig_id in self.current_detection_pigs):
                already_detected = 1
                prev_box = self.current_detection_boxes[self.current_detection_pigs.index(pig_id)]

            #if(pig_id in self.tracked_pigs):
                #already_detected = 2
                #prev_box = self.tracked_boxes[self.tracked_pigs.index(pig_id)]

            if(already_detected==1):
                #calculate if the new area is bigger than the last one
                #print("elected_rect:",elected_rect)
                #print("current_detection_boxes[pig_id]:",self.current_detection_boxes[self.current_detection_pigs.index(pig_id)])
                #print("new rect:",rect)
                #previous_a = self.calculate_overlapping_rect_area(prev_box,elected_rect)
                previous_d = self.calculate_center_distance(prev_box,elected_rect)
                #print("previous_a",previous_a)
                #print("d_min",d_min)
                #print("previous_d",previous_d)
                
                if(previous_d>d_min):
                    
                    print("previous_d>d_min",pig_id )
                    
                    idx = self.current_detection_pigs.index(pig_id)
                    del self.current_detection_pigs[idx]
                    del self.current_detection_boxes[idx]                        
                    
                elif(previous_d==d_min):
                    pass
                else:
                    # overlaps but not good enough
                    pig_id = -1
##                if(previous_a<a_max):
##                    
##                    print("previous_a<a_max",pig_id )
##                    
##                    idx = self.current_detection_pigs.index(pig_id)
##                    del self.current_detection_pigs[idx]
##                    del self.current_detection_boxes[idx]                        
##                    
##                elif(previous_a==a_max):
##                    pass
##                else:
##                    # overlaps but not good enough
##                    pig_id = -1

            #if it its from tracked, remove it before return
            if(pig_id in self.tracked_pigs):
                idx = self.tracked_pigs.index(pig_id)
                del self.tracked_pigs[idx]
                del self.tracked_boxes[idx]
                    
            return pig_id
        else:
            # new pig detected!!
            return -2
    def calculate_overlapping_rect_area(self,a,b):
        #print((a,b))
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        #print(dx*dy)
        if (dx>=0) and (dy>=0):
            return dx*dy
        else:
            return 0

    def calculate_center_distance(self,a,b):
        cax = int(a[0] + (a[2] - a[0])/2)
        cbx = int(b[0] + (b[2] - b[0])/2)
        cay = int(a[1] + (a[3] - a[1])/2)
        cby = int(b[1] + (b[3] - b[1])/2)
        a = np.array((cax,cay))
        b = np.array((cbx,cby))
        return np.sqrt(np.sum(np.square(a-b)))
        
    def draw_rects_and_pigs_ids(self,frame,draw_rects= True,draw_pigs = True):
        #print(self.current_detection_boxes)
        for j,b in enumerate(self.current_detection_boxes):
            if(draw_rects):
                cv2.rectangle(frame,(b[0], b[1]),(b[2], b[3]),(0,255,0),2)
            if(draw_pigs):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,str(self.current_detection_pigs[j]),(b[0] + int((b[2]-b[0])/2),b[1] + int((b[3]-b[1])/2)), font, 1,(0,0,255),2,cv2.LINE_AA)
        for j,b in enumerate(self.tracked_boxes):
            if(draw_rects):
                cv2.rectangle(frame,(b[0], b[1]),(b[2], b[3]),(0,255,255),2)
            if(draw_pigs):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,str(self.tracked_pigs[j]),(b[0] + int((b[2]-b[0])/2),b[1] + int((b[3]-b[1])/2)), font, 1,(0,0,255),2,cv2.LINE_AA)
        #cv2.imshow('ids',frame)
        #cv2.imshow('ids',frame)
        return frame

    def create_multitracker(self,mframe):
        self.multiTracker = cv2.MultiTracker_create()
        for j, b in enumerate(self.last_detection_boxes):
            #print(b)
            
            rect = (b[0],b[1],b[2]-b[0],b[3]-b[1])
            #print("rect:",rect)
            self.multiTracker.add(cv2.TrackerKCF_create(), mframe, rect)

            
    def update_tracked_pigs(self,mframe):
        ok, boxes = self.multiTracker.update(mframe)
        for j,newbox in enumerate(boxes):
            left,top,right,bottom = int(newbox[0]), int(newbox[1]),int(newbox[0] + newbox[2]),int(newbox[1] + newbox[3])
            p1 = (left, top)
            p2 = (right, bottom)
            #cv2.rectangle(mframe, p1, p2, (200,0,0),3)
            rect = (left,top,right,bottom)
            self.tracked_boxes[j] = rect

    def add_disappeared_pigs_to_tracker(self,mframe):
        self.multiTracker = cv2.MultiTracker_create()
        for j, b in enumerate(self.tracked_boxes):
            #print(b)
            rect = (b[0],b[1],b[2]-b[0],b[3]-b[1])
            #print("rect:",rect)
            self.multiTracker.add(cv2.TrackerKCF_create(), mframe, rect)
            
        for j, b in enumerate(self.disappeared_list_boxes):
            #print(b)
            rect = (b[0],b[1],b[2]-b[0],b[3]-b[1])
            #print("rect:",rect)
            self.multiTracker.add(cv2.TrackerKCF_create(), mframe, rect)
            self.tracked_pigs.append(self.disappeared_list_pigs[j])
            self.tracked_boxes.append(b)
        self.update_tracked_pigs(mframe)
            
        
    def start(self,feeders,videoPath, debug = True,start_sec = None,end_sec = None, write_on_video = False, generate_excel_table =  True,frame_step = 4,scale_factor = 0.5):
        self.feeders = feeders
        self.frame_step = frame_step
        self.scale_factor = scale_factor
        first_frame = True
        lock = threading.Lock()
        lock.acquire()
        self.finished = False
        lock.release()
        if(not self.started):
            self.started =  True
            self.generate_excel_table = generate_excel_table
            self.write_on_video = write_on_video
            self.videoPath = videoPath
            self.debug = debug
            
            if(not self.debug):
                #video output
                self.videoFile = cv2.VideoCapture(videoPath)
                
                totalFrames = int(self.videoFile.get(cv2.CAP_PROP_FRAME_COUNT))
                self.total_frames = totalFrames
                self.FPS = self.videoFile.get(cv2.CAP_PROP_FPS)
                
                if(start_sec is not None):
                    self.start_frame = self.FPS*start_sec
                    self.end_frame = self.FPS*end_sec
                    if(self.start_frame<=totalFrames):
                        self.videoFile.set(cv2.CAP_PROP_POS_FRAMES,self.start_frame)
                    
                if(self.write_on_video):
                    frame_width = int(self.videoFile.get(3)*self.scale_factor)
                    frame_height = int(self.videoFile.get(4)*self.scale_factor)
                    #video_output = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
                    self.video_output = cv2.VideoWriter(videoPath[0:-4]+"_processed.mp4",cv2.VideoWriter_fourcc(*'XVID'), self.FPS/self.frame_step, (frame_width,frame_height))

                
            while(self.debug or self.videoFile.isOpened()):
                if(self.paused):
                    pass
                # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
                # i.e. a single-column array, where each item in the column has the pixel RGB value
                #ret, frame = video.read()
                
                stime = time.time()
                if(self.debug):
                    #Acquire frame from screen
                    size  = pyautogui.size()
                    left,top,x2,y2, = int((size[0]/2)-(CAP_WIDTH/2)),int(size[1]/2-CAP_HEIGHT/2),int(size[0]/2+CAP_WIDTH/2),int(size[1]/2+CAP_HEIGHT/2)
                    screen = grab_screen(region=(left,top,x2,y2))      
                    frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                else:
                    self.processed_frames_counter = self.videoFile.get(cv2.CAP_PROP_POS_FRAMES) - self.start_frame
                    #print("processed:",self.processed_frames_counter)
                    
                    self.total_frames_to_process = self.end_frame - self.start_frame
                    #print("total to process:", self.total_frames_to_process)
                    self.processing_progress = int(100*(self.processed_frames_counter/self.total_frames_to_process))
                    #print("progress:",self.processing_progress)
                    ret,frame = self.videoFile.read()

                if(scale_factor !=1):
                    frame = cv2.resize(frame, (0,0), fx=self.scale_factor, fy=self.scale_factor)
                #end of grabscreen
                mframe = frame.copy()
                
                frame_expanded = np.expand_dims(frame, axis=0)
                
                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = self.sess.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: frame_expanded})
                #print(time.time() - stime)
                

                # Draw the results of the detection (aka 'visulaize the results')
                '''vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=0.85)'''


                piglets = []
                for j, b in enumerate(boxes[0]):
                    if scores[0][j] >= 0.8:
                        height = np.size(frame, 0)
                        width = np.size(frame, 1)
                
                        left, right, top, bottom = int(boxes[0][j][1]*width), int(boxes[0][j][3]*width), int(boxes[0][j][0]*height), int(boxes[0][j][2]*height)
                        
                        track = [left,bottom,right,top,scores[0][j]]
                        piglets.append(track)


            

                if(len(piglets)<=1):
                    continue
                #print(np.array(piglets))       
                self.tracks = self.mot_tracker.update(np.array(piglets))
                #print(self.tracks)

                if(first_frame):
                    self._randomly_enumerate_pigs(mframe,boxes,scores)
                    #self.create_multitracker(mframe)
                    #print("first frame")
                    #print(self.current_detection_pigs)
                    first_frame = False
                else:
                    # update tracked pigs, if any.
                    #if(len(self.tracked_pigs)>0):
                    #    self.update_tracked_pigs(mframe)
                        #print("update_tracked_pigs")
                        
                    #gives a unique ID to every pig
                    self.enumerate_pigs(mframe,boxes,scores)
                    
                    
                        #print("not None")
                #detect overlapping pigs and feeders
                mframe = self.detect_eating_pig_(mframe,feeders,draw_feeder = DRAW_FEEDER)
                #if(not self.debug):
                 #   self._update_eating_time_counters_list(eating_list,FPS)

                mframe = self.draw_rects_and_pigs_ids(mframe,draw_rects= DRAW_RECTS,draw_pigs = DRAW_PIG_IDS)
                self.last_detection_boxes = self.current_detection_boxes
                self.last_detection_pigs = self.current_detection_pigs
                self.current_detection_boxes = []
                self.current_detection_pigs = []
                
                if(not self.debug and write_on_video):
                    self.video_output.write(mframe)
                

                cv2.imshow('Pig detector', mframe)
                
                # All the results have been drawn on the frame, so it's time to display it.
                #cv2.imshow('Pig detector', frame)

                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    break

                if((not debug) and (end_sec is not None) and (self.videoFile.get(cv2.CAP_PROP_POS_FRAMES)>= self.FPS*end_sec)):
                    #print("VIDEO FILE RELEASED!")
                    self.videoFile.release()
                self.videoFile.set(cv2.CAP_PROP_POS_FRAMES,self.videoFile.get(cv2.CAP_PROP_POS_FRAMES)+self.frame_step-1)
                #print(time.time() - stime)
            
            #print(eating_time_counters_list)
            if(self.generate_excel_table):
                self.generate_excel_report()

            self.finished = True   
            
            # Clean up
            #video.release()
            cv2.destroyAllWindows()
            if(not debug and write_on_video):
                self.video_output.release()

            
    def pause(self):
        self.paused = True
        
    def stop(self):
        
        #print(eating_time_counters_list)
        if(self.generate_excel_table):
            self.generate_excel_report()
        # Clean up
        #video.release()
        
        if(self.write_on_video):
            self.video_output.release()
        if(not self.debug):
            self.videoFile.release()
        cv2.destroyAllWindows()
        lock = threading.Lock()
        lock.acquire()
        self.processing_progress = 0
        self.processed_frames_counter = 0
        lock.release()

    

if __name__ == "__main__":
    
    detector = PigDetector()
    feeders = []
    videoPath = "1_4.avi"#input("Digite o nome do arquivo de video:\n")
    
    n_feeders =2# int(input("Digite o número de comedouros:\n"))
    for i in range(n_feeders):
        print("Desenhe o comedouro {}".format(i+1))
        feeders.append(feeder_delimiter.get_feeder_points(videoPath,i+1))
    start_time = 257#int(input("Tempo inicial:\n"))
    end_time = 260#int(input("Tempo final:\n"))
    write_on_video = True
    #print("Pontos do comedouro: %s\n" % feeders)
    videoProcessingThread = threading.Thread(target=detector.start,
                                                 args=(feeders, videoPath, False,start_time,end_time,write_on_video))
        
    videoProcessingThread.start()
    #import time
    #time.sleep(10)
    #detector.stop()
    #detector.start(feeders, videoPath, False,start_time,end_time,write_on_video)
    



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

#Grabscreen size
CAP_WIDTH = 1280
CAP_HEIGHT = 720

#feeder color
FEEDER_COLOR = (66, 244, 98)


## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)




def detect_eating_pig_(frame,boxes,scores,feeders,draw_bbox=True):
    eating_list = []
    
    for i, f in enumerate(feeders):
        eating_list.append(False)
        frame = draw_feeder_area(frame,feeders[i],FEEDER_COLOR,i+1)
        boxes_image = np.zeros(frame.shape, np.uint8)
        boxes_image = cv2.cvtColor(boxes_image, cv2.COLOR_BGR2GRAY)
        for j, b in enumerate(boxes[0]):
            if scores[0][j] >= 0.5:
                height = np.size(frame, 0)
                width = np.size(frame, 1)
                left, right, top, bottom = int(boxes[0][j][1]*width), int(boxes[0][j][3]*width), int(boxes[0][j][0]*height), int(boxes[0][j][2]*height)
                pig_rect = [(left, top), (left, bottom), (right, bottom),(right, top)]
                intersect_value = _calculate_intersection(frame,feeders[i],pig_rect)
                if(draw_bbox):
                    cv2.rectangle(frame,(left, top),(right, bottom),(0,255,0),2)
                
                if(intersect_value>3000):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame,"Comendo!",(left,top), font, 1,(0,255,0),2,cv2.LINE_AA)
                    eating_list[i] = True
                #boxes_image = cv2.bitwise_or(boxes_image,box_image)
                    
    return frame, eating_list

def _calculate_intersection(frame,feeder_poly,pig_rect):
    feeder_image = np.zeros(frame.shape, np.uint8)
    cv2.fillPoly(feeder_image, np.array([feeder_poly]), (255,255,255))
    feeder_image = cv2.cvtColor(feeder_image, cv2.COLOR_BGR2GRAY)
    
    box_image = np.zeros(frame.shape, np.uint8)
    cv2.fillPoly(box_image, np.array([pig_rect]), (255,255,255))
    box_image = cv2.cvtColor(box_image, cv2.COLOR_BGR2GRAY)
    
    and_image = cv2.bitwise_and(feeder_image,box_image)
    return cv2.countNonZero(and_image)
    
    
def draw_feeder_area(frame,feeder,color,n_feeder= None):
    ALPHA = 0.2
    canvasBlack = np.ones(frame.shape, np.uint8)
    # of a filled polygon
    cv2.fillPoly(canvasBlack, np.array([feeder]), color)
    frame = cv2.addWeighted(canvasBlack,ALPHA,frame, 1-ALPHA,10)
    if(n_feeder is not None):
        cX,cY = find_center_of_polygon(canvasBlack)
        cv2.putText(frame,str(n_feeder),(cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


def _update_eating_time_counters_list(eating_list,eating_time_counters_list,FPS):
    for i in range(len(eating_list)):
        if(eating_list[i]):
            eating_time_counters_list[i]+=1/FPS
            
def generate_excel_report(eating_time_counters_list):
    workbook = excel.Workbook('Relatorio_experimento_suinos.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0,0,'Comedouro #')
    worksheet.write(0,1,'Tempo estimado de alimentação (s)')
    for i in range(len(eating_time_counters_list)):
        worksheet.write(i+1,0,i+1)
        worksheet.write(i+1,1,round(eating_time_counters_list[i],2))
    workbook.close()

def find_center_of_polygon(image):
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

def start(feeders,videoPath, debug = True,start_sec = None,end_sec = None, write_on_video = False, generate_excel_table =  True):
        # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)


    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    eating_time_counters_list = [0 for i in range (len(feeders))]
    if(not debug):
        #video output
        videoFile = cv2.VideoCapture(videoPath)
        
        totalFrames = int(videoFile.get(cv2.CAP_PROP_FRAME_COUNT))
        FPS = videoFile.get(cv2.CAP_PROP_FPS)
        
        if(start_sec is not None):
            start_frame = FPS*start_sec
            if(start_frame<=totalFrames):
                videoFile.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
            
        if(write_on_video):
            frame_width = int(videoFile.get(3))
            frame_height = int(videoFile.get(4))
            #video_output = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
            video_output = cv2.VideoWriter('outpy2.mp4',cv2.VideoWriter_fourcc(*'XVID'), FPS, (frame_width,frame_height))

        
    while(debug or videoFile.isOpened()):
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        #ret, frame = video.read()
        
        
        if(debug):
            #Acquire frame from screen
            size  = pyautogui.size()
            left,top,x2,y2, = int((size[0]/2)-(CAP_WIDTH/2)),int(size[1]/2-CAP_HEIGHT/2),int(size[0]/2+CAP_WIDTH/2),int(size[1]/2+CAP_HEIGHT/2)
            screen = grab_screen(region=(left,top,x2,y2))      
            frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        else:
            ret,frame = videoFile.read()
        #end of grabscreen
        mframe = frame.copy()
        
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

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

        mframe, eating_list = detect_eating_pig_(mframe,boxes,scores,feeders)
        if(not debug):
            _update_eating_time_counters_list(eating_list,eating_time_counters_list,FPS)
        
        if(not debug and write_on_video):
            video_output.write(mframe)
        

        cv2.imshow('Pig detector', mframe)
        
        # All the results have been drawn on the frame, so it's time to display it.
        #cv2.imshow('Pig detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        if((not debug) and (end_sec is not None) and (videoFile.get(cv2.CAP_PROP_POS_FRAMES)>= FPS*end_sec)):
            #print("VIDEO FILE RELEASED!")
            videoFile.release()
            
    #print(eating_time_counters_list)
    if(generate_excel_table):
        generate_excel_report(eating_time_counters_list)
    
    # Clean up
    #video.release()
    cv2.destroyAllWindows()
    if(not debug and write_on_video):
        video_output.release()



    

if __name__ == "__main__":
    feeders = []
    videoPath = input("Digite o nome do arquivo de video:\n")
    n_feeders = int(input("Digite o número de comedouros:\n"))
    for i in range(n_feeders):
        print("Desenhe o comedouro {}".format(i+1))
        feeders.append(feeder_delimiter.get_feeder_points(videoPath,i+1))
    start_time = 270#int(input("Tempo inicial:\n"))
    end_time = 285#int(input("Tempo final:\n"))
    write_on_video = True
    #print("Pontos do comedouro: %s\n" % feeders)
    start(feeders, videoPath, False,start_time,end_time,write_on_video)
    



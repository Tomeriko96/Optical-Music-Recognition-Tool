#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pandas as pd

import random
import math
import re
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import glob
import matplotlib.image as mpimg
matplotlib.use('TkAgg')

import six.moves.urllib as urllib
import tarfile
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
# Root directory of the project
ROOT_DIR = os.path.abspath('')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from datasets.balloon import balloon

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
BALLON_WEIGHTS_PATH = "mask_rcnn_balloon.h5"  # TODO: update this path
config = balloon.BalloonConfig()
MENSURAL_DIR = os.path.join(ROOT_DIR, "datasets/balloon")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode = "inference", model_dir = MODEL_DIR, config = config)
# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
# weights_path = "/path/to/mask_rcnn_balloon.h5"

# Or, load the last model you trained
weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name = True)
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# In order to use music21 a version of MuseScore is required. The environment need to be set as shown below. 

from music21 import *
us = environment.UserSettings()
us['musescoreDirectPNGPath'] = '/Applications/MuseScore 3.app/Contents/MacOS/mscore'
us['pdfPath'] = '/Applications/MuseScore 3.app/Contents/MacOS/mscore'
us['graphicsPath'] = '/Applications/MuseScore 3.app/Contents/MacOS/mscore'
us['musicxmlPath'] = '/Applications/MuseScore 3.app/Contents/MacOS/mscore'

# Function to clear the folders containing csv and png files. 
def clean_up():
    import os, shutil
    folders = ['csv2/', 'csv/', 'detect/']
    for folder in folders:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
# Class to display key signatures in tinynotation.
class KeyToken(tinyNotation.Token):
    def parse(self, parent):
        keyName = self.token
        return key.Key(keyName)
    
def find_items_within(l1, l2, dist):
    l1.sort()
    l2.sort()
    b = 0
    e = 0
    ans = []
    for a in l1:
        while b < len(l2) and a - l2[b] > dist:
            b += 1
        while e < len(l2) and l2[e] - a <= dist:
            e += 1
        ans.extend([(a,x) for x in l2[b:e]])
    return ans

def grouper(iterable, img):
    img = cv2.imread(img)
    img_height, img_width, dim = img.shape
    prev = None
    group = []
    for item in iterable:
        if not prev or item - prev <= 0.019*img_height:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group
        
def Average(lst): 
    try: 
        return sum(lst) / len(lst)
    except:
        print("average error")
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize = (size * cols, size * rows))
    return ax

def eroded(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    flag,b = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    cv2.imwrite("1tresh.jpg", b)

    element = np.ones((3,3))
    b = cv2.erode(b,element)
    cv2.imwrite("2erodedtresh.jpg", b)

    edges = cv2.Canny(b,10,100,apertureSize = 3)
    cv2.imwrite("3Canny.jpg", edges)
    return b

def one(IMAGE_NAME):
    eroded(IMAGE_NAME)
    image = '2erodedtresh.jpg'
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.bitwise_not(img)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    horizontal = th2
    vertical = th2
    rows, cols = horizontal.shape
    horizontalsize = int(cols / 50)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    cv2.imwrite("horizontal.jpg", horizontal)
    src = cv2.imread(image, cv2.IMREAD_COLOR)

    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    gray = cv2.bitwise_not(gray)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    cv2.imwrite('edges-50-150.jpg',edges)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    cols = horizontal.shape[1]
    horizontal_size = int(cols / 30)

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    cv2.imwrite("img_horizontal8.png", horizontal)

    h_transpose = np.transpose(np.nonzero(horizontal))

    rows = vertical.shape[0]
    verticalsize = int(rows / 30)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    cv2.imwrite("img_vertical8.png", vertical)

    v_transpose = np.transpose(np.nonzero(vertical))
    img = src.copy()
    img_height, img_width, dim = img.shape
    minLineLength = 50
    maxLineGap = 50
    lines = cv2.HoughLinesP(horizontal, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    values = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            values.append(y1)
            cv2.line(img, (x1, y1),(x2, y2),(0, 255, 0), 10)

    cv2.imwrite('houghlinesP.jpg', img)
    return values


def first_detection(img, img_width, img_height):
    # Number of classes the object detector can identify
    NUM_CLASSES = 33
    confidence = 0.50


    # Open the hitlist file.
    hitf = open("csv2/hitlist-%d.csv" %(i+1), 'w')
    hitf.write('image,class,score,bb0,bb1,bb2,bb3,image_width,image_height\n')
    hitlim = confidence
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes = NUM_CLASSES, use_display_name = True)
    category_index = label_map_util.create_category_index(categories)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name = '')

        sess = tf.Session(graph = detection_graph)

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

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(img)
    image_expanded = np.expand_dims(image, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict = {image_tensor: image_expanded})
    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates = True,
        line_thickness = 4,
        min_score_thresh = confidence
    )
    
    nprehit = scores.shape[1] # 2nd array dimension
    for j in range(nprehit):
        fname = "image" + str(j)
        classid = int(classes[0][j])
        classname = category_index[classid]["name"]
        score = scores[0][j]
        if (score >= hitlim):
            sscore = str(score)
            bbox = boxes[0][j]
            b0 = str(bbox[0])
            b1 = str(bbox[1])
            b2 = str(bbox[2])
            b3 = str(bbox[3])
            img_width = str(img_width)
            img_height = str(img_height)
            line = ",".join([fname, classname, sscore, b0, b1, b2, b3, img_width, img_height])
            hitf.write(line + "\n")
    # close hitlist
    hitf.flush()
    hitf.close()
    return image

def results(img):
    image = img
    image = cv2.imread(image)
    results = model.detect([image], verbose=1)
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            ['BG', 'note'], r['scores'], ax=ax,
                            title="Predictions")
    return r
def mrcnn(img):
    image = img
    image = cv2.imread(image)
    results = model.detect([image], verbose=1)
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            ['BG', 'note'], r['scores'], ax=ax,
                            title="Predictions")
    return image

def notes_try(row):
    for stave in lines:
        for i in range(len(stave)-1):
            if abs(stave[i]-row['y_avg']) < abs(stave[i+1] - row['y_avg']):
                return stave.index(stave[i])

def comparos(df):
    df = df.loc[df['object'].isin(['semifusa','topsemifusa','fusa','topfusa','semiminima',\
                                   'topsemiminima','minima','topminima','brevis', 'semibreve'])]
    return df['x'].values

def pnds(hitlist, notes, img_height, img_width):
    global lines
    global dff
    hitlist.sort_values(by=['bb1'], inplace=True)
    hitlist['x_avg'] = hitlist[['bb1', 'bb3']].mean(axis=1)
    hitlist['y_avg'] = hitlist[['bb0', 'bb2']].mean(axis=1)
    
    hitlist['bb0'] = hitlist['bb0'] * int(img_height)
    hitlist['bb2'] = hitlist['bb2'] * int(img_height)
    hitlist['x'] = hitlist['x_avg'] * int(img_width)
    hitlist['y'] = hitlist['y_avg'] * int(img_height)

    
    hitlist['prev_class'] = hitlist['class'].shift()
    hitlist['object'] = hitlist['class']
    hitlist['next_class'] = hitlist['class'].shift(-1)
    
    hitlist = hitlist.drop(['image', 'class', 'score', 'bb1', 'bb3', 'x_avg', 'y_avg'], axis=1)
    hitlist['dotted'] = np.where(hitlist['next_class'] == 'dot', 'yes', 'no')
    hitlist['bridged'] = np.where(hitlist['next_class'] == 'bridge', 'yes', 'no')
    hitlist['sharped'] = np.where(hitlist['prev_class'] == 'sharp', 'yes', 'no')
    hitlist['flatted'] = np.where(hitlist['prev_class'] == 'flat', 'yes', 'no')
    hitlist['notes'] = np.select([hitlist.object == 'semifusa', hitlist.object == 'topsemifusa',
                         hitlist.object == 'fusa', hitlist.object == 'topfusa',
                        hitlist.object == 'semiminima', hitlist.object == 'topsemiminima',
                        hitlist.object == 'minima', hitlist.object == 'topminima',
                        hitlist.object=='brevis', hitlist.object=='semibreve', 
                        hitlist.object=='wholerest', hitlist.object=='2rest', hitlist.object=='4rest',
                                  hitlist.object=='uprest', hitlist.object=='Rrest',
                                  hitlist.object=='downrest', hitlist.object =='rurest'],
                        [0.25, 0.25, 0.5, 0.5, 1, 1, 2, 2, 8, 4, 8, 16, 32, 4, 1, 2, 1], 
                        default=None)
    hitlist['prev_x'] = hitlist['x'].shift(1)
    hitlist['prev_x'].fillna(0)
    hitlist['current_x'] = hitlist['x']

    for index, row in hitlist.iterrows():
        if row['current_x'] - row['prev_x'] < 1:
            if row['object'] == row['prev_class']:

                hitlist = hitlist.drop(index)

    hitlist = hitlist.reset_index(drop=True)
    notes.sort_values(by=['one'], inplace = True)
    notes['y_avg'] = notes[['zero', 'two']].mean(axis = 1)
    notes['y_diff'] = notes['two'] - notes['zero']
    average_height = notes['y_avg'].values
    notes['x_avg'] = notes[['one', 'three']].mean(axis=1)
    a = notes['x_avg'].values
    b = comparos(hitlist)
    notes['line'] = notes.apply(notes_try, axis = 1)
    compared = find_items_within(a,b,20)
    new = pd.DataFrame(compared, columns = ['x_avg', 'one'])
    new.sort_values(by=['x_avg'], inplace = True)
    df = pd.merge(new, notes, on=['x_avg'])
    df = df.rename(columns={'one_x': 'x'})
    dff = pd.merge(df, hitlist, on=['x'], how='outer')
    df2 = dff.loc[dff['object'] == 'brevis']
    df2['y'] = df2['y'].astype(float)
    df2['y_avg'] = df2['y']
    dff['y_avg'].loc[df2.index.values] = df2['y_avg'].loc[df2.index.values]
    dff['line'] = dff.apply(notes_try, axis = 1)
    dff = dff.sort_values(by=['x'])
    return dff

        
def liner(IMAGE_NAME, number):
    global lines
    global i
    clean_up()
    i = number
    image = IMAGE_NAME
    img = cv2.imread(image)
    img_height, img_width, dim = img.shape
    src = cv2.imread(image, cv2.IMREAD_COLOR)
    l = sorted(one(IMAGE_NAME))
    G = dict(enumerate(grouper(l, IMAGE_NAME), 1))
    avgDict = {}
    for k,v in G.items():
        avgDict[k] = sum(v)/ float(len(v))
    num = avgDict.values()
    line_height = []
    for numb in num:
        line_height.append(numb)
    staves = [list(t) for t in zip(*[iter(line_height)]*5)]
    difference = [j-i for i, j in zip(line_height[:-1], line_height[1:])]
    inds=[0]+[ind for ind,(i,j) in enumerate(zip(difference,difference[1:]),1) if j-i>difference[0]*2]+[len(difference)+1]
    [difference[i:j] for i,j in zip(inds,inds[1:])]
    difference=[]
    delta = []
    for stave in staves:
        difference.append([j-i for i, j in zip(stave[:-1], stave[1:])])
        delta.append(np.mean(difference))
    delta = Average(delta)
    lines = []
    for stave in staves:
        line0 = stave[0] - delta * 1.5
        line1 = stave[0] - delta
        line2 = stave[0] - delta / 2
        line3 = stave[0]
        line4 = stave[0] + delta / 2
        line5 = stave[0] + delta
        line6 = stave[0] + delta * 1.5
        line7 = stave[0] + delta * 2
        line8 = stave[0] + delta * 2.5
        line9 = stave[0] + delta * 3
        line10 = stave[0] + delta * 3.5
        line11 = stave[0] + delta * 4
        line12 = stave[0] + delta * 4.5
        line13 = stave[0] + delta * 5
        line14 = stave[0] + delta * 5.5
        if line0<0:
            line0=0
        else:
            line0=line0
        if line14>img_height:
            line14=img_height
        else:
            line14=line14
        lines.append( [line0, line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14])
    img = src.copy()
    lineThickness = 2
    for line in lines:
        for line in line:
            cv2.line(img, (int(0), int(line)), (int(img_width), int(line)), (0,255,0), lineThickness)

    sys.path.append("..")
    from object_detection.utils import ops as utils_ops
    resss = results(IMAGE_NAME)
    cv2.imwrite("detect/detect-%.3d.jpg" % (i+1), first_detection(image, img_width, img_height))
    np.savetxt("csv/%d.csv" %(i+1), resss['rois'], delimiter=",")
    column_names = ['zero', 'one', 'two', 'three']
    try:
        notes = pd.read_csv('csv/%d.csv'%(i+1)) 
        notes.columns = column_names
    except:
        print('csv_error')
    
    hitlist = pd.read_csv("csv2/hitlist-%d.csv" %(i+1))

    dff = pnds(hitlist, notes, img_height, img_width)

    return dff


def merge(lst):
    df = pd.concat(lst,ignore_index=True)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_row', 500)
    return df



def to_tenor(row):
    if row['notes'] != None:
        if row['line'] == 0:
            return 'A4'
        elif row['line'] == 1:
            return 'G4'
        elif row['line'] == 2:
            return 'F4'
        elif row['line'] == 3:
            return 'E4'
        elif row['line'] == 4:
            return 'D4'
        elif row['line'] == 5:
            return 'C4'
        elif row['line'] == 6:
            return 'B3'
        elif row['line'] == 7:
            return 'A3'
        elif row['line'] == 8:
            return 'G3'
        elif row['line'] == 9:
            return 'F3'
        elif row['line'] == 10:
            return 'E3'
        elif row['line'] == 11:
            return 'D3'
        elif row['line'] == 12:
            return 'C3'
        elif row['line'] == 13:
            return 'B2'
        elif row['line'] == 14:
            return 'A2'
#         elif row['object'] == 'wholerest' or row['object'] == '2rest' or row['object'] == '4rest' or row['object'] == 'uprest' or row['object'] == 'Rrest' or row['object'] == 'downrest' or row['object'] == 'rurest':
#             return 'r'
        else: 
            return ''
    else:
        return ''

def to_alto(row):
    if row['notes'] != None:
        if row['line'] == 0:
            return 'C5'
        elif row['line'] == 1:
            return 'B4'
        elif row['line'] == 2:
            return 'A4'
        elif row['line'] == 3:
            return 'G4'
        elif row['line'] == 4:
            return 'F4'
        elif row['line'] == 5:
            return 'E4'
        elif row['line'] == 6:
            return 'D4'
        elif row['line'] == 7:
            return 'C4'
        elif row['line'] == 8:
            return 'B3'
        elif row['line'] == 9:
            return 'A3'
        elif row['line'] == 10:
            return 'G3'
        elif row['line'] == 11:
            return 'F3'
        elif row['line'] == 12:
            return 'E3'
        elif row['line'] == 13:
            return 'D3'
        elif row['line'] == 14:
            return 'C3'
#         elif row['object'] == 'wholerest' or row['object'] == '2rest' or row['object'] == '4rest' or row['object'] == 'uprest' or row['object'] == 'Rrest' or row['object'] == 'downrest' or row['object'] == 'rurest':
#             return 'r'
        else: 
            return ''
    else:
        return ''
def to_bass(row):
    if row['notes'] != None:
        if row['line'] == 0:
            return 'D4'
        elif row['line'] == 1:
            return 'C4'
        elif row['line'] == 2:
            return 'B3'
        elif row['line'] == 3:
            return 'A3'
        elif row['line'] == 4:
            return 'G3'
        elif row['line'] == 5:
            return 'F3'
        elif row['line'] == 6:
            return 'E3'
        elif row['line'] == 7:
            return 'D3'
        elif row['line'] == 8:
            return 'C3'
        elif row['line'] == 9:
            return 'B2'
        elif row['line'] == 10:
            return 'A2'
        elif row['line'] == 11:
            return 'G2'
        elif row['line'] == 12:
            return 'F2'
        elif row['line'] == 13:
            return 'E2'
        elif row['line'] == 14:
            return 'D2'
#         elif row['object'] == 'wholerest' or row['object'] == '2rest' or row['object'] == '4rest' or row['object'] == 'uprest' or row['object'] == 'Rrest' or row['object'] == 'downrest' or row['object'] == 'rurest':
#             return 'r'
        else: 
            return ''
    else:
        return ''
def to_treble(row):
    if row['notes'] != None:
        if row['line'] == 0:
            return 'B5'
        elif row['line'] == 1:
            return 'A5'
        elif row['line'] == 2:
            return 'G5'
        elif row['line'] == 3:
            return 'F5'
        elif row['line'] == 4:
            return 'E5'
        elif row['line'] == 5:
            return 'D5'
        elif row['line'] == 6:
            return 'C5'
        elif row['line'] == 7:
            return 'B4'
        elif row['line'] == 8:
            return 'A4'
        elif row['line'] == 9:
            return 'G4'
        elif row['line'] == 10:
            return 'F4'
        elif row['line'] == 11:
            return 'E4'
        elif row['line'] == 12:
            return 'D4'
        elif row['line'] == 13:
            return 'C4'
        elif row['line'] == 14:
            return 'B3'
#         elif row['object'] == 'wholerest' or row['object'] == '2rest' or row['object'] == '4rest' or row['object'] == 'uprest' or row['object'] == 'Rrest' or row['object'] == 'downrest' or row['object'] == 'rurest':
#             return 'r'
        else: 
            return ''
    else:
        return ''
def accidentals(row):
    if row['final'] != '':
        if row['sharped'] == 'yes' and row['x'] > 200:
            return row['final']+'#'
        elif row['flatted'] == 'yes' and row['x'] > 200:
            return row['final']+'-'
        else:
            return row['final']
    else:
        return ''
    
def tenor(df):
    df['final'] = df.apply(to_tenor, axis = 1)
    df['final'] = df.apply(accidentals, axis = 1)
    string = df['final'].values
    print(string)
    return string
    
def alto(df):
    df['final'] = df.apply(to_alto, axis = 1)
    df['final'] = df.apply(accidentals, axis = 1)
    string = df['final'].values
    print(string)
    return string

def bass(df):
    df['final'] = df.apply(to_bass, axis = 1)
    df['final'] = df.apply(accidentals, axis = 1)
    string = df['final'].values
    print(string)
    return string

def treble(df):
    df['final'] = df.apply(to_treble, axis = 1)
    df['final'] = df.apply(accidentals, axis = 1)
    string = df['final'].values
    print(string)
    return string

def dotted(i):
    if dff['dotted'][i] == 'yes':
        return True
    else: return False
def length(i):
    d= dff['notes'].fillna(0).astype(float)
    
    if dotted(i):
        d[i] = d[i]*1.5
    else: d[i] = d[i]

    return d[i]

def keyed(row):
    sharps = 0
    flats = 0
    index = dff['notes'].index[dff['notes'].notnull()]
    df_index = dff.index.values.tolist()
    c=[df_index.index(i) for i in index]
    g = c[0]

    for x in dff.object[0:g]:
        if x=='sharp':
            sharps+=1
        elif x=='flat':
            flats+=1
        else:
            flats = flats
            sharps = sharps
    if sharps > flats:
        return sharps
    elif flats > sharps:
        return -flats
    else:
        return 0
def key_element(x):
    global k
    if k == 1:
        if x == 'F3':
            x = 'F#3'
        elif x == 'F2':
            x = 'F#2'
        elif x == 'F4':
            x = 'F#4'
        elif x == 'F5':
            x = 'F#5'
        elif x == 'F2-':
            x = 'F2'
        elif x == 'F3-':
            x = 'F3'
        elif x == 'F4-':
            x = 'F4'
        elif x == 'F5-':
            x = 'F5'
        else:
            x=x
    elif k == 2:
        if x == 'F3':
            x = 'F#3'
        elif x == 'F2':
            x = 'F#2'
        elif x == 'F4':
            x = 'F#4'
        elif x == 'F5':
            x = 'F#5'
        elif x == 'C3':
            x = 'C#3'
        elif x == 'C2':
            x = 'C#2'
        elif x == 'F4':
            x = 'C#4'
        elif x == 'F5':
            x = 'C#5'
        else:
            x=x
    elif k == 3:
        if x == 'F3':
            x = 'F#3'
        elif x == 'F2':
            x = 'F#2'
        elif x == 'F4':
            x = 'F#4'
        elif x == 'F5':
            x = 'F#5'
        elif x == 'C3':
            x = 'C#3'
        elif x == 'C2':
            x = 'C#2'
        elif x == 'C4':
            x = 'C#4'
        elif x == 'C5':
            x = 'C#5'
        elif x == 'G3':
            x = 'G#3'
        elif x == 'G2':
            x = 'G#2'
        elif x == 'G4':
            x = 'G#4'
        elif x == 'G5':
            x = 'G#5'
        
        else:
            x=x
    elif k == 4:
        if x == 'F3':
            x = 'F#3'
        elif x == 'F2':
            x = 'F#2'
        elif x == 'F4':
            x = 'F#4'
        elif x == 'F5':
            x = 'F#5'
        elif x == 'C3':
            x = 'C#3'
        elif x == 'C2':
            x = 'C#2'
        elif x == 'C4':
            x = 'C#4'
        elif x == 'C5':
            x = 'C#5'
        elif x == 'G3':
            x = 'G#3'
        elif x == 'G2':
            x = 'G#2'
        elif x == 'G4':
            x = 'G#4'
        elif x == 'G5':
            x = 'G#5'
        elif x == 'D3':
            x = 'D#3'
        elif x == 'D2':
            x = 'D#2'
        elif x == 'D4':
            x = 'D#4'
        elif x == 'D5':
            x = 'D#5'

        else:
            x=x
    elif k == 5:
        if x == 'F3':
            x = 'F#3'
        elif x == 'F2':
            x = 'F#2'
        elif x == 'F4':
            x = 'F#4'
        elif x == 'F5':
            x = 'F#5'
        elif x == 'C3':
            x = 'C#3'
        elif x == 'C2':
            x = 'C#2'
        elif x == 'C4':
            x = 'C#4'
        elif x == 'C5':
            x = 'C#5'
        elif x == 'G3':
            x = 'G#3'
        elif x == 'G2':
            x = 'G#2'
        elif x == 'G4':
            x = 'G#4'
        elif x == 'G5':
            x = 'G#5'
        elif x == 'D3':
            x = 'D#3'
        elif x == 'D2':
            x = 'D#2'
        elif x == 'D4':
            x = 'D#4'
        elif x == 'D5':
            x = 'D#5'
        elif x == 'A3':
            x = 'A#3'
        elif x == 'A2':
            x = 'A#2'
        elif x == 'A4':
            x = 'A#4'
        elif x == 'A5':
            x = 'A#5'

        else:
            x=x
    elif k == 6:
        if x == 'F3':
            x = 'F#3'
        elif x == 'F2':
            x = 'F#2'
        elif x == 'F4':
            x = 'F#4'
        elif x == 'F5':
            x = 'F#5'
        elif x == 'C3':
            x = 'C#3'
        elif x == 'C2':
            x = 'C#2'
        elif x == 'C4':
            x = 'C#4'
        elif x == 'C5':
            x = 'C#5'
        elif x == 'G3':
            x = 'G#3'
        elif x == 'G2':
            x = 'G#2'
        elif x == 'G4':
            x = 'G#4'
        elif x == 'G5':
            x = 'G#5'
        elif x == 'D3':
            x = 'D#3'
        elif x == 'D2':
            x = 'D#2'
        elif x == 'D4':
            x = 'D#4'
        elif x == 'D5':
            x = 'D#5'
        elif x == 'A3':
            x = 'A#3'
        elif x == 'A2':
            x = 'A#2'
        elif x == 'A4':
            x = 'A#4'
        elif x == 'A5':
            x = 'A#5'
        elif x == 'E3':
            x = 'E#3'
        elif x == 'E2':
            x = 'E#2'
        elif x == 'E4':
            x = 'E#4'
        elif x == 'E5':
            x = 'E#5'
        else:
            x=x
    elif k == 7:
        if x == 'F3':
            x = 'F#3'
        elif x == 'F2':
            x = 'F#2'
        elif x == 'F4':
            x = 'F#4'
        elif x == 'F5':
            x = 'F#5'
        elif x == 'C3':
            x = 'C#3'
        elif x == 'C2':
            x = 'C#2'
        elif x == 'C4':
            x = 'C#4'
        elif x == 'C5':
            x = 'C#5'
        elif x == 'G3':
            x = 'G#3'
        elif x == 'G2':
            x = 'G#2'
        elif x == 'G4':
            x = 'G#4'
        elif x == 'G5':
            x = 'G#5'
        elif x == 'D3':
            x = 'D#3'
        elif x == 'D2':
            x = 'D#2'
        elif x == 'D4':
            x = 'D#4'
        elif x == 'D5':
            x = 'D#5'
        elif x == 'A3':
            x = 'A#3'
        elif x == 'A2':
            x = 'A#2'
        elif x == 'A4':
            x = 'A#4'
        elif x == 'A5':
            x = 'A#5'
        elif x == 'E3':
            x = 'E#3'
        elif x == 'E2':
            x = 'E#2'
        elif x == 'E4':
            x = 'E#4'
        elif x == 'E5':
            x = 'E#5'
        elif x == 'B3':
            x = 'B#3'
        elif x == 'B2':
            x = 'B#2'
        elif x == 'B4':
            x = 'B#4'
        elif x == 'B5':
            x = 'B#5'
        else:
            x=x
    elif k == -1:
        if x == 'B3':
            x = 'B-3'
        elif x == 'B2':
            x = 'B-2'
        elif x == 'B4':
            x = 'B-4'
        elif x == 'B5':
            x = 'B-5'
        else:
            x=x
    elif k == -2:
        if x == 'B3':
            x = 'B-3'
        elif x == 'B2':
            x = 'B-2'
        elif x == 'B4':
            x = 'B-4'
        elif x == 'B5':
            x = 'B-5'
        elif x == 'E3':
            x = 'E-3'
        elif x == 'E2':
            x = 'E-2'
        elif x == 'E4':
            x = 'E-4'
        elif x == 'E5':
            x = 'E-5'

        else:
            x=x
    elif k == -3:
        if x == 'B3':
            x = 'B-3'
        elif x == 'B2':
            x = 'B-2'
        elif x == 'B4':
            x = 'B-4'
        elif x == 'B5':
            x = 'B-5'
        elif x == 'E3':
            x = 'E-3'
        elif x == 'E2':
            x = 'E-2'
        elif x == 'E4':
            x = 'E-4'
        elif x == 'E5':
            x = 'E-5'
        elif x == 'A3':
            x = 'A-3'
        elif x == 'A2':
            x = 'A-2'
        elif x == 'A4':
            x = 'A-4'
        elif x == 'A5':
            x = 'A-5'

        else:
            x=x
    elif k == -4:
        if x == 'B3':
            x = 'B-3'
        elif x == 'B2':
            x = 'B-2'
        elif x == 'B4':
            x = 'B-4'
        elif x == 'B5':
            x = 'B-5'
        elif x == 'E3':
            x = 'E-3'
        elif x == 'E2':
            x = 'E-2'
        elif x == 'E4':
            x = 'E-4'
        elif x == 'E5':
            x = 'E-5'
        elif x == 'A3':
            x = 'A-3'
        elif x == 'A2':
            x = 'A-2'
        elif x == 'A4':
            x = 'A-4'
        elif x == 'A5':
            x = 'A-5'
        elif x == 'D3':
            x = 'D-3'
        elif x == 'D2':
            x = 'D-2'
        elif x == 'D4':
            x = 'D-4'
        elif x == 'D5':
            x = 'D-5'

        else:
            x=x
    elif k == -5:
        if x == 'B3':
            x = 'B-3'
        elif x == 'B2':
            x = 'B-2'
        elif x == 'B4':
            x = 'B-4'
        elif x == 'B5':
            x = 'B-5'
        elif x == 'E3':
            x = 'E-3'
        elif x == 'E2':
            x = 'E-2'
        elif x == 'E4':
            x = 'E-4'
        elif x == 'E5':
            x = 'E-5'
        elif x == 'A3':
            x = 'A-3'
        elif x == 'A2':
            x = 'A-2'
        elif x == 'A4':
            x = 'A-4'
        elif x == 'A5':
            x = 'A-5'
        elif x == 'D3':
            x = 'D-3'
        elif x == 'G3':
            x = 'G-3'
        elif x == 'G2':
            x = 'G-2'
        elif x == 'G4':
            x = 'G-4'
        elif x == 'G5':
            x = 'G-5'

        else:
            x=x
    elif k == -6:
        if x == 'B3':
            x = 'B-3'
        elif x == 'B2':
            x = 'B-2'
        elif x == 'B4':
            x = 'B-4'
        elif x == 'B5':
            x = 'B-5'
        elif x == 'E3':
            x = 'E-3'
        elif x == 'E2':
            x = 'E-2'
        elif x == 'E4':
            x = 'E-4'
        elif x == 'E5':
            x = 'E-5'
        elif x == 'A3':
            x = 'A-3'
        elif x == 'A2':
            x = 'A-2'
        elif x == 'A4':
            x = 'A-4'
        elif x == 'A5':
            x = 'A-5'
        elif x == 'D3':
            x = 'D-3'
        elif x == 'G3':
            x = 'G-3'
        elif x == 'G2':
            x = 'G-2'
        elif x == 'G4':
            x = 'G-4'
        elif x == 'G5':
            x = 'G-5'
        elif x == 'C3':
            x = 'C-3'
        elif x == 'C2':
            x = 'C-2'
        elif x == 'C4':
            x = 'C-4'
        elif x == 'C5':
            x = 'C-5'

        else:
            x=x
    elif k == -7:
        if x == 'B3':
            x = 'B-3'
        elif x == 'B2':
            x = 'B-2'
        elif x == 'B4':
            x = 'B-4'
        elif x == 'B5':
            x = 'B-5'
        elif x == 'E3':
            x = 'E-3'
        elif x == 'E2':
            x = 'E-2'
        elif x == 'E4':
            x = 'E-4'
        elif x == 'E5':
            x = 'E-5'
        elif x == 'A3':
            x = 'A-3'
        elif x == 'A2':
            x = 'A-2'
        elif x == 'A4':
            x = 'A-4'
        elif x == 'A5':
            x = 'A-5'
        elif x == 'D3':
            x = 'D-3'
        elif x == 'G3':
            x = 'G-3'
        elif x == 'G2':
            x = 'G-2'
        elif x == 'G4':
            x = 'G-4'
        elif x == 'G5':
            x = 'G-5'
        elif x == 'C3':
            x = 'C-3'
        elif x == 'C2':
            x = 'C-2'
        elif x == 'C4':
            x = 'C-4'
        elif x == 'C5':
            x = 'C-5'
        elif x == 'F3':
            x = 'F-3'
        elif x == 'F2':
            x = 'F-2'
        elif x == 'F4':
            x = 'F-4'
        elif x == 'F5':
            x = 'F-5'
        else:
            x=x
    return x
    
def tenored(lst, df):
    global dff
    dff = df
    tenor = clef.TenorClef()
    m = stream.Measure([tenor])
    m.append(meter.TimeSignature('4/4'))
    k = keyed(dff)
    m.append(key.KeySignature(k))
    for i,x in enumerate(lst):
        r = length(i)
        try:
            if x != 'r':
                x = key_element(x)
                lst[i] = key_element(x)
                m.append(note.Note(x, quarterLength = r))
            else:
                m.append(note.Rest(x, quarterLength = r))
        except: 
            pass
    m.show()
    m.show('midi')

    


def trebled(lst, df):
    global dff
    dff = df
    treble = clef.TrebleClef()
    m = stream.Measure([treble])
    m.append(meter.TimeSignature('4/4'))
    k = keyed(dff)
    m.append(key.KeySignature(k))
    for i,x in enumerate(lst):
        r = length(i)
        try:
            if x != 'r':
                x = key_element(x)
                lst[i] = key_element(x)
                m.append(note.Note(x, quarterLength = r))
            else:
                m.append(note.Rest(x, quarterLength = r))
        except: 
            pass
    m.show()
    m.show('midi')
def bassed(lst, df):
    global dff
    dff = df
    bass = clef.BassClef()
    m = stream.Measure([bass])
    m.append(meter.TimeSignature('4/4'))
    k = keyed(dff)
    m.append(key.KeySignature(k))
    for i,x in enumerate(lst):
        r = length(i)
        try:
            if x != 'r':
                x = key_element(x)
                lst[i] = key_element(x)
                m.append(note.Note(x, quarterLength = r))
            else:
                m.append(note.Rest(x, quarterLength = r))
        except: 
            pass
    m.show()
    m.show('midi')
    return lst
def altoed(lst, df):
    global k
    global dff
    dff = df
    alto = clef.AltoClef()
    m = stream.Measure([alto])
    m.append(meter.TimeSignature('4/4'))
    k = keyed(dff)
    m.append(key.KeySignature(k))
    for i,x in enumerate(lst):
        r = length(i)
        try:
            if x != 'r':
                x = key_element(x)
                lst[i] = key_element(x)
                m.append(note.Note(x, quarterLength = r))
            else:
                m.append(note.Rest(x, quarterLength = r))
        except: 
            pass
    m.show()
    m.show('midi')
def sequence(row):
    if row['new'] == 'c,':
        return 0
    elif row['new'] == 'c#,' or row['new'] == 'd-,':
        return 1
    elif row['new'] == 'd,':
        return 2
    elif row['new'] == 'd#,' or row['new'] == 'e-,':
        return 3
    elif row['new'] == 'e,':
        return 4
    elif row['new'] == 'f,':
        return 5
    elif row['new'] == 'f#,' or row['new'] == 'g-,':
        return 6
    elif row['new'] == 'g,':
        return 7
    elif row['new'] == 'g#,' or row['new'] == 'a-,':
        return 8
    elif row['new'] == 'a,':
        return 9
    elif row['new'] == 'a#,' or row['new'] == 'b-,':
        return 10
    elif row['new'] == 'b,':
        return 11
    elif row['new'] == 'b#,' or row['new'] == 'c-':
        return 12
    elif row['new'] == 'c':
        return 13
    elif row['new'] == 'c#' or row['new'] == 'd-':
        return 14
    elif row['new'] == 'd':
        return 15
    elif row['new'] == 'd#' or row['new'] == 'e-':
        return 16
    elif row['new'] == 'e':
        return 17
    elif row['new'] == 'f':
        return 18
    elif row['new'] == 'f#' or row['new'] == 'g-':
        return 19
    elif row['new'] == 'g':
        return 20
    elif row['new'] == 'g#' or row['new'] == 'a-':
        return 21
    elif row['new'] == 'a':
        return 22
    elif row['new'] == 'a#' or row['new'] == 'b-':
        return 23
    elif row['new'] == 'b':
        return 24
    elif row['new'] == 'b#' or row['new'] == "c-\'":
        return 25
    elif row['new'] == "c\'":
        return 26
    elif row['new'] == "c#\'" or row['new'] == "d-\'":
        return 27
    elif row['new'] == "d\'":
        return 28
    elif row['new'] == "d#\'" or row['new'] == "e-\'":
        return 29
    elif row['new'] == "e\'":
        return 30
    elif row['new'] == "f\'":
        return 31
    elif row['new'] == "f#\'" or row['new'] == "g-\'":
        return 32
    elif row['new'] == "g\'":
        return 33
    elif row['new'] == "g#\'" or row['new'] == "a-\'":
        return 34
    elif row['new'] == "a\'":
        return 35
    elif row['new'] == "a#\'" or row['new'] == "b-\'":
        return 36
    elif row['new'] == "b\'":
        return 37
    elif row['new'] == "b#\'" or row['new'] == "c-\'\'":
        return 38
    elif row['new'] == "c''":
        return 39
    elif row['new'] == "c#\'\'" or row['new'] == "d-\'\'":
        return 40
    elif row['new'] == "d\'\'":
        return 41
    elif row['new'] == "d#\'\'" or row['new'] == "e-\'\'":
        return 42
    elif row['new'] == "e\'\'":
        return 43
    elif row['new'] == "f\'\'":
        return 44
    elif row['new'] == "f#\'\'" or row['new'] == "g-\'\'":
        return 45
    elif row['new'] == "g\'\'":
        return 46
    elif row['new'] == "g#\'\'" or row['new'] == "a-\'\'":
        return 47
    elif row['new'] == "a\'\'":
        return 48
    elif row['new'] == "a#\'\'" or row['new'] == "b-\'\'":
        return 49
    elif row['new'] == "b\'\'":
        return 50

    else:
        return np.NaN    
    
def patterns(df):
    df = df[df.final != 'r']
    df = df.dropna(axis=0, subset=['notes'])
    columns = ['final','line', 'notes']
    df['sequence'] = df.apply(pred_sequence, axis=1)
    df['difference'] = df['sequence'].shift(-1) - df['sequence']
    columns = ['final','new','difference', 'notes']
    df=df[columns]
    df['6prev'] = df['difference'].shift(6)
    df['5prev'] = df['difference'].shift(5)
    df['4prev'] = df['difference'].shift(4)
    df['3prev'] = df['difference'].shift(3)
    df['2prev'] = df['difference'].shift(2)
    df['prev'] = df['difference'].shift(1)
    df['current'] = df['difference']
    df['next'] = df['difference'].shift(-1)
    df['2next'] = df['difference'].shift(-2)
    df['3next'] = df['difference'].shift(-3)
    df['4next'] = df['difference'].shift(-4)
    df['5next'] = df['difference'].shift(-5)
    df['6next'] = df['difference'].shift(-6)
    df['prevnote'] = df['final'].shift(1)
    
#     
    df['nextnote'] = df['final'].shift(-1)
#     df['prevnote'] = df['prevnote'].str.replace('#','')
    df = pd.DataFrame(df).fillna(0)
    return df

def gold(X_X):
    import re
    df = pd.DataFrame()
    words = []
    for line in open('Kyrie.txt'):
        for word in line.split():
            word = word.replace(u'\\', u'')
            word = word.replace(".", "")
            word = word.replace("~", "")
            word = word.replace("breve", "")
            word = re.sub('\d', '', word)
            words.append(word)
    df = pd.Series( (v for v in words) )
    df = pd.DataFrame(df)
    df.columns = ['new']
    df = df.loc[df['new']!='r']
    df['sequence'] = df.apply(sequence, axis = 1)
    df['difference'] = df['sequence'].shift(-1) - df['sequence']
    df['6prev'] = df['difference'].shift(6)
    df['5prev'] = df['difference'].shift(5)
    df['4prev'] = df['difference'].shift(4)
    df['3prev'] = df['difference'].shift(3)
    df['2prev'] = df['difference'].shift(2)
    df['prev'] = df['difference'].shift(1)
    df['current'] = df['difference']
    df['next'] = df['difference'].shift(-1)
    df['2next'] = df['difference'].shift(-2)
    df['3next'] = df['difference'].shift(-3)
    df['4next'] = df['difference'].shift(-4)
    df['5next'] = df['difference'].shift(-5)
    df['6next'] = df['difference'].shift(-6)
    df = pd.DataFrame(df).fillna(0)
    from sklearn.model_selection import train_test_split
    X=df[['3prev', '2prev', 'prev', 'next', '2next']] # Features

    y=df['current']  # Labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
    from sklearn.ensemble import RandomForestClassifier

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100, random_state=27)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_X)
    return y_pred

def get_gold():
    dff = pd.DataFrame()
    words = []
    for line in open('Tenor.txt'):
        for word in line.split():
            word = word.replace(u'\\', u'')

            word = word.replace("breve", "")
            word = word.replace("r", "")
            word = word.replace("'", "")
            word = word.replace(",", "")
            word = word.replace(".", "")
            word = word.replace("~", "")

            word = re.sub('\d', '', word)
            words.append(word)
    dff = pd.Series( (v for v in words) )
    dff = pd.DataFrame(dff)
    dff.columns = ['new']
    str5 = dff['new'].values
    tenor_gold = ' '.join(filter(None, str5))
    return tenor_gold
def get_pred(df):
    str4 = df['final'].values
    str1 = ' '.join(filter(None, str4))
    str1 = re.sub('\d', '', str1)
    tenor_pred = str1.lower()
    return tenor_pred

def make_new(row):
    try:
        l = ['C2', 'C2#', 'D2', 'D2#', 'E2','E2#', 'F2','F2#', 'G2','G2#', 'A2','A2#', 'B2','B2#', 'C3','C3#','D3','D3#', 'E3','E3#', 'F3','F3#', 'G3','G3#', 'A3','A3#', 'B3','B3#','C4','C4#','D4','D4#', 'E4','E4#', 'F4','F4#', 'G4','G4#', 'A4','A4#', 'B4','B4#','C5','C5#','D5','D5#', 'E5','E5#', 'F5','F5#', 'G5','G5#', 'A5','A5#', 'B5','B5#', ]
        if row['prevnote'] != '':
            c = l.index(row['prevnote'])
            return l[c-int(row['y_pred'])]
#         
        else:
            c = l.index(row['nextnote'])
            return l[c+int(row['y_pred'])]
    except:
        pass
def make_new_2(row):
    try:
        l = ['C2', 'C2#', 'D2', 'D2#', 'E2','E2#', 'F2','F2#', 'G2','G2#', 'A2','A2#', 'B2','B2#', 'C3','C3#','D3','D3#', 'E3','E3#', 'F3','F3#', 'G3','G3#', 'A3','A3#', 'B3','B3#','C4','C4#','D4','D4#', 'E4','E4#', 'F4','F4#', 'G4','G4#', 'A4','A4#', 'B4','B4#','C5','C5#','D5','D5#', 'E5','E5#', 'F5','F5#', 'G5','G5#', 'A5','A5#', 'B5','B5#', ]
        if row['previousnote'] != '':
            c = l.index(row['previousnote'])
            return l[c-int(row['y_pred'])]
        else:
            c = l.index(row['nextusnote'])
            return l[c+int(row['y_pred'])]
    except:
        pass
def make_prediction(df2):
    y = df2[df2.final == '']
    y['previousnote'] = y['final'].shift(1)

    y['nextusnote'] = y['final'].shift(-1)
    X = y[['3prev', '2prev', 'prev', 'next', '2next']]
    y['y_pred'] = gold(X)
    y['final'] = y.apply(make_new_2, axis=1)
    df2['final'].loc[y.index.values] = y['final'].loc[y.index.values]
    str4 = df2['final'].values
    str1 = ' '.join(filter(None, str4))
    str1 = re.sub('\d', '', str1)
    tenor_pred = str1.lower()
    return df2
def make_prediction_2(df2):
    y = df2[df2['final'].isnull()]
    
    X = y[['3prev', '2prev', 'prev', 'next', '2next']]
    y['y_pred'] = gold(X)
    y['final'] = y.apply(make_new, axis=1)
    df2['final'].loc[y.index.values] = y['final'].loc[y.index.values]
    str4 = df2['final'].values
    str1 = ' '.join(filter(None, str4))
    str1 = re.sub('\d', '', str1)
    tenor_pred = str1.lower()
    return tenor_pred
def pred_sequence(row):
    if row['new'] == 'C2':
        return 0
    elif row['new'] == 'C2#' or row['new'] == 'D2-':
        return 1
    elif row['new'] == 'D2':
        return 2
    elif row['new'] == 'D2#' or row['new'] == 'E2-':
        return 3
    elif row['new'] == 'E2':
        return 4
    elif row['new'] == 'F2':
        return 5
    elif row['new'] == 'F2#' or row['new'] == 'G2-':
        return 6
    elif row['new'] == 'G2':
        return 7
    elif row['new'] == 'G2#' or row['new'] == 'A2-':
        return 8
    elif row['new'] == 'A2':
        return 9
    elif row['new'] == 'A2#' or row['new'] == 'B2-':
        return 10
    elif row['new'] == 'B2':
        return 11
    elif row['new'] == 'B2#' or row['new'] == 'C3-':
        return 12
    elif row['new'] == 'C3':
        return 13
    elif row['new'] == 'C3#' or row['new'] == 'D3-':
        return 14
    elif row['new'] == 'D3':
        return 15
    elif row['new'] == 'D3#' or row['new'] == 'E3-':
        return 16
    elif row['new'] == 'E3':
        return 17
    elif row['new'] == 'F3':
        return 18
    elif row['new'] == 'F3#' or row['new'] == 'G3-':
        return 19
    elif row['new'] == 'G3':
        return 20
    elif row['new'] == 'G3#' or row['new'] == 'A3-':
        return 21
    elif row['new'] == 'A3':
        return 22
    elif row['new'] == 'A3#' or row['new'] == 'B3-':
        return 23
    elif row['new'] == 'B3':
        return 24
    elif row['new'] == 'B3#' or row['new'] == "C4":
        return 25
    elif row['new'] == "C4":
        return 26
    elif row['new'] == "C4#" or row['new'] == "D4-":
        return 27
    elif row['new'] == "D4":
        return 28
    elif row['new'] == "D4#" or row['new'] == "E4-":
        return 29
    elif row['new'] == "E4":
        return 30
    elif row['new'] == "F4":
        return 31
    elif row['new'] == "F4#" or row['new'] == "G4-":
        return 32
    elif row['new'] == "G4":
        return 33
    elif row['new'] == "G4#" or row['new'] == "A4-":
        return 34
    elif row['new'] == "A4":
        return 35
    elif row['new'] == "A4#" or row['new'] == "B4-":
        return 36
    elif row['new'] == "B4":
        return 37
    elif row['new'] == "B4#" or row['new'] == "C5-":
        return 38
    elif row['new'] == "C5":
        return 39
    elif row['new'] == "C5#" or row['new'] == "D5-":
        return 40
    elif row['new'] == "D5":
        return 41
    elif row['new'] == "D5#" or row['new'] == "E5-":
        return 42
    elif row['new'] == "E5":
        return 43
    elif row['new'] == "F5":
        return 44
    elif row['new'] == "F5#" or row['new'] == "G5-":
        return 45
    elif row['new'] == "G5":
        return 46
    elif row['new'] == "G5#" or row['new'] == "A5-":
        return 47
    elif row['new'] == "A5":
        return 48
    elif row['new'] == "A5#" or row['new'] == "B5-":
        return 49
    elif row['new'] == "B5":
        return 50

    else:
        return np.NaN  
    
def bronze(df,X_X):
    
    from sklearn.model_selection import train_test_split
    X=df[['3prev', '2prev', 'prev', 'next', '2next']] # Features

    y=df['current']  # Labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
    from sklearn.ensemble import RandomForestClassifier

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100, random_state=27)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_X)
    return y_pred
def bronze_prediction(df2):
    y = df2[df2.final == '']
    y['previousnote'] = y['final'].shift(1)

    y['nextusnote'] = y['final'].shift(-1)
    X = y[['3prev', '2prev', 'prev', 'next', '2next']]
    y['y_pred'] = bronze(df2, X)
    y['final'] = y.apply(make_new_2, axis=1)
    df2['final'].loc[y.index.values] = y['final'].loc[y.index.values]
    str4 = df2['final'].values
    str1 = ' '.join(filter(None, str4))
    str1 = re.sub('\d', '', str1)
    tenor_pred = str1.lower()
    return df2
def bronze_prediction_2(df2):
    y = df2[df2['final'].isnull()]
    
    X = y[['3prev', '2prev', 'prev', 'next', '2next']]
    y['y_pred'] = bronze(df2, X)
    y['final'] = y.apply(make_new, axis=1)
    df2['final'].loc[y.index.values] = y['final'].loc[y.index.values]
    str4 = df2['final'].values
    str1 = ' '.join(filter(None, str4))
    str1 = re.sub('\d', '', str1)
    tenor_pred = str1.lower()
    return tenor_pred

def get_all(df):
    tenor_pred = get_pred(df)
    tenor_gold = get_gold()
    df2 = patterns(df)
    tenor_pred2 = make_prediction(df2)
    tenor_pred3 = make_prediction_2(tenor_pred2)
    df2 = patterns(df)
    bronzed = bronze_prediction(df2)
    bronzed = bronze_prediction_2(bronzed)
    return tenor_pred, tenor_gold, tenor_pred3, bronzed 



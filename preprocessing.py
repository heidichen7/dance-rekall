from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from rekall.predicates import *
from vgrid import VGridSpec, VideoMetadata, VideoBlockFormat, FlatFormat, SpatialType_Bbox, SpatialType_Keypoints, Metadata_Keypoints
from vgrid_jupyter import VGridWidget
import os, json
import pandas as pd
from const import *
# from detection import 

import json

def load_op_data_all(json_dirs, video_metadata_intel):
    """
    Loads openpose data for all videos.
    Arguments: 
        list json_dirs: directory where opdata jsons are located for each video
        video_metadata_intel: list of VideoMetaData for each video
    Returns: 
        list of frame_lists, each a list of frame-dictionaries
    """
    frame_list_all_videos = []
    for path_to_json, vm in zip(json_dirs, video_metadata_intel):
        frame_list_all_videos.append(load_op_data(path_to_json, vm))
    return frame_list_all_videos
    


def load_op_data(path_to_json, vm, person_metric="largest_bbox"):
    """
    Loads openpose data for single video.
    Arguments:
        keypoint_files: json dir where keypoint.json files are located for one video
            each json file contains dictionary of openpose output info for one frame. 
        vm: VideoMetadata object for one video
    Returns:
        frame_list: list of frames where each frame is a dictionary of part idxs to points {0: [x y conf], 1: etc.} 
    """
    frame_list = []
    keypoint_files = [os.path.join(path_to_json, pos_json) for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    for js in keypoint_files:
        with open(js) as json_file:
            keypoint_data = json.load(json_file)
            
            # preprocess keypoint data per person detected (list of all coords --> dict of parts to coords)
            max_bbox_sz = 0  #TODO  (see below todo)
            selected_part_data = {}
            for person_id in range(len(keypoint_data['people'])):
                pose_keypoints_2d = keypoint_data['people'][person_id]['pose_keypoints_2d']
                part_data = {}
                for index in INDEX_TO_PART:# transform keypoint unstructured list into dictionary of parts
                    keypoint_index = index * POINTS_PER_PART
                    part_data[index] = pose_keypoints_2d[keypoint_index : keypoint_index + POINTS_PER_PART]
                    if len(part_data[index]) != 0: #normalize
                        part_data[index][0] /= vm.width
                        part_data[index][1] /= vm.height
                    else:
                        part_data[index] = [0, 0, 0]
                # select person w/ largest bbox # TODO: change to more generalized metric if needed; just change this snippet plus max_bbox_sz declaration above
                if person_metric == "largest_bbox":
                    curr_bbox_sz = get_bbox_size(part_data)
                elif person_metric == "rightmost_bbox":
                    curr_bbox_sz = get_bbox_maxpt(part_data)
                    
                if curr_bbox_sz > max_bbox_sz:
                    max_bbox_sz = curr_bbox_sz
                    selected_part_data = part_data
            if len(keypoint_data['people']) == 0: # no people detected in this frame, set all values in frame to zero
                selected_part_data = {part_index: [0, 0, 0] for part_index in range(len(INDEX_TO_PART))
                                     }
            frame_list.append(selected_part_data)
    return frame_list

def get_op_bbox(frame):
    """
    Arguments: 
        frame: dictionary of joint indices to normalized coords [x, y, conf]. ie {0: [.5, .5, .98]}
    Returns:
        4 normalized bounding box coordinates x1, x2, y1, y2
    """
    x1 = 1
    x2 = 0
    y1 = 1
    y2 = 0
    for key in frame:
        joint = frame[key]
        if len(joint) != 0:
            if (joint[0] != 0):
                x1 = min(x1, joint[0])
                x2 = max(x2, joint[0])
            if (joint[1] != 0):
                y1 = min(y1, joint[1])
                y2 = max(y2, joint[1])
        
    return x1, x2, y1, y2

def get_bbox_maxpt(frame):
    x1, x2, y1, y2 = get_op_bbox(frame)
    return x2

def get_bbox_size(frame):
    x1, x2, y1, y2 = get_op_bbox(frame)
    return (x2 - x1) * (y2 - y1)

def load_editor_frame(framefile): # NOTE: NOT NORMALIZED, SINCE WE LOAD THIS IN FROM ARBITRARY EDITOR EDITING
    with open(framefile) as json_file:
        keypoint_data = json.load(json_file)
        pose_keypoints_2d = keypoint_data['people'][0]['pose_keypoints_2d']
        part_data = {}
        for index in INDEX_TO_PART:
            keypoint_index = index * POINTS_PER_PART
            part_data[index] = pose_keypoints_2d[keypoint_index : keypoint_index + POINTS_PER_PART]
            if len(part_data[index]) == 0: #normalize
                part_data[index] = [0, 0, 0]
            else:
                part_data[index][0] /= 1280
                part_data[index][1] /= 720
        return part_data


# code that i may need later as base to modify on

# from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt


# try: adding hard threshold
# def evaluate_gaussian(intervals_predict, intervals_truth, vm, segment_length=1, sigma=.5, visualize=True):
#     y_predict_raw = get_action_segments(intervals_predict, vm, segment_length)
#     y_truth_raw = get_action_segments(intervals_truth, vm, segment_length)
    
#     #smooth, but keep all the 1s as 1s
#     y_predict = np.maximum(gaussian_filter(y_predict_raw, sigma), y_predict_raw)
#     y_truth = np.maximum(gaussian_filter(y_truth_raw, sigma), y_truth_raw)

    
#     if visualize:
#         plt.plot(range(len(y_predict_raw)), y_predict_raw, label="original")
#         plt.plot(range(len(y_predict)), y_predict, label="smoothed")
#         plt.legend()
#         plt.title("Predictions")
#         plt.show()
    
#     #calculate recall
#     print("Evaluating with segment length = {}s, sigma = {}s".format(segment_length, sigma))
#     relative_score = y_predict.dot(y_truth)
#     print ("Relative score (cont. dot product): {}".format(relative_score))    
#     print()
                   
# for i in range(1, 20, 2):
#     segment_length = i / 10.0
#     evaluate_gaussian(rekall_labels, hand_labels, vm, segment_length, visualize=False)

    
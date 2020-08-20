import numpy as np
from scipy import spatial
from preprocessing import get_op_bbox
from const import *
from helpers import *
from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from rekall.predicates import *

def get_coords_dict(interval):
    return interval['payload']['metadata']['pose'].to_json()['args']['keypoints']

def coords_in_bbox(pose, bbox):
    """
    Return coords normalized to interval's bbox as dict of np arrays. origin at box center, (.5, .5)
    """
    x1, x2, y1, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    origin = np.array([x1, y1])
    scale = np.array([width, height])
    normalized_pose = {}
    for joint in pose:
        normalized_pose[joint] = np.array(pose[joint][:-1])
        normalized_pose[joint] = (normalized_pose[joint] - origin) / scale - np.array([.5, .5])
    return normalized_pose
    
def bbox_similarity(interval, ref_pose): # aspec ratio x:y
    interval_bounds = [interval['bounds']['x1'], interval['bounds']['x2'], interval['bounds']['y1'], interval['bounds']['y2']]
    interval_aspect = (interval_bounds[1] - interval_bounds[0]) / (interval_bounds[3] - interval_bounds[2])
    ref_pose_bounds = get_op_bbox(ref_pose)
    ref_pose_aspect = (ref_pose_bounds[1] - ref_pose_bounds[0]) / (ref_pose_bounds[3] - ref_pose_bounds[2])
    
    return 1 - abs(ref_pose_aspect - interval_aspect) / ref_pose_aspect
        
    
def joint_similarity(interval, ref_pose): 
    # Pose_frame is a dictionary of {0: [x, y, conf]} for one frame.
    # re align all coords to be relative to bounding box top left corner, and scaled to bounding box
    # (to account for far away dancers being smaller in the video frame)
    interval_pose_orig = get_coords_dict(interval)
    interval_bounds_orig = [interval['bounds']['x1'], interval['bounds']['x2'], interval['bounds']['y1'], interval['bounds']['y2']]
    ref_pose_orig = ref_pose
    ref_pose_bounds_orig = get_op_bbox(ref_pose_orig)
    
    # reset to bbox-normalized version of coords
    interval_pose = coords_in_bbox(interval_pose_orig, interval_bounds_orig)
    ref_pose = coords_in_bbox(ref_pose_orig, ref_pose_bounds_orig)

    total_dist = 0
    # get sum of distances between interval pose and ref - should be between 0 and 1 b/c we normalized
    for joint in interval_pose:
        #check for "zeros" aka joint not found, from original coords
        if interval_pose_orig[joint] == [0, 0, 0]:
#             print("Skipping {}".format(joint))
            continue
#         if joint in range(15,25):
#             continue
        curr_dist = spatial.distance.cosine(interval_pose[joint],ref_pose[joint])
#         print("Distance for joint {}: {}".format(joint,curr_dist))
#         print(curr_dist)
        total_dist += curr_dist
    total_dist /= len(interval_pose)
    
    return 1.0 - total_dist


def concat_frames(payload1, payload2):
    if isinstance(payload1, list): # if payload1 is an already running list of op frame dicts
        return payload1.append(payload2)
    else: # otherwise, start the running list
        return [payload1, payload2]
    
def detected(interval): 
    """
    Check if any joints were set to 0,0,0 (aka, not detected)
    """
    for joint_idx in INDEX_TO_PART:
        if joint_idx == PART_TO_INDEX['Background']:
            continue
        coords = get_coords(interval, joint_idx)
        if coords == [0,0,0]:
            return False
    return True

def search_pose_sequence(interval_mapping, poses, seconds_between=1, similarity_threshold=.95):
    """
    Search for any interval with a sequence of poses. 
    Args:
        poses: list of reference frames (dictionaries of openpose data)
        seconds_between: max number of seconds between poses for a valid sequence (TODO: make this a list)
        similarity_threshold: % similar poses must be to count as "detected"
    Returns:
        fully coalesced and joined intervals that contain the sequence of poses.
    Future implementation add-ons: 
        specifying epsilons / time leeway between each pose

    """
    # get segments for first pose
    frames = interval_mapping.filter(detected).filter(lambda interval: joint_similarity(interval, poses[0]) >= similarity_threshold)
    output= frames.coalesce(('t1', 't2'),
        bounds_merge_op = Bounds3D.span, 
        payload_merge_op = concat_frames,
        epsilon=.1)
    # for each remaining pose, coalesce and join segments with running output
    for pose in poses[1:]:
        frames = interval_mapping.filter(detected)
        frames = frames.filter(lambda interval: joint_similarity(interval, pose) >= similarity_threshold)
        segments = frames.coalesce(('t1', 't2'),
            bounds_merge_op = Bounds3D.span, 
            payload_merge_op = concat_frames,
            epsilon=.1)
        output = output.join(segments,
            predicate = or_pred(overlaps(), before(max_dist = seconds_between)),
            merge_op = lambda i1, i2: Interval(i1['bounds'].span(i2['bounds'])),
            window = 1)
    return output


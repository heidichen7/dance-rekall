import numpy as np
from const import *

# GENERAL HELPER FUNCTIONS
def not_zero(coords): #if x,y = 0,0, usually means part not detected
    return coords[0] != 0 and coords[1] != 0

def get_coords(interval, joint_idx):
    md = interval['payload']['metadata']['pose'].to_json()
    return md['args']['keypoints'][joint_idx]

def is_close_to(interval, jointname1, jointname2, magic_threshold = .1):
    joint1 = get_coords(interval, PART_TO_INDEX[jointname1])
    joint2 = get_coords(interval, PART_TO_INDEX[jointname2])
    bbox_width = interval.bounds['x2'] - interval.bounds['x1']
    bbox_height = interval['y2'] - interval.bounds['y1']
    x_dist = abs(joint2[0] - joint1[0])
    y_dist = abs(joint2[1] - joint1[1])
    return not_zero(joint1) and not_zero(joint2) \
        and x_dist <= magic_threshold * bbox_width and y_dist <= magic_threshold * bbox_height

# NOTE: openpose directions are relative to the way the dancer/subject is facing, aka OPPOSITE of viewer.
def is_left(interval, jointname1, jointname2):
    joint1 = get_coords(interval, PART_TO_INDEX[jointname1])
    joint2 = get_coords(interval, PART_TO_INDEX[jointname2])
    return not_zero(joint1) and not_zero(joint2) and joint1[0] > joint2[0]

def is_right(interval, jointname1, jointname2):
    joint1 = get_coords(interval, PART_TO_INDEX[jointname1])
    joint2 = get_coords(interval, PART_TO_INDEX[jointname2])
    return not_zero(joint1) and not_zero(joint2) and joint1[0] < joint2[0]

def is_below(interval, jointname1, jointname2):
    joint1 = get_coords(interval, PART_TO_INDEX[jointname1])
    joint2 = get_coords(interval, PART_TO_INDEX[jointname2])
    return not_zero(joint1) and not_zero(joint2) and joint1[1] > joint2[1]

def is_above(interval, jointname1, jointname2):
    joint1 = get_coords(interval, PART_TO_INDEX[jointname1])
    joint2 = get_coords(interval, PART_TO_INDEX[jointname2])
    return not_zero(joint1) and not_zero(joint2) and joint1[1] < joint2[1]

def calc_angle(vec1, vec2):
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)
    dotproduct = np.dot(vec1, vec2)
    return np.degrees(np.arccos(dotproduct))

def angle_from_x(interval, jointname1, jointname2): #coord1 is closer to origin / top left corner of screen. "from x" meaning x axis, top left corner to top right corner
    coord1 = get_coords(interval, PART_TO_INDEX[jointname1])
    coord2 = get_coords(interval, PART_TO_INDEX[jointname2])
    if not_zero(coord1) and not_zero(coord2):
        x_vec = [1, 0]
        line_vec = [coord2[0]-coord1[0], coord2[1]-coord1[1]]
        return calc_angle(x_vec, line_vec)
    return -1
#     x_vec /= np.linalg.norm(x_vec)
#     line_vec /= np.linalg.norm(line_vec)
    
#     return np.degrees(np.arccos(np.dot(x_vec, line_vec)))

def smaller_angle_from_x(interval, pair1joint1, pair1joint2, pair2joint1, pair2joint2, epsilon=10): #Main effect: less tilted
    """
    Takes in 4 joints which compose 2 body parts / bones (i.e., the spine, right forearm, etc.)
    Part 1 formed by pair1joint1--> pair1joint2 , Part 2 formed by pair2joint1-->pair2joint2
    Returns true if Part 1 has a greater angle from the x axis than Part 2. 
    """
#     p1j1, p1j2, p2j1, p2j2 = [get_coords(interval, PART_TO_INDEX[name]) for name in 
#                              [pair1joint1, pair1joint2, pair2joint1, pair2joint2]]
#     is_not_zero = all([not_zero(joint) for joint in [p1j1, p1j2, p2j1, p2j2]])
#     if is_not_zero:
    angle1 = angle_from_x(interval, pair1joint1, pair1joint2)
    angle2 = angle_from_x(interval, pair2joint1, pair2joint2)
    if (angle1 != -1 and angle2 != -1): #both found:
        return angle1 + epsilon < angle2
    return False

#next three functions are from w3schools
def on_segment(p, q, r):
    # check if r lies on (p,q)
    if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
        return True
    return False

def orientation(p, q, r):
    # return 0/1/-1 for colinear/clockwise/counterclockwise
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0 : return 0
    return 1 if val > 0 else -1

def intersects(seg1, seg2):
    # check if seg1 and seg2 intersect

    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
    # find all orientations

    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
    # check general case
        return True

    if o1 == 0 and on_segment(p1, q1, p2) : return True
    # check special cases

    if o2 == 0 and on_segment(p1, q1, q2) : return True
    if o3 == 0 and on_segment(p2, q2, p1) : return True
    if o4 == 0 and on_segment(p2, q2, q1) : return True

    return False


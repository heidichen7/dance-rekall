import rekallpy
import os, json
import pandas as pd

# OUTLINE:
# For each video, combine all keypoint.json files into one dictionary with frame # : keypoint content
# Filter down this dictionary to lower frame rate (1f/second for now?) into dictionary of second # : keypoint content (TODO REFINE / OPTIMIZE FOR FULL FRAME RATE)
# convert unformatted list of keypoint content to mapped version BODY25

frame_rate = 29.84  # video frame rate, which we'll use to convert keypoitn data to timestamps. openpose generates on keypoint per frame. TODO update to generalize to all vids
# frames_to_save_per_second = 1
# frame_rate = frame_rate // frames_to_save_per_second

path_to_json = "C:/Users/heidi/Documents/seniorproject/openpose-1.5.1-binaries-win64-only_cpu-python-flir-3d/openpose-1.5.1-binaries-win64-only_cpu-python-flir-3d/openpose/long_output/"
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

# here I define my pandas Dataframe with the columns I want to get from the json
jsons_data = pd.DataFrame(columns=['person_id', 'pose_keypoints_2d'])

# we need both the json and an index number so use enumerate()
# for index, js in enumerate(json_files):
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)
        person_id = json_text['people'][0]['person_id']
        pose_keypoints_2d = json_text['people'][0]['pose_keypoints_2d']
        # WHERE I LEFT OFF : ADDING THE TIME STAMP TO THE DATAFRAME
        # here I push a list of data into a pandas DataFrame at row given by 'index'
        jsons_data.loc[index] = [person_id, pose_keypoints_2d]

# now that we have the pertinent json data in our DataFrame let's look at it
print(jsons_data)

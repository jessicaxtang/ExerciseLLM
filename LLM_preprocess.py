# Preprocess UI-PRMD dataset for input to LLM
# 1. temporal downsample
# 2. slice most relevant joints

"""
TO DO:
Jul 4, 2024
- generalize m, s, e for inputs
- include angles too, not just positions (maybe skeleton needed, from RehabExerAssess repo 'preprocess.py')
- make downsampling rate a parameter
- make relevant joints a parameter
- maybe turn this entire file into a function for pos BODY JOINTS
- also make another function for FEATURES
"""

import os
import pandas as pd
import numpy as np
import options.option_data as option_data
from utils.skel_features import RShAbd_features, LShAbd_features
from utils.skel_conversions import rel2abs, transform_data

# run below command in terminal 
# ex. UI-PRMD\correct\kinect\positions\m07_s01_e01_positions.txts
'''
python LLM_preprocess.py --dataname UI-PRMD --input_type pos --downsample 5 --joints 12 13 14 --device kinect --correctness correct --subdir positions --m 7 --s 1 --e 1

python LLM_preprocess.py --dataname UI-PRMD --input_type feat --downsample 1 --joints 12 13 14 --device kinect --correctness incorrect --subdir positions --m 7 --s 1 --e 4

python3 LLM_preprocess.py --dataname UI-PRMD --input_type feat --downsample 3 --joints 12 13 14 --device kinect --correctness correct --subdir positions --m 7 --s 7 --e 5
'''

# order of joint connections
J = np.array([[3, 5, 4, 2, 1, 2, 6, 7, 8, 2, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20],
              [2, 4, 2, 1, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])

def relevant_frames(features):
    '''
    features: a numpy array of features extracted from the data
    downsample_rate: int
    '''

    indices = []

    frame_max = np.argmax(features, axis=0) # indices with max shoulder abduction angle
    frame_max = frame_max[frame_max != 0]
    frame_max = np.sort(frame_max).tolist()
    indices.extend(frame_max)
    # print("frame_max: ", frame_max)

    # min features BEFORE first max angles
    # print("features[frame_max[0]]", features[frame_max[0]])
    # print("features[frame_max[1]]", features[frame_max[1]])
    # print("features[frame_max[2]]", features[frame_max[2]])
    frame_min = np.argmin(features[:frame_max[0]], axis=0) # indices with min shoulder abduction angle
    frame_min = frame_min[frame_min != 0]
    frame_min = np.sort(frame_min).tolist()
    indices.extend(frame_min)
    # print("frame_min BEFORE: ", frame_min)

    # min features AFTER last max angles
    # print("features[frame_max[-1]+1:]", features[frame_max[-1]+1:])
    frame_min = np.argmin(features[frame_max[-1]+1:], axis=0) # indices with min shoulder abduction angle
    frame_min = frame_min[frame_min != 0]
    frame_min += frame_max[-1]
    frame_min = np.sort(frame_min).tolist()
    indices.extend(frame_min)



    indices = sorted(indices)

    # print("sorted indices: ", indices)
    # print("type(indices): ", type(indices))

    features_sliced = features[indices]
    # print("features_sliced", features_sliced)

    return features_sliced

def relevant_frames1(features, downsample_rate):

    indices = []

    frame_max = np.argmax(features, axis=0) # indices with max shoulder abduction angle
    frame_max = np.sort(frame_max).tolist()
    indices.extend(frame_max)

    # min features BEFORE first max angles
    before = list(range(0, frame_max[0], downsample_rate))
    indices.extend(before)

    # min features AFTER last max angles
    after = list(range(frame_max[-1], features.shape[0], downsample_rate))
    indices.extend(after)

    indices = sorted(indices)

    # print("sorted indices: ", indices)
    # print("type(indices): ", type(indices))

    features_sliced = features[indices]
    # print("features_sliced", features_sliced)

    return features_sliced

# def relevant_features(data, features):
#     '''
#     features: a list of indices of most relevant features for this exercise
#     as defined in Exercise-Specific Feature Extraction Approach for Assessing Physical Rehabilitation (Guo 2021)

#     0 left elbow angle
#     1 right elbow angle
#     2 hand shoulder ratio
#     3 torso tilted angle 
#     4 hand tilted angle
#     5 elbow angles diff
#     6 left shoulder angle
#     7 right shoulder angle
#     8 left arm torso angle
#     9 right arm torso angle
#     10 knee hip ratio
#     11 shoulder level angle
#     12 left knee angle
#     13 right knee angle
#     '''
#     features_data = []
#     for feature in features:
#         if feature == 1:
#             features_data.append(data[:, 13] - data[:, 12])
#         features_data.append()
#     return features_data

def preprocess_joints_old(data_file, downsample_rate):
    # pick a few joints relevant joints for shoulder abduction
    # for example, 12: right upper arm, 13: right forearm, 14: right hand
    column_names = ['Right Upper Arm', 'Right Forearm', 'Right Hand']
    data_relevant = preprocess_pos(data_file, joints)

    # downsample and convert to pandas dataframe
    print((data_relevant[::downsample_rate, :]).shape)
    df_sliced = pd.DataFrame(data_relevant[::downsample_rate, :], columns=column_names)

    return df_sliced

def preprocess_joints(pos_data, ang_data, num_kp, num_axes, num_frames):
    p_data = transform_data(pos_data, num_kp, num_axes)
    a_data = transform_data(ang_data, num_kp, num_axes)

    p = np.copy(p_data)
    a = np.copy(a_data)

    skel = rel2abs(p, a, num_kp, num_axes, num_frames)
    # print("skel.shape: ", skel.shape)
    # print(skel)
    
    return np.transpose(skel, (2, 0, 1))

def preprocess_pos(data, dominant_side):
    '''
    joints: a list of most relevant joints ranging from 1 to 22 for kinect
    '''
    joints = []
    column_names = ['Waist_X', 'Waist_Y', 'Waist_Z', 
                    'Spine_X', 'Spine_Y', 'Spine_Z', 
                    'Chest_X', 'Chest_Y', 'Chest_Z', 
                    'Neck_X', 'Neck_Y', 'Neck_Z', 
                    'Head_X', 'Head_Y', 'Head_Z', 
                    'HeadTip_X', 'HeadTip_Y', 'HeadTip_Z', 
                    'LeftCollar_X', 'LeftCollar_Y', 'LeftCollar_Z', 
                    'LeftUpperArm_X', 'LeftUpperArm_Y', 'LeftUpperArm_Z', 
                    'LeftForearm_X', 'LeftForearm_Y', 'LeftForearm_Z', 
                    'LeftHand_X', 'LeftHand_Y', 'LeftHand_Z', 
                    'RightCollar_X', 'RightCollar_Y', 'RightCollar_Z', 
                    'RightUpperArm_X', 'RightUpperArm_Y', 'RightUpperArm_Z', 
                    'RightForearm_X', 'RightForearm_Y', 'RightForearm_Z', 
                    'RightHand_X', 'RightHand_Y', 'RightHand_Z', 
                    'LeftUpperLeg_X', 'LeftUpperLeg_Y', 'LeftUpperLeg_Z', 
                    'LeftLowerLeg_X', 'LeftLowerLeg_Y', 'LeftLowerLeg_Z', 
                    'LeftFoot_X', 'LeftFoot_Y', 'LeftFoot_Z', 
                    'LeftLegToes_X', 'LeftLegToes_Y', 'LeftLegToes_Z', 
                    'RightUpperLeg_X', 'RightUpperLeg_Y', 'RightUpperLeg_Z', 
                    'RightLowerLeg_X', 'RightLowerLeg_Y', 'RightLowerLeg_Z', 
                    'RightFoot_X', 'RightFoot_Y', 'RightFoot_Z', 
                    'RightLegToes_X', 'RightLegToes_Y', 'RightLegToes_Z'
                ]
    if dominant_side == 'right':
        joints = [0, 2, 11, 12, 13] # waist, chest, right shoulder, right elbow, right hand

    else: # 'left'
        joints = [0, 2, 7, 8, 9] # waist, chest, left shoulder, left elbow, left hand

    # Create a list of column indices for the relevant joints
    joint_indices = []
    for j in joints:
        base = j * 3
        joint_indices.extend([base, base + 1, base + 2])

    positions = data[:, joint_indices]
    column_names = [column_names[i] for i in joint_indices]
    data_file = pd.DataFrame(positions, columns=column_names)
    
    return data_file

def preprocess_features(skel_T, dominant_side='right'):
    if dominant_side == 'right':
        features = np.array(RShAbd_features(skel_T), dtype=int)
    else: # 'left'
        features = np.array(LShAbd_features(skel_T), dtype=int)
    column_names = ['Shoulder Abduction Angle', 'Elbow Flexion Angle', 'Torso Inclination Angle']
    # print(features)
    # downsample and convert to pandas dataframe
    # features_sliced = relevant_frames(features)
    # features_sliced = relevant_frames1(features, downsample_rate)
    data_file = pd.DataFrame(features, columns=column_names)

    return data_file


if __name__ == '__main__':

    # define arguments
    args = option_data.get_args_parser()
    dataname = args.dataname
    input_type = args.input_type
    downsample_rate = args.downsample
    joints = args.joints
    device = args.device
    correctness = args.correctness
    cor_tag = ''
    if correctness == 'incorrect':
        cor_tag = '_inc'
    subdir = args.subdir
    m = args.m
    s = args.s
    e = args.e

    num_kp = 22 # turn into a parameter
    num_axes = 3 # turn into a parameter
    dominant_side = 'right'

    # load data
    pos_path = 'dataset/{}/{}/{}/positions/m{:02d}_s{:02d}_e{:02d}_positions{}.txt'.format(dataname, correctness, device, m, s, e, cor_tag)
    ang_path = 'dataset/{}/{}/{}/angles/m{:02d}_s{:02d}_e{:02d}_angles{}.txt'.format(dataname, correctness, device, m, s, e, cor_tag)

    print("pos_path: ", pos_path)
    print("ang_path: ", ang_path)

    pos_data = np.loadtxt(pos_path, delimiter=',')
    ang_data = np.loadtxt(ang_path, delimiter=',')

    num_frames = pos_data.shape[0] # pos and ang data should have the same number of frames
    print("NUM_FRAMES:", num_frames)
    data_file = 0
    if input_type == 'pos':
        # data_file = preprocess_joints(pos_data, downsample_rate)
        skel_T = preprocess_joints(pos_data, ang_data, num_kp, num_axes, num_frames)
        skel_T = np.reshape(skel_T, (skel_T.shape[0], num_axes*num_kp))
        # print(skel_T.shape)
        data_file = preprocess_pos(skel_T, dominant_side)

    elif input_type == 'feat':
        skel_T = preprocess_joints(pos_data, ang_data, num_kp, num_axes, num_frames)
        data_file = preprocess_features(skel_T, dominant_side)

    # save_dir = 'dataset/{}/{}/{}/{}_{}'.format(dataname, correctness, device, input_type, subdir)
    save_dir = f'dataset/{dataname}_{input_type}/{correctness}/{device}/{input_type}'
    # print("Save directory: ", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save_path = subdir + '_' + correctness + '_m{:02d}_s{:02d}_e{:02d}.npy'.format(m, s, e)
    save_path = f'dataset/{dataname}_{input_type}/{correctness}/{device}/{input_type}/' + 'm{:02d}_s{:02d}_e{:02d}{}_{}'.format(m, s, e, cor_tag, input_type)
    print("Save path: ", save_path)
    # np.save(save_path, data_file)

    data_file.to_csv(save_path + '.csv', index=False)

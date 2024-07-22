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
- maybe turn this entire file into a function for RAW BODY JOINTS
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
python LLM_preprocess.py --dataname UI-PRMD --input_type raw --downsample 5 --joints 12 13 14 --device kinect --correctness correct --subdir positions --m 7 --s 1 --e 1

python3 LLM_preprocess.py --dataname UI-PRMD --input_type features --downsample 1 --joints 12 13 14 --device kinect --correctness incorrect --subdir positions --m 7 --s 1 --e 4

python3 LLM_preprocess.py --dataname UI-PRMD --input_type features --downsample 3 --joints 12 13 14 --device kinect --correctness correct --subdir positions --m 7 --s 7 --e 5
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

def relevant_joints(data, joints):
    '''
    joints: a list of most relevant joints ranging from 1 to 22 for kinect, 1 to 39 for vicon
    '''
    return data[:, joints]

def relevant_features(data, features):
    '''
    features: a list of indices of most relevant features for this exercise
    as defined in Exercise-Specific Feature Extraction Approach for Assessing Physical Rehabilitation (Guo 2021)

    0 left elbow angle
    1 right elbow angle
    2 hand shoulder ratio
    3 torso tilted angle 
    4 hand tilted angle
    5 elbow angles diff
    6 left shoulder angle
    7 right shoulder angle
    8 left arm torso angle
    9 right arm torso angle
    10 knee hip ratio
    11 shoulder level angle
    12 left knee angle
    13 right knee angle
    '''
    features_data = []
    for feature in features:
        if feature == 1:
            features_data.append(data[:, 13] - data[:, 12])
        features_data.append()
    return features_data

def preprocess_raw_joints(data_file, downsample_rate):
    # pick a few joints relevant joints for shoulder abduction
    # for example, 12: right upper arm, 13: right forearm, 14: right hand
    column_names = ['Right Upper Arm', 'Right Forearm', 'Right Hand']
    data_relevant = relevant_joints(data_file, joints)

    # downsample and convert to pandas dataframe
    print((data_relevant[::downsample_rate, :]).shape)
    df_sliced = pd.DataFrame(data_relevant[::downsample_rate, :], columns=column_names)

    return df_sliced

def preprocess_features(pos_data, ang_data, num_kp, num_axes, num_frames, downsample_rate):
    p_data = transform_data(pos_data, num_kp, num_axes)
    a_data = transform_data(ang_data, num_kp, num_axes)

    p = np.copy(p_data)
    a = np.copy(a_data)

    skel = rel2abs(p, a, num_kp, num_axes, num_frames)
    
    new = np.transpose(skel, (2, 0, 1))

    features = np.array(RShAbd_features(new), dtype=int) # switch out function for feature/dominant side
    column_names = ['Shoulder Abduction Angle', 'Elbow Flexion Angle', 'Torso Inclination Angle']
    # print(features)
    # downsample and convert to pandas dataframe
    # features_sliced = relevant_frames(features)
    features_sliced = relevant_frames1(features, downsample_rate)
    features_sliced = pd.DataFrame(features_sliced, columns=column_names)

    return features_sliced


if __name__ == '__main__':

    # define arguments
    args = option_data.get_args_parser()
    dataname = args.dataname
    input_type = args.input_type
    downsample_rate = args.downsample
    joints = args.joints # maybe remove this if we predefine joints for each exercise
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
    if input_type == 'raw':
        data_file = preprocess_raw_joints(pos_data, downsample_rate)
        # TO FIX: i think the data is wrong
    elif input_type == 'features':
        data_file = preprocess_features(pos_data, ang_data, num_kp, num_axes, num_frames, downsample_rate)

    # save_dir = 'dataset/{}/{}/{}/{}_{}'.format(dataname, correctness, device, input_type, subdir)
    save_dir = 'dataset/{}_LLM/{}/{}/{}'.format(dataname, correctness, device, input_type)
    # print("Save directory: ", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save_path = subdir + '_' + correctness + '_m{:02d}_s{:02d}_e{:02d}.npy'.format(m, s, e)
    save_path = 'dataset/{}_LLM/{}/{}/{}/m{:02d}_s{:02d}_e{:02d}{}_dr{:02d}_{}1'.format(dataname, correctness, device, input_type, m, s, e, cor_tag, downsample_rate, input_type)
    print("Save path: ", save_path)
    # np.save(save_path, data_file)

    data_file.to_csv(save_path + '.csv', index=False)

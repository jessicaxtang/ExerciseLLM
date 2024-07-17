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
from utils.skel_features import extract_features
from utils.skel_conversions import rel2abs, transform_data

# run below command in terminal 
# ex. UI-PRMD\correct\kinect\positions\m07_s01_e01_positions.txt
'''
python LLM_preprocess.py --dataname UI-PRMD --input_type raw --downsample 5 --joints 12 13 14 --device kinect --correctness correct --subdir positions --m 7 --s 1 --e 1

python LLM_preprocess.py --dataname UI-PRMD --input_type features --downsample 1 --joints 12 13 14 --device kinect --correctness correct --subdir positions --m 7 --s 1 --e 1
'''

# order of joint connections
J = np.array([[3, 5, 4, 2, 1, 2, 6, 7, 8, 2, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20],
              [2, 4, 2, 1, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])

# def temporal_downsample(data, downsample_rate):
#     '''
#     downsample_rate: int
#     '''
#     return data[::downsample_rate]

def relevant_joints(data, joints):
    '''
    joints: a list of most relevant joints ranging from 1 to 22 for kinect, 1 to 39 for vicon
    '''
    return data[:, joints]

def preprocess_features(pos_data, ang_data, num_kp, num_axes, downsample_rate):
    p_data = transform_data(pos_data, num_kp, num_axes)
    a_data = transform_data(ang_data, num_kp, num_axes)
    p = np.copy(p_data)
    a = np.copy(a_data)

    skel = rel2abs(p, a, num_kp, num_axes, num_frames)

    new = np.transpose(skel, (2, 0, 1))
    features = np.array(extract_features(new), dtype=int)
    column_names = ['Shoulder Abduction Angle', 'Elbow Flexion Angle', 'Torso Inclination Angle']

    # downsample and convert to pandas dataframe
    df_sliced = pd.DataFrame(features[::downsample_rate, :], columns=column_names)

    return df_sliced

if __name__ == '__main__':

    # define parameters
    dataname = 'UI-PRMD'
    input_type = 'features'
    downsample_rate = 3
    device = 'kinect'
    correctness = 'correct'
    cor_tag = ''
    # if correctness == 'incorrect':
    #     cor_tag = '_inc'
    subdir = 'positions'
    m = 7
    s = 1
    e = 5

    num_kp = 22 # turn into a parameter
    num_axes = 3 # turn into a parameter

    count = 0
    for correctness in ['correct', 'incorrect']:
        if correctness == 'incorrect':
            cor_tag = '_inc'
        for s in range(1, 11): # m
            # for s in range(1, 11):
                # for e in range(1, 11):
                # load data
                pos_path = 'dataset/{}/{}/{}/positions/m{:02d}_s{:02d}_e{:02d}_positions{}.txt'.format(dataname, correctness, device, m, s, e, cor_tag)
                ang_path = 'dataset/{}/{}/{}/angles/m{:02d}_s{:02d}_e{:02d}_angles{}.txt'.format(dataname, correctness, device, m, s, e, cor_tag)

                pos_data = np.loadtxt(pos_path, delimiter=',')
                ang_data = np.loadtxt(pos_path, delimiter=',')

                num_frames = pos_data.shape[0]

                data_file = preprocess_features(pos_data, ang_data, num_kp, num_axes, downsample_rate)

                # save_dir = 'dataset/{}/{}/{}/{}_{}'.format(dataname, correctness, device, input_type, subdir)
                save_dir = 'dataset/{}_LLM/{}/{}/{}'.format(dataname, correctness, device, input_type)
                print("Save directory: ", save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # save_path = subdir + '_' + correctness + '_m{:02d}_s{:02d}_e{:02d}.npy'.format(m, s, e)
                save_path = 'dataset/{}_LLM/{}/{}/{}/m{:02d}_s{:02d}_e{:02d}{}_dr{:02d}_{}'.format(dataname, correctness, device, input_type, m, s, e, cor_tag, downsample_rate, input_type)
                # print("Save path: ", save_path)
                # np.save(save_path, data_file)

                data_file.to_csv(save_path + '.csv', index=False)
                count += 1

    print("Total number of files saved: ", count)

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
# from utils.skel_features import extract_features
# from utils.skel_conversions import rel2abs, transform_data
from LLM_preprocess import preprocess_features

# order of joint connections
J = np.array([[3, 5, 4, 2, 1, 2, 6, 7, 8, 2, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20],
              [2, 4, 2, 1, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])

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
    e = 2

    num_kp = 22 # turn into a parameter
    num_axes = 3 # turn into a parameter

    count = 0
    for correctness in ['correct', 'incorrect']:
        if correctness == 'incorrect':
            cor_tag = '_inc'
        for s in range(1, 11):
            for e in range(1, 11):
                # load data
                pos_path = 'dataset/{}/{}/{}/positions/m{:02d}_s{:02d}_e{:02d}_positions{}.txt'.format(dataname, correctness, device, m, s, e, cor_tag)
                ang_path = 'dataset/{}/{}/{}/angles/m{:02d}_s{:02d}_e{:02d}_angles{}.txt'.format(dataname, correctness, device, m, s, e, cor_tag)

                print("pos_path: ", pos_path)

                pos_data = np.loadtxt(pos_path, delimiter=',')
                ang_data = np.loadtxt(ang_path, delimiter=',')

                num_frames = pos_data.shape[0]

                if s not in [7, 10]: # right
                    data_file = preprocess_features(pos_data, ang_data, num_kp, num_axes, num_frames, 'right')
                else: # if s in [7, 10]: # left
                    data_file = preprocess_features(pos_data, ang_data, num_kp, num_axes, num_frames, 'left')

                # save_dir = 'dataset/{}/{}/{}/{}_{}'.format(dataname, correctness, device, input_type, subdir)
                save_dir = 'dataset/{}_features/{}/{}/{}'.format(dataname, correctness, device, input_type)
                # print("Save directory: ", save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # save_path = subdir + '_' + correctness + '_m{:02d}_s{:02d}_e{:02d}.npy'.format(m, s, e)
                # save_path = 'dataset/{}_LLM/{}/{}/{}/m{:02d}_s{:02d}_e{:02d}{}_dr{:02d}_{}'.format(dataname, correctness, device, input_type, m, s, e, cor_tag, downsample_rate, input_type)
                save_path = 'dataset/{}_features/{}/{}/{}/m{:02d}_s{:02d}_e{:02d}{}_{}'.format(dataname, correctness, device, input_type, m, s, e, cor_tag, input_type)
                print("Save path: ", save_path)
                # np.save(save_path, data_file)

                data_file.to_csv(save_path + '.csv', index=False)
                count += 1

    print("Total number of files saved: ", count)

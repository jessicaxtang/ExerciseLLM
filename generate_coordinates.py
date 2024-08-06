# Preprocess UI-PRMD dataset for input to LLM
# transform data from relative to absolute coordinates
# creates a new directory under './dataset' called UI-PRMD_pos containing the absolute xyz coordinates of the joints

import os
import pandas as pd
import numpy as np
from LLM_preprocess import preprocess_joints, preprocess_pos

# # order of joint connections
# J = np.array([[3, 5, 4, 2, 1, 2, 6, 7, 8, 2, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20],
#               [2, 4, 2, 1, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])

if __name__ == '__main__':

    # define parameters (eventually make it command line arg?)
    dataname = 'UI-PRMD'
    input_type = 'coordinates'
    downsample_rate = 3
    device = 'kinect'
    correctness = 'correct'
    cor_tag = ''
    m = 7
    s = 1
    e = 2

    num_kp = 22 # turn into a parameter if generalize to other datasets
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

                # print("pos_path: ", pos_path)

                pos_data = np.loadtxt(pos_path, delimiter=',')
                ang_data = np.loadtxt(ang_path, delimiter=',')

                num_frames = pos_data.shape[0]
                skel_T = preprocess_joints(pos_data, ang_data, num_kp, num_axes, num_frames)
                skel_T = np.reshape(skel_T, (skel_T.shape[0], num_axes*num_kp))

                if s not in [7, 10]: # right
                    data_file = preprocess_pos(skel_T, 'right')
                else: # if s in [7, 10]: # left
                    data_file = preprocess_pos(skel_T, 'left')

                save_dir = f'dataset/{dataname}_generated/{correctness}/{device}/{input_type}'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_path = f'dataset/{dataname}_generated/{correctness}/{device}/{input_type}/' + 'm{:02d}_s{:02d}_e{:02d}{}_{}'.format(m, s, e, cor_tag, input_type)
                # print("Save path: ", save_path)

                data_file.to_csv(save_path + '.csv', index=False)
                count += 1

    print(f"All {count} files saved! Last file to: {save_dir}")

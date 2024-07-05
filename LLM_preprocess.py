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
import numpy as np
import options.option_data as option_data

# run below command in terminal 
# ex. UI-PRMD\correct\kinect\positions\m07_s01_e01_positions.txt
'''
python LLM_preprocess.py --dataname UI-PRMD --input_type raw --downsample 5 --joints 12 13 14 --device kinect --correctness correct --subdir positions --m 7 --s 1 --e 1
'''

def temporal_downsample(data, downsample_rate):
    '''
    downsample_rate: int
    '''
    return data[::downsample_rate]

def relevant_joints(data, joints):
    '''
    joints: a list of most relevant joints ranging from 1 to 22 for kinect, 1 to 39 for vicon
    '''
    return data[:, joints]

def preprocess_raw_joints(data_file, downsample_rate):
    data_sliced = temporal_downsample(data_file, downsample_rate)
    # pick a few joints relevant joints for shoulder abduction
    # 12: right upper arm, 13: right forearm, 14: right hand
    data_sliced = relevant_joints(data_sliced, joints)
    return data_sliced


# def preprocess_features(device, correctness, subdir, m, s, e):

if __name__ == '__main__':

    # define arguments
    args = option_data.get_args_parser()
    dataname = args.dataname
    input_type = args.input_type
    downsample_rate = args.downsample
    joints = args.joints
    print("Joints: ", joints)
    device = args.device
    correctness = args.correctness
    cor_tag = ''
    if correctness == 'incorrect':
        cor_tag = '_inc'
    subdir = args.subdir
    m = args.m
    s = args.s
    e = args.e



    # load data
    data_path = 'dataset/{}/{}/{}/{}/m{:02d}_s{:02d}_e{:02d}_{}{}.txt'.format(dataname, correctness, device, subdir, m, s, e, subdir, cor_tag)
    # print("Data path:", data_path)
    data_file = np.loadtxt(data_path, delimiter=',')
    data_file = data_file.reshape(data_file.shape[0], -1, 3) # from (frames, 66) to (frames, 22, 3)
    # print("data_file shape: ", data_file.shape)
    # print(data_file[0])

    if input_type == 'raw':
        data_file = preprocess_raw_joints(data_file, downsample_rate)

    save_dir = 'dataset/{}/{}/{}/{}/{}'.format(dataname, correctness, device, subdir, input_type)
    print("Save directory: ", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save_path = subdir + '_' + correctness + '_m{:02d}_s{:02d}_e{:02d}.npy'.format(m, s, e)
    save_path = 'dataset/{}/{}/{}/{}/{}/m{:02d}_s{:02d}_e{:02d}_{}{}_dr{:02d}'.format(dataname, correctness, device, subdir, input_type, m, s, e, subdir, cor_tag, downsample_rate)
    print("Save path: ", save_path)
    np.save(save_path, data_file)

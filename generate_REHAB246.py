import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Cython.Compiler.Future import annotations
from tqdm import tqdm

from utils.features_REHAB246 import ex01, plot_features, ex02, ex03, ex04, ex05, ex06

joint_names = {'Hips': 0,
               'Spine': 1,
               'Spine1': 2,
               'Neck': 3,
               'Head': 4,
               'Head_end': 5,
               'LeftShoulder': 6,
               'LeftArm': 7,
               'LeftForeArm': 8,
               'LeftHand': 9,
               'LeftHand_end': 10,
               'RightShoulder': 11,
               'RightArm': 12,
               'RightForeArm': 13,
               'RightHand': 14,
               'RightHand_end': 15,
               'LeftUpLeg': 16,
               'LeftLeg': 17,
               'LeftFoot': 18,
               'eftToeBase': 19,
               'LeftToeBase_end': 20,
               'RightUpLeg': 21,
               'RightLeg': 22,
               'RightFoot': 23,
               'RightToeBase': 24,
               'RightToeBase_end': 25
               }

root = './dataset/REHAB24-6/2d_joints_segmented'
annotations = pd.read_csv(os.path.join(root, 'annotations.csv'))

function_map = {
    'ex01': ex01,
    'ex02': ex02,
    'ex03': ex03,
    'ex04': ex04,
    'ex05': ex05,
    'ex06': ex06
}

total_iterations = len(annotations['file_name'].unique())
with tqdm(total=total_iterations, desc="Processing all files") as pbar:
    for file in annotations['file_name'].unique():
        exercise_id = annotations[annotations['file_name'] == file]['exercise_id'].values[0]
        Ex = f'Ex{exercise_id}-segmented'
        # file = 'PM_029_c18_unknown-3-rep6-0.npy'

        side = annotations[annotations['file_name'] == file]['side'].values[0]
        joints = np.load(os.path.join(root, Ex, file))

        features, feature_names = function_map.get(f'ex{exercise_id:02}', lambda x: "Function not found")(joints)
        # print(features.shape, feature_names)

        # save features as csv in respective exercise folders
        save_path = os.path.join(root, 'features', Ex)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        features_df = pd.DataFrame(features, columns=feature_names)
        # limit decimal places to 3
        features_df = features_df.round(3)
        features_df.to_csv(os.path.join(save_path, f"{file.split('.')[0]}_features.csv"), index=False)

        pbar.update(1)  # Increment the progress bar


# features, feature_names = ex06(joints)
# print(features.shape, feature_names)
# plot_features(features, feature_names, feature, side)




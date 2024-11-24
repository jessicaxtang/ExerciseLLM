'''
Run this script to generate UI-PRMD dataset for input to LLM. 
Generates: 
    - set of absolute coordinates
    - exercise-specific features (modify utils/features.py)
'''

import os
import pandas as pd
import numpy as np
from utils import features
from utils.skel_conversions import relative_absolute, get_exercise_side
from utils.features import m01, m02, m03, m04, m05, m06, m07, m08, m09, m10
import pickle


if __name__ == '__main__':

    function_map = {
        'm01': m01,
        'm02': m02,
        'm03': m03,
        'm04': m04,
        'm05': m05,
        'm06': m06,
        'm07': m07,
        'm08': m08,
        'm09': m09,
        'm10': m10,
    }

    correct = True

    root = './dataset/UI-PRMD/'
    source = os.path.join(root, 'correct' if correct else 'incorrect', 'kinect')

    sides_df = pd.read_csv('left-right.csv')

    # Initialize lists to collect data
    positions = []
    angles = []

    for i in range(1, 11):
        for j in range(1, 11):
            for k in range(1, 11):
                filename = f"m{i:02}_s{j:02}_e{k:02}_"

                suffix = '' if correct else '_inc'
                position = np.loadtxt(os.path.join(source, 'positions', f"{filename}positions{suffix}.txt"), delimiter=',')
                angle = np.loadtxt(os.path.join(source, 'angles', f"{filename}angles{suffix}.txt"), delimiter=',')

                position = position.T.reshape(22, 3, -1)
                angle = angle.T.reshape(22, 3, -1)
                absolute = relative_absolute(position, angle)
                absolute = np.transpose(absolute, (2, 0, 1))
                side = get_exercise_side(sides_df, f"s{i:02}", f"m{j:02}")

                # features, feature_names = m02(absolute, side)
                features, feature_names = function_map.get(f"m{i:02}", lambda x: "Function not found")(absolute, side)

                # print(filename, absolute.shape, side, features.shape)

                # save features folder and save data
                features_path = os.path.join(source, 'features')
                if not os.path.exists(features_path):
                    os.makedirs(features_path)
                with open(os.path.join(source, 'features', f"{filename}features{suffix}.pkl"), 'wb') as file:
                    pickle.dump((features, feature_names), file)

                # save absolutes folder and save data
                absolutes_path = os.path.join(source, 'absolutes')
                if not os.path.exists(absolutes_path):
                    os.makedirs(absolutes_path)
                with open(os.path.join(source, 'absolutes', f"{filename}absolutes{suffix}.pkl"), 'wb') as file:
                    pickle.dump(absolute, file)
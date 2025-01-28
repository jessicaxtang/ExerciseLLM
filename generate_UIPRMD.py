import os
import pandas as pd
import numpy as np
from utils.skel_conversions import relative_absolute, get_exercise_side
from utils.features_UIPRMD import m01, m02, m03, m04, m05, m06, m07, m08, m09, m10
from tqdm import tqdm

'''
UI-PRMD JOINT MAPPINGS (0 INDEXED):
0: Waist
1: Spine
2: Chest
3: Neck
4: Head
5: Head tip
6: Left collar
7: Left upper arm
8: Left forearm
9: Left hand
10: Right collar
11: Right upper arm
12: Right forearm
13: Right hand
14: Left upper leg
15: Left lower leg
16: Left foot
17: Left leg toes
18: Right upper leg
19: Right lower leg
20: Right foot
21: Right leg toes
'''

# Define the mapping of specific joints and features to movements
specific_joints_features = {
    'm01': { # Deep Squat
        'joints': [0, 1, 2, 14, 15, 16, 18, 19, 20], 
        'features': [1, 4, 2]  # Hip Flexion Angle, Depth of Squat, Knee Valgus Angle
    },
    'm02': { # Hurdle Step
        'joints': [0, 1, 2, 14, 15, 16, 18, 19, 20], 
        'features': [0, 1, 5]  # Trunk Inclination Angle, Hip Flexion Angle, Leg Height
    },
    'm03': { # Inline Lunge
        'joints': [0, 1, 2, 14, 15, 16, 18, 19, 20], 
        'features': [4, 0, 9]  # Forward Knee Flexion Angle, Trunk Inclination Angle, Foot Distance
    },
    'm04': { # Side Lunge
        'joints': [0, 1, 2, 14, 15, 16, 18, 19, 20], 
        'features': [0, 3, 9]  # Knee Valgus Angle, Thigh Angle, Pelvic Stability  # Same as left
    },
    'm05': { # Sit to Stand
        'joints': [0, 1, 2, 14, 15, 16, 18, 19, 20], 
        'features': [0, 12, 2]  # Trunk Inclination Angle, Hip Flexion Angle, Pelvic Stability
    },
    'm06': { # Active Straight Leg Raise
        'joints': [0, 1, 2, 14, 15, 16, 18, 19, 20], 
        'features': [2, 3, 0]  # Hip Flexion Angle, Leg Elevation Angle, Pelvic Stability
    },
    'm07': { # Shoulder Abduction
        'joints': [0, 1, 3, 6, 7, 8, 9, 10, 11], 
        'features': [1, 0, 2]  # Arm Abduction Angle, Trunk Inclination Angle, Arm Plane Deviation
    },
    'm08': { # Shoulder Extension
        'joints': [0, 1, 2, 3, 6, 7, 8, 9, 10, 11], 
        'features': [3, 1, 0]  # Shoulder Extension Angle, Head Neutral Position, Trunk Inclination Angle
    },
    'm09': { # Shoulder Internal-External Rotation
        'joints': [0, 1, 2, 3, 6, 7, 8, 10, 11, 12], 
        'features': [2, 3, 4]  # Arm Internal Rotation Angle, Arm External Rotation Angle, Elbow Flexion Angle
    },
    'm10': { # Shoulder Scaption
        'joints': [0, 1, 2, 3, 6, 7, 8, 9, 10, 11], 
        'features': [3, 0, 2]  # Arm Elevation Angle, Trunk Inclination Angle, Arm Plane Deviation
    }
}

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

    root = './dataset/UI-PRMD/'
    sides_df = pd.read_csv('left-right.csv')

    total_iterations = 2 * 10 * 10 * 10 # Correct/Incorrect, Movements, Subjects, Episodes
    with tqdm(total=total_iterations, desc="Processing all experiments") as pbar:
        for correct in [True, False]:
            print(f"Processing {'correct' if correct else 'incorrect'} data")
            source = os.path.join(root, 'correct' if correct else 'incorrect', 'kinect')
            
            for i in range(1, 11):  # Movements
                movement_id = f"m{i:02}"
                movement_config = specific_joints_features.get(movement_id, None)

                joints_to_select = movement_config['joints']
                features_to_select = movement_config['features']

                for j in range(1, 11):  # Subjects
                    for k in range(1, 11):  # Episodes
                        filename = f"{movement_id}_s{j:02}_e{k:02}_"
                        suffix = '' if correct else '_inc'

                        # Load data
                        position = np.loadtxt(os.path.join(source, 'positions', f"{filename}positions{suffix}.txt"), delimiter=',')
                        angle = np.loadtxt(os.path.join(source, 'angles', f"{filename}angles{suffix}.txt"), delimiter=',')

                        # Reshape data
                        position = position.T.reshape(22, 3, -1)
                        angle = angle.T.reshape(22, 3, -1)
                        absolute = relative_absolute(position, angle)
                        absolute = np.transpose(absolute, (2, 0, 1))
                        side = get_exercise_side(sides_df, f"s{i:02}", f"m{j:02}")

                        # Process features using the movement-specific function
                        all_features, all_feature_names = function_map.get(movement_id, lambda x: "Function not found")(absolute, side)

                        # Select specific features
                        selected_features = all_features[:, features_to_select]
                        selected_feature_names = [all_feature_names[idx] for idx in features_to_select]

                        # Select specific joints
                        selected_absolutes = absolute[:, joints_to_select, :]
                        selected_absolutes = selected_absolutes.reshape(selected_absolutes.shape[0], -1)
                        selected_joint_names = [
                            f"joint_{joint_idx}_{dim}" for joint_idx in joints_to_select for dim in ['x', 'y', 'z']
                        ]

                        # print(filename, selected_absolutes.shape, side, selected_features.shape)

                        # Save selected features as CSV
                        features_path = os.path.join(source, 'features')
                        os.makedirs(features_path, exist_ok=True)
                        features_df = pd.DataFrame(selected_features, columns=selected_feature_names)
                        features_df.to_csv(os.path.join(features_path, f"{filename}features{suffix}.csv"), index=False)

                        # Save selected absolutes as CSV
                        absolutes_path = os.path.join(source, 'absolutes')
                        os.makedirs(absolutes_path, exist_ok=True)
                        absolute_df = pd.DataFrame(selected_absolutes, columns=selected_joint_names)
                        absolute_df.to_csv(os.path.join(absolutes_path, f"{filename}absolutes{suffix}.csv"), index=False)
                        
                        pbar.update(1)  # Increment the progress bar
    
    print("All experiments processed successfully.")
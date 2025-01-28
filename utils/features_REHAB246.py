import numpy as np
import matplotlib.pyplot as plt


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
               'RightToeBase_end': 25}


def plot_features(features, feature_names, feature, side):
    # Create a 3 x 5 grid for the plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(feature + ' (' + side + ')', fontsize=16)

    # Plot each feature in a subplot
    for i, ax in enumerate(axes.flatten()):
        if i < features.shape[1]:  # Check if there are enough features for each subplot
            ax.plot(features[:, i])
            ax.set_title(feature_names[i], fontsize=10)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Value")
        else:
            # Hide any unused subplots
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to include the main title
    plt.show()


def calculate_angle(a, b, c):
    """Calculate the angle ABC (in degrees) given three points a, b, and c."""
    ab = a - b
    bc = c - b
    cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.round(np.degrees(angle), 4)


def ex01(frames_tensor, side='left'):
    """
    Feature extraction for Ex1: Arm Abduction (sideway raising of the straightened arm).

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 2) with (x, y) positions.
        side (str): 'left' or 'right' to specify the arm performing the abduction.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    # Map side to joint indices
    if side.__contains__('left'):
        shoulder = joint_names['LeftShoulder']
        elbow = joint_names['LeftArm']
        hand = joint_names['LeftHand']
        neck = joint_names['Neck']
        spine = joint_names['Spine']
    elif side.__contains__('right'):
        shoulder = joint_names['RightShoulder']
        elbow = joint_names['RightArm']
        hand = joint_names['RightHand']
        neck = joint_names['Neck']
        spine = joint_names['Spine']
    else:
        raise ValueError("Side must contain 'left' or 'right'")

    # Define feature names
    feature_names = [
        f"{side.capitalize()} Arm Elevation Angle",  # 0
        "Trunk Inclination Angle",                  # 1
        f"{side.capitalize()} Elbow Angle",     # 2
        # f"{side.capitalize()} Shoulder Stability",  # 3
        f"{side.capitalize()} Arm Plane Deviation"  # 4
    ]

    # Prepare list to store features for each frame
    features = []

    for frame in frames_tensor:
        # Extract key joint positions
        shoulder_joint = frame[shoulder]
        elbow_joint = frame[elbow]
        hand_joint = frame[hand]
        neck_joint = frame[neck]
        spine_joint = frame[spine]

        # Feature calculations

        # 0. Arm Elevation Angle
        # Angle between the arm and the torso (Neck -> Shoulder -> Elbow).
        arm_elevation_angle = calculate_angle(neck_joint, shoulder_joint, elbow_joint)

        # 1. Trunk Inclination Angle
        # Measures the angle of the trunk (Spine -> Neck) relative to vertical.
        trunk_inclination_angle = calculate_angle(spine_joint, neck_joint, np.array([spine_joint[0], spine_joint[1] + 1]))

        # 2. Elbow Stability
        # Angle at the elbow joint to ensure the arm remains straight (Shoulder -> Elbow -> Hand).
        elbow_angle = calculate_angle(shoulder_joint, elbow_joint, hand_joint)

        # 3. Shoulder Stability
        # Horizontal stability of the shoulder relative to the neck.
        shoulder_stability = np.abs(shoulder_joint[0] - neck_joint[0])

        # 4. Arm Plane Deviation
        # Deviation of the arm from the correct plane of motion.
        arm_plane_deviation = calculate_angle(spine_joint, shoulder_joint, elbow_joint)

        # Append features for the current frame
        features.append([
            arm_elevation_angle,
            trunk_inclination_angle,
            elbow_angle,
            # shoulder_stability,
            arm_plane_deviation
        ])

    # Convert features to a numpy array
    return np.array(features), feature_names


def ex02(frames_tensor):
    """
    Feature extraction for Ex2: Arm VW (transition between V-shape and W-shape arms).

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 2) with (x, y) positions.
        joint_names (dict): Dictionary mapping joint names to indices.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    # Dynamically retrieve indices for relevant joints
    left_shoulder = joint_names['LeftShoulder']
    right_shoulder = joint_names['RightShoulder']
    left_elbow = joint_names['LeftArm']
    right_elbow = joint_names['RightArm']
    left_hand = joint_names['LeftHand']
    right_hand = joint_names['RightHand']
    neck = joint_names['Neck']
    spine = joint_names['Spine']

    # Define feature names
    feature_names = [
        "V-Shape Angle (Shoulder)",            # 0
        "W-Shape Angle (Elbow)",               # 1
        # "Arm Elevation Consistency",           # 2
        "Trunk to Vertical Angle",                     # 3
        # "Elbow Position Consistency"           # 4
    ]

    # Prepare list to store features for each frame
    features = []

    for i, frame in enumerate(frames_tensor):
        # Extract key joint positions dynamically
        left_shoulder_joint = frame[left_shoulder]
        right_shoulder_joint = frame[right_shoulder]
        left_elbow_joint = frame[left_elbow]
        right_elbow_joint = frame[right_elbow]
        left_hand_joint = frame[left_hand]
        right_hand_joint = frame[right_hand]
        neck_joint = frame[neck]
        spine_joint = frame[spine]

        # Feature calculations

        # 0. V-Shape Angle (Shoulder)
        # Angle between Neck -> Shoulder -> Hand for both arms.
        left_v_angle = calculate_angle(neck_joint, left_shoulder_joint, left_hand_joint)
        right_v_angle = calculate_angle(neck_joint, right_shoulder_joint, right_hand_joint)
        v_shape_angle = (left_v_angle + right_v_angle) / 2  # Average for both arms

        # 1. W-Shape Angle (Elbow)
        # Angle at the elbow joint for both arms (Shoulder -> Elbow -> Hand).
        left_w_angle = calculate_angle(left_shoulder_joint, left_elbow_joint, left_hand_joint)
        right_w_angle = calculate_angle(right_shoulder_joint, right_elbow_joint, right_hand_joint)
        w_shape_angle = (left_w_angle + right_w_angle) / 2  # Average for both arms

        # 2. Arm Elevation Consistency
        # Vertical distance of hands relative to shoulders.
        left_arm_elevation = left_hand_joint[1] - left_shoulder_joint[1]
        right_arm_elevation = right_hand_joint[1] - right_shoulder_joint[1]
        arm_elevation_consistency = np.abs(left_arm_elevation - right_arm_elevation)

        # 3. Trunk Stability
        # Measures the angle of the trunk (Spine -> Neck) relative to vertical.
        trunk_stability = calculate_angle(spine_joint, neck_joint, np.array([spine_joint[0], spine_joint[1] + 1]))

        # 4. Elbow Position Consistency
        # Vertical distance of elbows relative to shoulders for symmetry.
        left_elbow_position = left_elbow_joint[1] - left_shoulder_joint[1]
        right_elbow_position = right_elbow_joint[1] - right_shoulder_joint[1]
        elbow_position_consistency = np.abs(left_elbow_position - right_elbow_position)

        # Append features for the current frame
        features.append([
            v_shape_angle,
            w_shape_angle,
            # arm_elevation_consistency,
            trunk_stability,
            # elbow_position_consistency
        ])

    # Convert features to a numpy array
    return np.array(features), feature_names


def ex03(frames_tensor):
    """
    Feature extraction for Ex3: Push-ups (hands on a table).

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 2) with (x, y) positions.
        joint_names (dict): Dictionary mapping joint names to indices.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    # Dynamically retrieve indices for relevant joints
    left_hand = joint_names['LeftHand']
    right_hand = joint_names['RightHand']
    left_elbow = joint_names['LeftArm']
    right_elbow = joint_names['RightArm']
    left_shoulder = joint_names['LeftShoulder']
    right_shoulder = joint_names['RightShoulder']
    hips = joint_names['Hips']
    spine = joint_names['Spine']

    # Define feature names
    feature_names = [
        "Elbow Flexion Angle",                # 0
        "Trunk Inclination Angle",            # 1
        "Hand Symmetry",                      # 2
        "Pelvic Stability",                   # 3
        # "Shoulder Stability"                  # 4
    ]

    # Prepare list to store features for each frame
    features = []

    for frame in frames_tensor:
        # Extract key joint positions dynamically
        left_hand_joint = frame[left_hand]
        right_hand_joint = frame[right_hand]
        left_elbow_joint = frame[left_elbow]
        right_elbow_joint = frame[right_elbow]
        left_shoulder_joint = frame[left_shoulder]
        right_shoulder_joint = frame[right_shoulder]
        hips_joint = frame[hips]
        spine_joint = frame[spine]

        # Feature calculations

        # 0. Elbow Flexion Angle
        # Angle at both elbows (Shoulder -> Elbow -> Hand).
        left_elbow_angle = calculate_angle(left_shoulder_joint, left_elbow_joint, left_hand_joint)
        right_elbow_angle = calculate_angle(right_shoulder_joint, right_elbow_joint, right_hand_joint)
        elbow_flexion_angle = (left_elbow_angle + right_elbow_angle) / 2  # Average for both elbows

        # 1. Trunk Inclination Angle
        # Measures the alignment of the trunk (Hips -> Spine) relative to horizontal.
        trunk_inclination_angle = calculate_angle(hips_joint, spine_joint, np.array([hips_joint[0] + 1, hips_joint[1]]))

        # 2. Hand Symmetry
        # Horizontal alignment of both hands.
        # lower values indicate better symmetry
        hand_symmetry = np.abs(left_hand_joint[0] - right_hand_joint[0])

        # 3. Pelvic Stability
        # Horizontal alignment of the hips.
        pelvic_stability = np.abs(left_shoulder_joint[0] - right_shoulder_joint[0])

        # 4. Shoulder Stability
        # Horizontal alignment of both shoulders.
        shoulder_stability = np.abs(left_shoulder_joint[0] - right_shoulder_joint[0])

        # Append features for the current frame
        features.append([
            elbow_flexion_angle,
            trunk_inclination_angle,
            hand_symmetry,
            pelvic_stability,
            # shoulder_stability
        ])

    # Convert features to a numpy array
    return np.array(features), feature_names


def ex04(frames_tensor, side='left'):
    """
    Feature extraction for Ex4: Leg Abduction (sideway raising of the straightened leg).

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 2) with (x, y) positions.
        joint_names (dict): Dictionary mapping joint names to indices.
        side (str): 'left' or 'right' to specify the leg performing the abduction.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    # Map side to joint indices
    if side.__contains__('left'):
        hip = joint_names['LeftUpLeg']
        knee = joint_names['LeftLeg']
        foot = joint_names['LeftFoot']
        pelvis = joint_names['Hips']
        spine = joint_names['Spine']
    elif side.__contains__('right'):
        hip = joint_names['RightUpLeg']
        knee = joint_names['RightLeg']
        foot = joint_names['RightFoot']
        pelvis = joint_names['Hips']
        spine = joint_names['Spine']
    else:
        raise ValueError("Side must be 'left' or 'right'")

    # Define feature names
    feature_names = [
        f"{side.capitalize()} Leg Elevation Angle",   # 0
        "Trunk Angle",                            # 1
        f"{side.capitalize()} Pelvic Tilt Angle",     # 2
        f"{side.capitalize()} Knee Angle",        # 3
        f"{side.capitalize()} Leg Plane Deviation"    # 4
    ]

    # Prepare list to store features for each frame
    features = []

    for frame in frames_tensor:
        # Extract key joint positions dynamically
        hip_joint = frame[hip]
        knee_joint = frame[knee]
        foot_joint = frame[foot]
        pelvis_joint = frame[pelvis]
        spine_joint = frame[spine]

        # Feature calculations

        # 0. Leg Elevation Angle
        # Angle between Pelvis -> Hip -> Knee to evaluate leg elevation.
        leg_elevation_angle = calculate_angle(pelvis_joint, hip_joint, knee_joint)

        # 1. Trunk Stability
        # Measures the alignment of the trunk (Pelvis -> Spine) relative to vertical.
        trunk_angle = calculate_angle(pelvis_joint, spine_joint, np.array([pelvis_joint[0], pelvis_joint[1] + 1]))

        # 2. Pelvic Tilt Angle
        # Measures the tilt of the pelvis during the movement.
        pelvic_tilt_angle = calculate_angle(spine_joint, pelvis_joint, hip_joint)

        # 3. Knee Stability
        # Ensures the knee remains straight (Hip -> Knee -> Foot).
        knee_angle = calculate_angle(hip_joint, knee_joint, foot_joint)

        # 4. Leg Plane Deviation
        # Measures if the leg remains in the correct plane (Pelvis -> Hip -> Knee).
        leg_plane_deviation = calculate_angle(spine_joint, hip_joint, knee_joint)

        # Append features for the current frame
        features.append([
            leg_elevation_angle,
            trunk_angle,
            pelvic_tilt_angle,
            knee_angle,
            leg_plane_deviation
        ])

    # Convert features to a numpy array
    return np.array(features), feature_names


def ex05(frames_tensor, side='left'):
    """
    Feature extraction for Ex5: Leg Lunge.

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 2) with (x, y) positions.
        joint_names (dict): Dictionary mapping joint names to indices.
        side (str): 'left' or 'right' to specify the front leg in the lunge.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    # Map side to joint indices for front and back leg
    if side.__contains__('left'):
        front_hip = joint_names['LeftUpLeg']
        front_knee = joint_names['LeftLeg']
        front_foot = joint_names['LeftFoot']
        back_hip = joint_names['RightUpLeg']
        back_knee = joint_names['RightLeg']
        back_foot = joint_names['RightFoot']
        pelvis = joint_names['Hips']
        spine = joint_names['Spine']
    elif side.__contains__('right'):
        front_hip = joint_names['RightUpLeg']
        front_knee = joint_names['RightLeg']
        front_foot = joint_names['RightFoot']
        back_hip = joint_names['LeftUpLeg']
        back_knee = joint_names['LeftLeg']
        back_foot = joint_names['LeftFoot']
        pelvis = joint_names['Hips']
        spine = joint_names['Spine']
    else:
        raise ValueError("Side must be 'left' or 'right'")

    # Define feature names
    feature_names = [
        f"Front Knee Angle ({side.capitalize()})",    # 0
        f"Back Knee Angle ({side.capitalize()})",     # 1
        "Trunk Angle",                            # 2
        "Lateral Pelvic Movement",                    # 3
        "Foot Symmetry"                               # 4
    ]

    # Prepare list to store features for each frame
    features = []

    for frame in frames_tensor:
        # Extract key joint positions dynamically
        front_hip_joint = frame[front_hip]
        front_knee_joint = frame[front_knee]
        front_foot_joint = frame[front_foot]
        back_hip_joint = frame[back_hip]
        back_knee_joint = frame[back_knee]
        back_foot_joint = frame[back_foot]
        pelvis_joint = frame[pelvis]
        spine_joint = frame[spine]

        # Feature calculations

        # 0. Front Knee Angle
        # Angle at the front knee joint (Hip -> Knee -> Foot).
        front_knee_angle = calculate_angle(front_hip_joint, front_knee_joint, front_foot_joint)

        # 1. Back Knee Angle
        # Angle at the back knee joint (Hip -> Knee -> Foot).
        back_knee_angle = calculate_angle(back_hip_joint, back_knee_joint, back_foot_joint)

        # 2. Trunk Stability
        # Measures the alignment of the trunk (Pelvis -> Spine) relative to vertical.
        trunk_angle = calculate_angle(pelvis_joint, spine_joint, np.array([pelvis_joint[0], pelvis_joint[1] + 1]))

        # 3. Pelvic Stability
        # Measures lateral movement of the pelvis relative to the spine.
        lateral_pelvic_mvt = np.abs(pelvis_joint[0] - spine_joint[0])

        # 4. Foot Symmetry
        # Measures the horizontal distance between the front and back feet.
        foot_symmetry = np.abs(front_foot_joint[0] - back_foot_joint[0])

        # Append features for the current frame
        features.append([
            front_knee_angle,
            back_knee_angle,
            trunk_angle,
            lateral_pelvic_mvt,
            foot_symmetry
        ])

    # Convert features to a numpy array
    return np.array(features), feature_names


def ex06(frames_tensor):
    """
    Feature extraction for Ex6: Squats.

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 2) with (x, y) positions.
        joint_names (dict): Dictionary mapping joint names to indices.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    # Map joint indices for knees, hips, and spine
    left_knee = joint_names['LeftLeg']
    right_knee = joint_names['RightLeg']
    left_hip = joint_names['LeftUpLeg']
    right_hip = joint_names['RightUpLeg']
    pelvis = joint_names['Hips']
    spine = joint_names['Spine']
    left_foot = joint_names['LeftFoot']
    right_foot = joint_names['RightFoot']

    # Define feature names
    feature_names = [
        "Knee Flexion Angle (Average)",        # 0
        "Hip Flexion Angle",                  # 1
        "Trunk Angle",                    # 2
        "Lateral Pelvic Movement",                   # 3
        "Foot Symmetry"                       # 4
    ]

    # Prepare list to store features for each frame
    features = []

    for frame in frames_tensor:
        # Extract key joint positions dynamically
        left_knee_joint = frame[left_knee]
        right_knee_joint = frame[right_knee]
        left_hip_joint = frame[left_hip]
        right_hip_joint = frame[right_hip]
        pelvis_joint = frame[pelvis]
        spine_joint = frame[spine]
        left_foot_joint = frame[left_foot]
        right_foot_joint = frame[right_foot]

        # Feature calculations

        # 0. Knee Flexion Angle (Average)
        # Angle at both knees (Hip -> Knee -> Foot).
        left_knee_angle = calculate_angle(left_hip_joint, left_knee_joint, left_foot_joint)
        right_knee_angle = calculate_angle(right_hip_joint, right_knee_joint, right_foot_joint)
        knee_flexion_angle = (left_knee_angle + right_knee_angle) / 2  # Average for both knees

        # 1. Hip Flexion Angle
        # Angle between Pelvis -> Hip -> Knee (for both hips, average).
        left_hip_angle = calculate_angle(pelvis_joint, left_hip_joint, left_knee_joint)
        right_hip_angle = calculate_angle(pelvis_joint, right_hip_joint, right_knee_joint)
        hip_flexion_angle = (left_hip_angle + right_hip_angle) / 2  # Average for both hips

        # 2. Trunk Stability
        # Measures the alignment of the trunk (Pelvis -> Spine) relative to vertical.
        trunk_angle = calculate_angle(pelvis_joint, spine_joint, np.array([pelvis_joint[0], pelvis_joint[1] + 1]))

        # 3. Pelvic Stability
        # Measures lateral movement of the pelvis relative to the spine.
        lateral_pelvic_mvt = np.abs(pelvis_joint[0] - spine_joint[0])

        # 4. Foot Symmetry
        # Measures the horizontal distance between the left and right feet.
        foot_symmetry = np.abs(left_foot_joint[0] - right_foot_joint[0])

        # Append features for the current frame
        features.append([
            knee_flexion_angle,
            hip_flexion_angle,
            trunk_angle,
            lateral_pelvic_mvt,
            foot_symmetry
        ])

    # Convert features to a numpy array
    return np.array(features), feature_names
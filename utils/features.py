import numpy as np
import matplotlib.pyplot as plt


def plot_features(features, feature_names, feature, side):
    # Create a 3x5 grid for the plots
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
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
    return np.degrees(angle)


def m01(frames_tensor, side='left'):
    """
    Feature extractor for m01: Deep Squat exercise.
    Extracts key features to assess exercise quality based on non-optimal movement criteria and additional features.

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 3) with (x, y, z) positions.
        side (str): 'left' or 'right' to specify the leg side for balance and symmetry checks.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    features = []

    # Define the feature names, starting with non-optimal indicators, followed by general assessment features
    feature_names = [
        "Trunk Inclination Angle",  # 0 - Non-optimal: Measures upright trunk posture
        "Hip Flexion Angle",  # 1 - Non-optimal: Assess squat depth
        "Knee Valgus Angle",  # 2 - Non-optimal: Detect knee collapse inward
        "Trunk Flexion",  # 3 - Non-optimal: Forward trunk flexion, should be under 30°
        "Depth of Squat",  # 4 - Non-optimal: Squat depth assessment
        "Pelvic Tilt Angle",  # 5 - Indicates tilt of the pelvis, useful for balance assessment
        "Spine Flexion Angle",  # 6 - Measures the bend in the spine, indicating posture quality
        "Symmetry in Hip Position",  # 7 - Checks horizontal symmetry of hips (Left Hip vs. Right Hip)
        "Symmetry in Knee Position",  # 8 - Checks horizontal symmetry of knees (Left Knee vs. Right Knee)
        "Ankle Dorsiflexion Angle",  # 9 - Measures the ankle bend, important for deep squat flexibility
        "Foot Position X",  # 10 - X-coordinate of the foot for lateral balance tracking
        "Foot Position Y",  # 11 - Y-coordinate of the foot for vertical stability tracking
        "Foot Position Z",  # 12 - Z-coordinate of the foot for forward/backward tracking
        "Knee Height",  # 13 - Y-axis height of the knee, used to assess squat depth
        "Hip Height"  # 14 - Y-axis height of the hip, lower values indicate deeper squat
    ]

    # Define indices based on side for symmetry and side-specific checks
    if side == 'left':
        hip = 14
        knee = 15
        ankle = 16
        opposite_hip = 18
        opposite_knee = 19
        opposite_ankle = 20
    elif side == 'right':
        hip = 18
        knee = 19
        ankle = 20
        opposite_hip = 14
        opposite_knee = 15
        opposite_ankle = 16
    else:
        raise ValueError("Side must be 'left' or 'right'")

    for i, frame in enumerate(frames_tensor):
        # Extract key joints
        waist = frame[0]
        spine = frame[1]
        chest = frame[2]
        neck = frame[3]
        hip_joint = frame[hip]
        knee_joint = frame[knee]
        ankle_joint = frame[ankle]
        opposite_hip_joint = frame[opposite_hip]
        opposite_knee_joint = frame[opposite_knee]
        opposite_ankle_joint = frame[opposite_ankle]

        # Feature calculations

        # 0. Trunk Inclination Angle
        # Measures the angle between the Waist -> Neck line and the vertical axis to assess upright posture.
        # Large deviations from 0° indicate poor posture or excessive forward lean.
        trunk_inclination_angle = calculate_angle(waist, neck, np.array([waist[0], waist[1] + 1, waist[2]]))

        # 1. Hip Flexion Angle
        # Measures the angle at the hip joint (Waist, Hip, Knee).
        # Smaller angles indicate a deeper squat, as the hip bends to lower the body.
        hip_flexion_angle = calculate_angle(waist, hip_joint, knee_joint)

        # 2. Knee Valgus Angle
        # Measures the alignment between Hip, Knee, and Ankle.
        # This angle should be close to 180°; a lower angle may indicate valgus (inward collapse).
        knee_valgus_angle = calculate_angle(hip_joint, knee_joint, ankle_joint)

        # 3. Trunk Flexion
        # Measures the forward bending angle of the trunk (Spine, Chest, Neck).
        # Excessive values above 30° indicate forward flexion beyond optimal range.
        trunk_flexion = calculate_angle(spine, chest, neck)

        # 4. Depth of Squat
        # Calculates the Y-axis position of the hip relative to the knee.
        # A negative or small positive value implies a squat past parallel, which is ideal.
        depth_of_squat = hip_joint[1] - knee_joint[1]

        # 5. Pelvic Tilt Angle
        # Measures the tilt of the pelvis based on the line between the two hips (Left Hip, Right Hip).
        # Large deviations can indicate poor pelvic alignment or asymmetry.
        pelvic_tilt_angle = calculate_angle(waist, hip_joint, opposite_hip_joint)

        # 6. Spine Flexion Angle
        # Measures the curvature in the spine by calculating the angle between Spine, Chest, and Neck.
        # Helps in assessing if there is excess curvature or flexion in the spine.
        spine_flexion_angle = calculate_angle(spine, chest, neck)

        # 7. Symmetry in Hip Position
        # Measures the horizontal distance (X-axis) between the two hips.
        # Large discrepancies could indicate asymmetry or imbalance.
        hip_symmetry = np.abs(hip_joint[0] - opposite_hip_joint[0])

        # 8. Symmetry in Knee Position
        # Measures the horizontal distance (X-axis) between the two knees.
        # This checks if the knees are aligned horizontally, which is desirable for stability.
        knee_symmetry = np.abs(knee_joint[0] - opposite_knee_joint[0])

        # 9. Ankle Dorsiflexion Angle
        # Measures the angle between the Knee, Ankle, and Foot.
        # Indicates how much the ankle is flexed. Greater dorsiflexion is generally required in deeper squats.
        ankle_dorsiflexion_angle = calculate_angle(knee_joint, ankle_joint,
                                                   np.array([ankle_joint[0], ankle_joint[1], ankle_joint[2] + 1]))

        # 10-12. Foot Position (X, Y, Z)
        # Tracks the position of the foot in the X, Y, and Z directions to assess overall stability and stance.
        foot_position_x = ankle_joint[0]
        foot_position_y = ankle_joint[1]
        foot_position_z = ankle_joint[2]

        # 13. Knee Height
        # Measures the Y-axis height of the knee. Lower values indicate a deeper squat.
        knee_height = knee_joint[1]

        # 14. Hip Height
        # Measures the Y-axis height of the hip. Lower values correspond to a deeper squat position.
        hip_height = hip_joint[1]

        # Collect all features for this frame
        features.append([
            trunk_inclination_angle,
            hip_flexion_angle,
            knee_valgus_angle,
            trunk_flexion,
            depth_of_squat,
            pelvic_tilt_angle,
            spine_flexion_angle,
            hip_symmetry,
            knee_symmetry,
            ankle_dorsiflexion_angle,
            foot_position_x,
            foot_position_y,
            foot_position_z,
            knee_height,
            hip_height
        ])

    # Convert to NumPy array for easy handling
    return np.array(features), feature_names


def m02(frames_tensor, side='left'):
    """
    Feature extractor for m02: Hurdle Step exercise.
    Extracts key features to assess exercise quality based on non-optimal movement criteria and additional features.

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 3) with (x, y, z) positions.
        side (str): 'left' or 'right' to specify the stepping leg for balance and movement tracking.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    features = []

    # Define the feature names, starting with non-optimal indicators, followed by general assessment features
    feature_names = [
        "Trunk Inclination Angle",  # 0 - Non-optimal: Measures upright trunk posture
        f"{side.capitalize()} Hip Flexion Angle",  # 1 - Non-optimal: Hip flexion to check 89° threshold
        f"{side.capitalize()} Femur Alignment",  # 2 - Non-optimal: Femur neutrality
        "Pelvic Stability",  # 3 - Lateral tilt or movement of pelvis for balance
        f"{side.capitalize()} Knee Flexion Angle",  # 4 - Knee flexion angle
        f"{side.capitalize()} Leg Height",  # 5 - Height of stepping leg relative to hip
        "Spine Stability Angle",  # 6 - Stability of spine during movement
        "Pelvic Rotation Angle",  # 7 - Rotation of pelvis relative to the hip line
        "Foot Position X",  # 8 - X-coordinate of foot for lateral tracking
        "Foot Position Y",  # 9 - Y-coordinate of foot for vertical stability tracking
        "Foot Position Z",  # 10 - Z-coordinate of foot for depth tracking
        f"{side.capitalize()} Ankle Dorsiflexion Angle",  # 11 - Flexion of the ankle for stepping leg
        f"{side.capitalize()} Hip Height",  # 12 - Y-axis height of hip for step analysis
        "Spine Flexion Angle",  # 13 - Indicates overall spine flexion
        "Head Stability"  # 14 - Head stability indicating overall posture
    ]

    # Define indices based on side for stepping leg tracking
    if side == 'left':
        hip = 14
        knee = 15
        ankle = 16
        opposite_hip = 18
        opposite_knee = 19
        opposite_ankle = 20
    elif side == 'right':
        hip = 18
        knee = 19
        ankle = 20
        opposite_hip = 14
        opposite_knee = 15
        opposite_ankle = 16
    else:
        raise ValueError("Side must be 'left' or 'right'")

    for i, frame in enumerate(frames_tensor):
        # Extract key joints
        waist = frame[0]
        spine = frame[1]
        chest = frame[2]
        neck = frame[3]
        hip_joint = frame[hip]
        knee_joint = frame[knee]
        ankle_joint = frame[ankle]
        opposite_hip_joint = frame[opposite_hip]
        opposite_knee_joint = frame[opposite_knee]
        head = frame[5]

        # Feature calculations

        # 0. Trunk Inclination Angle
        # Measures the angle between the Waist -> Neck line and the vertical axis to assess upright posture.
        trunk_inclination_angle = calculate_angle(waist, neck, np.array([waist[0], waist[1] + 1, waist[2]]))

        # 1. Hip Flexion Angle
        # Measures the angle at the hip joint (Waist, Hip, Knee).
        # An angle close to 89° indicates appropriate flexion during the hurdle step.
        hip_flexion_angle = calculate_angle(waist, hip_joint, knee_joint)

        # 2. Femur Alignment
        # Measures alignment of femur with Hip and Knee joints.
        # Neutral femur alignment implies proper positioning.
        femur_alignment = calculate_angle(hip_joint, knee_joint,
                                          np.array([hip_joint[0], hip_joint[1], hip_joint[2] + 1]))

        # 3. Pelvic Stability
        # Calculates the horizontal distance (X-axis) between both hips.
        # Large deviations may indicate poor pelvic stability.
        pelvic_stability = np.abs(hip_joint[0] - opposite_hip_joint[0])

        # 4. Knee Flexion Angle
        # Measures the angle at the knee joint (Hip, Knee, Ankle).
        # Detects if there’s excessive flexion in the stepping leg.
        knee_flexion_angle = calculate_angle(hip_joint, knee_joint, ankle_joint)

        # 5. Leg Height (Stepping Leg)
        # Measures the Y-axis height of the ankle relative to the hip for the stepping leg.
        # Indicates how high the stepping leg is lifted during the hurdle step.
        leg_height = ankle_joint[1] - hip_joint[1]

        # 6. Spine Stability Angle
        # Measures stability by calculating the angle of the spine (Spine, Chest, Neck).
        spine_stability_angle = calculate_angle(spine, chest, neck)

        # 7. Pelvic Rotation Angle
        # Measures rotation of the pelvis based on alignment between the two hips.
        # Indicates if there’s rotational instability during the step.
        pelvic_rotation_angle = calculate_angle(waist, hip_joint, opposite_hip_joint)

        # 8-10. Foot Position (X, Y, Z)
        # Tracks the position of the foot in the X, Y, and Z directions for balance and movement stability.
        foot_position_x = ankle_joint[0]
        foot_position_y = ankle_joint[1]
        foot_position_z = ankle_joint[2]

        # 11. Ankle Dorsiflexion Angle (Stepping Leg)
        # Measures ankle flexion in the stepping leg (Knee, Ankle, Foot).
        # Important for stability in the hurdle step.
        ankle_dorsiflexion_angle = calculate_angle(knee_joint, ankle_joint,
                                                   np.array([ankle_joint[0], ankle_joint[1], ankle_joint[2] + 1]))

        # 12. Hip Height
        # Measures the Y-axis height of the hip joint in the stepping leg.
        # Important for assessing if the hip is raised correctly during the step.
        hip_height = hip_joint[1]

        # 13. Spine Flexion Angle
        # Measures curvature or bending of the spine during the movement.
        spine_flexion_angle = calculate_angle(spine, chest, neck)

        # 14. Head Stability
        # Calculates the stability of the head position by checking alignment (Neck, Head).
        # Important for maintaining overall posture.
        head_stability = calculate_angle(neck, head, np.array([neck[0], neck[1] + 1, neck[2]]))

        # Collect all features for this frame
        features.append([
            trunk_inclination_angle,
            hip_flexion_angle,
            femur_alignment,
            pelvic_stability,
            knee_flexion_angle,
            leg_height,
            spine_stability_angle,
            pelvic_rotation_angle,
            foot_position_x,
            foot_position_y,
            foot_position_z,
            ankle_dorsiflexion_angle,
            hip_height,
            spine_flexion_angle,
            head_stability
        ])

        # Convert to NumPy array for easy handling
        return np.array(features), feature_names


def m03(frames_tensor, side='left'):
    """
    Feature extractor for m03: Inline Lunge exercise.
    Extracts key features to assess exercise quality based on non-optimal movement criteria and additional features.

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 3) with (x, y, z) positions.
        side (str): 'left' or 'right' to specify the rear leg for stability and movement tracking.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    features = []

    # Define the feature names, starting with non-optimal indicators, followed by general assessment features
    feature_names = [
        "Trunk Inclination Angle",           # 0 - Non-optimal: Measures upright trunk posture
        f"{side.capitalize()} Rear Knee Depth",     # 1 - Non-optimal: Vertical position of rear knee
        "Lateral Deviation",                 # 2 - Non-optimal: Sideways deviation of forward step
        "Pelvic Stability",                  # 3 - Lateral tilt or movement of pelvis for balance
        f"{side.capitalize()} Forward Knee Flexion Angle", # 4 - Forward knee flexion angle
        f"{side.capitalize()} Hip Flexion Angle",          # 5 - Hip flexion to check forward lean
        "Spine Alignment",                   # 6 - Alignment of spine for posture check
        "Pelvic Rotation Angle",             # 7 - Rotation of pelvis for stability
        "Head Stability",                    # 8 - Head stability indicating overall balance
        "Foot Distance",                     # 9 - Distance between the two feet for lunge depth
        f"{side.capitalize()} Ankle Dorsiflexion Angle", # 10 - Dorsiflexion at ankle of rear leg
        "Forward Knee Stability",            # 11 - Forward knee position tracking
        "Hip Height",                        # 12 - Y-axis height of hip for depth analysis
        "Spine Flexion Angle",               # 13 - Spine flexibility during movement
        "Pelvic Symmetry"                    # 14 - Symmetry of pelvis position
    ]

    # Define indices based on side for forward/rear leg tracking
    if side == 'left':
        forward_hip = 18
        forward_knee = 19
        forward_ankle = 20
        rear_hip = 14
        rear_knee = 15
        rear_ankle = 16
    elif side == 'right':
        forward_hip = 14
        forward_knee = 15
        forward_ankle = 16
        rear_hip = 18
        rear_knee = 19
        rear_ankle = 20
    else:
        raise ValueError("Side must be 'left' or 'right'")

    for i, frame in enumerate(frames_tensor):
        # Extract key joints
        waist = frame[0]
        spine = frame[1]
        chest = frame[2]
        neck = frame[3]
        forward_hip_joint = frame[forward_hip]
        forward_knee_joint = frame[forward_knee]
        forward_ankle_joint = frame[forward_ankle]
        rear_hip_joint = frame[rear_hip]
        rear_knee_joint = frame[rear_knee]
        rear_ankle_joint = frame[rear_ankle]
        head = frame[5]

        # Feature calculations

        # 0. Trunk Inclination Angle
        # Measures the angle between the Waist -> Neck line and the vertical axis to assess upright posture.
        trunk_inclination_angle = calculate_angle(waist, neck, np.array([waist[0], waist[1] + 1, waist[2]]))

        # 1. Rear Knee Depth
        # Measures the Y-axis height of the rear knee to check if it is too close to the floor.
        rear_knee_depth = rear_knee_joint[1]

        # 2. Lateral Deviation
        # Measures the horizontal (X-axis) deviation of the forward ankle relative to the waist.
        # Large values may indicate lateral instability.
        lateral_deviation = np.abs(forward_ankle_joint[0] - waist[0])

        # 3. Pelvic Stability
        # Calculates the horizontal distance (X-axis) between both hips.
        # Large deviations may indicate poor pelvic stability.
        pelvic_stability = np.abs(forward_hip_joint[0] - rear_hip_joint[0])

        # 4. Forward Knee Flexion Angle
        # Measures the angle at the forward knee (Forward Hip, Forward Knee, Forward Ankle).
        forward_knee_flexion_angle = calculate_angle(forward_hip_joint, forward_knee_joint, forward_ankle_joint)

        # 5. Hip Flexion Angle
        # Measures the angle at the hip joint (Waist, Forward Hip, Forward Knee).
        # Indicates forward lean, important for balance.
        hip_flexion_angle = calculate_angle(waist, forward_hip_joint, forward_knee_joint)

        # 6. Spine Alignment
        # Measures the alignment of the spine (Spine, Chest, Neck) for posture assessment.
        spine_alignment = calculate_angle(spine, chest, neck)

        # 7. Pelvic Rotation Angle
        # Measures rotation of the pelvis based on alignment between the two hips.
        pelvic_rotation_angle = calculate_angle(waist, forward_hip_joint, rear_hip_joint)

        # 8. Head Stability
        # Calculates stability of the head position by checking alignment (Neck, Head).
        # Important for maintaining overall balance.
        head_stability = calculate_angle(neck, head, np.array([neck[0], neck[1] + 1, neck[2]]))

        # 9. Foot Distance
        # Measures the Euclidean distance between the forward and rear feet (Forward Ankle, Rear Ankle).
        # Used to assess the depth of the lunge.
        foot_distance = np.linalg.norm(forward_ankle_joint - rear_ankle_joint)

        # 10. Ankle Dorsiflexion Angle (Rear Leg)
        # Measures dorsiflexion at the ankle in the rear leg (Rear Knee, Rear Ankle, Foot).
        # Important for stability in the inline lunge.
        ankle_dorsiflexion_angle = calculate_angle(rear_knee_joint, rear_ankle_joint, np.array([rear_ankle_joint[0], rear_ankle_joint[1], rear_ankle_joint[2] + 1]))

        # 11. Forward Knee Stability
        # Tracks the X-axis position of the forward knee relative to the forward ankle.
        forward_knee_stability = np.abs(forward_knee_joint[0] - forward_ankle_joint[0])

        # 12. Hip Height
        # Measures the Y-axis height of the forward hip joint.
        hip_height = forward_hip_joint[1]

        # 13. Spine Flexion Angle
        # Measures flexibility or curvature of the spine during movement.
        spine_flexion_angle = calculate_angle(spine, chest, neck)

        # 14. Pelvic Symmetry
        # Checks horizontal alignment between hips for symmetrical movement.
        pelvic_symmetry = np.abs(forward_hip_joint[0] - rear_hip_joint[0])

        # Collect all features for this frame
        features.append([
            trunk_inclination_angle,
            rear_knee_depth,
            lateral_deviation,
            pelvic_stability,
            forward_knee_flexion_angle,
            hip_flexion_angle,
            spine_alignment,
            pelvic_rotation_angle,
            head_stability,
            foot_distance,
            ankle_dorsiflexion_angle,
            forward_knee_stability,
            hip_height,
            spine_flexion_angle,
            pelvic_symmetry
        ])

    # Convert to NumPy array for easy handling
    return np.array(features), feature_names


def m04(frames_tensor, side='left'):
    """
    Feature extractor for m04: Side Lunge exercise.
    Extracts key features to assess exercise quality based on non-optimal movement criteria and additional features.

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 3) with (x, y, z) positions.
        side (str): 'left' or 'right' to specify the lunging leg.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    features = []

    # Define the feature names, starting with non-optimal indicators, followed by general assessment features
    feature_names = [
        f"{side.capitalize()} Knee Valgus Angle",       # 0 - Non-optimal: Measures knee valgus (inward collapse)
        "Pelvic Tilt Angle",                            # 1 - Non-optimal: Pelvic tilt over 5° is undesirable
        "Trunk Inclination Angle",                      # 2 - Non-optimal: Should be below 30°
        f"{side.capitalize()} Thigh Angle",             # 3 - Non-optimal: Should be below 45°
        f"{side.capitalize()} Knee-to-Toe Distance",    # 4 - Non-optimal: Checks if knee goes past toes
        f"{side.capitalize()} Leg Abduction Distance",  # 5 - Lateral distance between feet for lunge spread
        f"{side.capitalize()} Hip Flexion Angle",       # 6 - Measures depth of lunge
        "Trunk Flexion Angle",                          # 7 - Measures spine bending for posture
        "Spine Alignment",                              # 8 - Monitors alignment of spine (Spine, Chest, Neck)
        "Pelvic Stability",                             # 9 - Horizontal movement of the pelvis
        f"{side.capitalize()} Ankle Dorsiflexion Angle",# 10 - Dorsiflexion of lunging leg's ankle
        "Hip Height",                                   # 11 - Y-axis height of hip for depth analysis
        "Head Stability",                               # 12 - Stability of head for overall balance
        f"{side.capitalize()} Knee Flexion Angle",      # 13 - Bending of the lunging leg's knee
        "Pelvic Symmetry"                               # 14 - Symmetry of pelvis position
    ]

    # Define indices based on side for lunging leg
    if side == 'left':
        lunging_hip = 14
        lunging_knee = 15
        lunging_ankle = 16
        opposite_hip = 18
        opposite_knee = 19
        opposite_ankle = 20
    elif side == 'right':
        lunging_hip = 18
        lunging_knee = 19
        lunging_ankle = 20
        opposite_hip = 14
        opposite_knee = 15
        opposite_ankle = 16
    else:
        raise ValueError("Side must be 'left' or 'right'")

    for i, frame in enumerate(frames_tensor):
        # Extract key joints
        waist = frame[0]
        spine = frame[1]
        chest = frame[2]
        neck = frame[3]
        lunging_hip_joint = frame[lunging_hip]
        lunging_knee_joint = frame[lunging_knee]
        lunging_ankle_joint = frame[lunging_ankle]
        opposite_hip_joint = frame[opposite_hip]
        opposite_ankle_joint = frame[opposite_ankle]
        head = frame[5]

        # Feature calculations

        # 0. Knee Valgus Angle
        # Measures the angle between the Hip, Knee, and Ankle joints of the lunging leg.
        # A lower angle than 180° suggests valgus or inward knee collapse.
        knee_valgus_angle = calculate_angle(lunging_hip_joint, lunging_knee_joint, lunging_ankle_joint)

        # 1. Pelvic Tilt Angle
        # Measures the tilt of the pelvis by checking the alignment of the two hips.
        # Large deviations indicate poor pelvic stability.
        pelvic_tilt_angle = calculate_angle(waist, lunging_hip_joint, opposite_hip_joint)

        # 2. Trunk Inclination Angle
        # Measures the forward tilt of the trunk (Neck, Chest, Spine) relative to the vertical.
        # This angle should generally be less than 30°.
        trunk_inclination_angle = calculate_angle(waist, neck, np.array([waist[0], waist[1] + 1, waist[2]]))

        # 3. Thigh Angle
        # Measures the angle between Waist, Hip, and Knee to evaluate the depth of the side lunge.
        # This angle should generally be below 45° to indicate proper lunge depth.
        thigh_angle = calculate_angle(waist, lunging_hip_joint, lunging_knee_joint)

        # 4. Knee-to-Toe Distance
        # Calculates the horizontal distance (X-axis) between the knee and the toes (ankle).
        # If the knee is too far past the toes, it can indicate improper form.
        knee_to_toe_distance = lunging_knee_joint[0] - lunging_ankle_joint[0]

        # 5. Leg Abduction Distance
        # Measures the lateral distance between the ankle joints of both legs to assess the width of the side lunge.
        leg_abduction_distance = np.abs(lunging_ankle_joint[0] - opposite_ankle_joint[0])

        # 6. Hip Flexion Angle
        # Measures the bending at the hip joint (Waist, Hip, Knee) for the lunging leg.
        hip_flexion_angle = calculate_angle(waist, lunging_hip_joint, lunging_knee_joint)

        # 7. Trunk Flexion Angle
        # Measures the bending in the trunk by checking the angle (Spine, Chest, Neck).
        # Helps assess if the trunk is excessively flexed forward.
        trunk_flexion_angle = calculate_angle(spine, chest, neck)

        # 8. Spine Alignment
        # Measures the alignment of the spine for proper posture (Spine, Chest, Neck).
        spine_alignment = calculate_angle(spine, chest, neck)

        # 9. Pelvic Stability
        # Measures horizontal movement of the pelvis by comparing the positions of the two hips.
        # Significant deviations can indicate poor stability.
        pelvic_stability = np.abs(lunging_hip_joint[0] - opposite_hip_joint[0])

        # 10. Ankle Dorsiflexion Angle
        # Measures the dorsiflexion of the ankle in the lunging leg (Knee, Ankle, Foot).
        # Important for flexibility and stability in the side lunge.
        ankle_dorsiflexion_angle = calculate_angle(lunging_knee_joint, lunging_ankle_joint, np.array([lunging_ankle_joint[0], lunging_ankle_joint[1], lunging_ankle_joint[2] + 1]))

        # 11. Hip Height
        # Measures the Y-axis height of the hip joint in the lunging leg.
        # Lower values correspond to a deeper lunge.
        hip_height = lunging_hip_joint[1]

        # 12. Head Stability
        # Checks the stability of the head by measuring its alignment with the neck.
        # Useful for overall balance monitoring.
        head_stability = calculate_angle(neck, head, np.array([neck[0], neck[1] + 1, neck[2]]))

        # 13. Knee Flexion Angle
        # Measures the flexion in the lunging knee joint.
        knee_flexion_angle = calculate_angle(lunging_hip_joint, lunging_knee_joint, lunging_ankle_joint)

        # 14. Pelvic Symmetry
        # Checks for symmetry in the horizontal alignment of both hips.
        pelvic_symmetry = np.abs(lunging_hip_joint[0] - opposite_hip_joint[0])

        # Collect all features for this frame
        features.append([
            knee_valgus_angle,
            pelvic_tilt_angle,
            trunk_inclination_angle,
            thigh_angle,
            knee_to_toe_distance,
            leg_abduction_distance,
            hip_flexion_angle,
            trunk_flexion_angle,
            spine_alignment,
            pelvic_stability,
            ankle_dorsiflexion_angle,
            hip_height,
            head_stability,
            knee_flexion_angle,
            pelvic_symmetry
        ])

    # Convert to NumPy array for easy handling
    return np.array(features), feature_names


def m05(frames_tensor, side='left'):
    """
    Feature extractor for m05: Sit to Stand exercise.
    Extracts key features to assess exercise quality based on non-optimal movement criteria and additional features.

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 3) with (x, y, z) positions.
        side (str): 'left' or 'right' to specify the dominant leg for balance and stability tracking.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    features = []

    # Define the feature names, starting with non-optimal indicators, followed by general assessment features
    feature_names = [
        "Trunk Inclination Angle",  # 0 - Non-optimal: Measures upright trunk posture
        "Pelvic Rise",  # 1 - Non-optimal: Excessive upward movement of pelvis
        "Pelvic Stability",  # 2 - Lateral tilt or movement of pelvis for balance
        "Weight Distribution",  # 3 - Weight shift between legs for balance
        f"{side.capitalize()} Knee Valgus Angle",  # 4 - Non-optimal: Detect knee collapse inward
        f"{side.capitalize()} Knee Flexion Angle",  # 5 - Knee flexion during standing
        f"{side.capitalize()} Ankle Dorsiflexion Angle",  # 6 - Dorsiflexion of ankle during movement
        "Head Stability",  # 7 - Stability of head for overall balance
        "Symmetry in Hip Position",  # 8 - Horizontal symmetry of hips (left vs. right)
        "Symmetry in Knee Position",  # 9 - Horizontal symmetry of knees (left vs. right)
        "Spine Flexion Angle",  # 10 - Forward bending of spine for posture
        "Pelvic Rotation",  # 11 - Rotational stability of pelvis
        f"{side.capitalize()} Hip Flexion Angle",  # 12 - Bending at the hip for sit-to-stand
        f"{side.capitalize()} Leg Extension Distance",  # 13 - Distance between hip and ankle for full extension
        "Foot Position Stability"  # 14 - Stability of foot positions in X, Y, and Z axes
    ]

    # Define indices based on side for dominant and opposite leg tracking
    if side == 'left':
        hip = 14
        knee = 15
        ankle = 16
        opposite_hip = 18
        opposite_knee = 19
        opposite_ankle = 20
    elif side == 'right':
        hip = 18
        knee = 19
        ankle = 20
        opposite_hip = 14
        opposite_knee = 15
        opposite_ankle = 16
    else:
        raise ValueError("Side must be 'left' or 'right'")

    for i, frame in enumerate(frames_tensor):
        # Extract key joints
        waist = frame[0]
        spine = frame[1]
        chest = frame[2]
        neck = frame[3]
        hip_joint = frame[hip]
        knee_joint = frame[knee]
        ankle_joint = frame[ankle]

        # Ensure opposite ankle joint is correctly accessed
        opposite_hip_joint = frame[opposite_hip]
        opposite_knee_joint = frame[opposite_knee]
        opposite_ankle_joint = frame[opposite_ankle]  # Pay attention to handling this joint

        head = frame[5]

        # Feature calculations

        # 0. Trunk Inclination Angle
        trunk_inclination_angle = calculate_angle(waist, neck, np.array([waist[0], waist[1] + 1, waist[2]]))

        # 1. Pelvic Rise
        pelvic_rise = hip_joint[1] - waist[1]

        # 2. Pelvic Stability
        pelvic_stability = np.abs(hip_joint[0] - opposite_hip_joint[0])

        # 3. Weight Distribution
        weight_distribution = np.abs(ankle_joint[1] - opposite_ankle_joint[1])

        # 4. Knee Valgus Angle
        knee_valgus_angle = calculate_angle(hip_joint, knee_joint, ankle_joint)

        # 5. Knee Flexion Angle
        knee_flexion_angle = calculate_angle(hip_joint, knee_joint, ankle_joint)

        # 6. Ankle Dorsiflexion Angle
        ankle_dorsiflexion_angle = calculate_angle(knee_joint, ankle_joint,
                                                   np.array([ankle_joint[0], ankle_joint[1], ankle_joint[2] + 1]))

        # 7. Head Stability
        head_stability = calculate_angle(neck, head, np.array([neck[0], neck[1] + 1, neck[2]]))

        # 8. Symmetry in Hip Position
        hip_symmetry = np.abs(hip_joint[0] - opposite_hip_joint[0])

        # 9. Symmetry in Knee Position
        knee_symmetry = np.abs(knee_joint[0] - opposite_knee_joint[0])

        # 10. Spine Flexion Angle
        spine_flexion_angle = calculate_angle(spine, chest, neck)

        # 11. Pelvic Rotation
        pelvic_rotation = calculate_angle(waist, hip_joint, opposite_hip_joint)

        # 12. Hip Flexion Angle
        hip_flexion_angle = calculate_angle(waist, hip_joint, knee_joint)

        # 13. Leg Extension Distance
        leg_extension_distance = np.linalg.norm(hip_joint - ankle_joint)

        # 14. Foot Position Stability
        # Checks the stability of the foot positions in X, Y, and Z directions
        foot_position_stability = np.linalg.norm(ankle_joint - opposite_ankle_joint)

        # Collect all features for this frame
        features.append([
            trunk_inclination_angle,
            pelvic_rise,
            pelvic_stability,
            weight_distribution,
            knee_valgus_angle,
            knee_flexion_angle,
            ankle_dorsiflexion_angle,
            head_stability,
            hip_symmetry,
            knee_symmetry,
            spine_flexion_angle,
            pelvic_rotation,
            hip_flexion_angle,
            leg_extension_distance,
            foot_position_stability
        ])

    # Convert to NumPy array for easy handling
    return np.array(features), feature_names


def m06(frames_tensor, side='left'):
    """
    Feature extractor for m06: Standing Active Straight Leg Raise exercise.
    Extracts key features to assess exercise quality based on non-optimal movement criteria and additional features.

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 3) with (x, y, z) positions.
        side (str): 'left' or 'right' to specify the raised leg.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    features = []

    # Define the feature names, starting with non-optimal indicators, followed by general assessment features
    feature_names = [
        "Pelvic Stability (Tilt Angle)",              # 0 - Non-optimal: Pelvic tilt over 5° is undesirable
        f"{side.capitalize()} Knee Flexion Angle",    # 1 - Non-optimal: Knee flexion over 6° is undesirable
        f"{side.capitalize()} Hip Flexion Angle",     # 2 - Non-optimal: Hip flexion under 59° is undesirable
        f"{side.capitalize()} Leg Elevation Angle",   # 3 - Elevation angle of raised leg relative to body
        "Trunk Inclination Angle",                    # 4 - Ensures upright trunk posture
        "Spine Flexion Angle",                        # 5 - Checks forward bending of spine
        "Pelvic Rotation Angle",                      # 6 - Measures rotation of the pelvis for stability
        "Head Stability",                             # 7 - Monitors head position for balance
        f"{side.capitalize()} Foot Height",           # 8 - Y-coordinate of raised foot for elevation check
        f"{side.capitalize()} Leg Extension Distance" # 9 - Distance between hip and raised foot
    ]

    # Define indices based on side for raised leg tracking
    if side == 'left':
        hip = 14
        knee = 15
        ankle = 16
        opposite_hip = 18
        opposite_knee = 19
        opposite_ankle = 20
    elif side == 'right':
        hip = 18
        knee = 19
        ankle = 20
        opposite_hip = 14
        opposite_knee = 15
        opposite_ankle = 16
    else:
        raise ValueError("Side must be 'left' or 'right'")

    for i, frame in enumerate(frames_tensor):
        # Extract key joints
        waist = frame[0]
        spine = frame[1]
        chest = frame[2]
        neck = frame[3]
        hip_joint = frame[hip]
        knee_joint = frame[knee]
        ankle_joint = frame[ankle]
        opposite_hip_joint = frame[opposite_hip]
        head = frame[5]

        # Feature calculations

        # 0. Pelvic Stability (Tilt Angle)
        # Measures the tilt of the pelvis by checking the alignment of the two hips.
        # Should be under 5° for stability.
        pelvic_tilt_angle = calculate_angle(waist, hip_joint, opposite_hip_joint)

        # 1. Knee Flexion Angle
        # Measures the bending of the knee. Should be under 6° to keep the leg straight.
        knee_flexion_angle = calculate_angle(hip_joint, knee_joint, ankle_joint)

        # 2. Hip Flexion Angle
        # Measures the angle at the hip joint (Waist, Hip, Knee).
        # Should be at least 59° for adequate leg elevation.
        hip_flexion_angle = calculate_angle(waist, hip_joint, knee_joint)

        # 3. Leg Elevation Angle
        # Measures the angle between the waist, hip, and raised foot.
        # Useful for assessing how high the leg is raised.
        leg_elevation_angle = calculate_angle(waist, hip_joint, ankle_joint)

        # 4. Trunk Inclination Angle
        # Measures the angle of the trunk (Neck, Chest, Spine) relative to the vertical.
        # Ensures the trunk remains upright.
        trunk_inclination_angle = calculate_angle(waist, neck, np.array([waist[0], waist[1] + 1, waist[2]]))

        # 5. Spine Flexion Angle
        # Measures the bending in the spine by checking the angle (Spine, Chest, Neck).
        spine_flexion_angle = calculate_angle(spine, chest, neck)

        # 6. Pelvic Rotation Angle
        # Measures rotation of the pelvis based on alignment between the two hips.
        pelvic_rotation_angle = calculate_angle(waist, hip_joint, opposite_hip_joint)

        # 7. Head Stability
        # Measures the stability of the head by checking its alignment with the neck.
        head_stability = calculate_angle(neck, head, np.array([neck[0], neck[1] + 1, neck[2]]))

        # 8. Foot Height (Raised Leg)
        # Measures the Y-axis height of the raised foot, indicating elevation.
        foot_height = ankle_joint[1]

        # 9. Leg Extension Distance
        # Measures the Euclidean distance between the hip and the raised foot.
        # Useful for assessing the reach of the raised leg.
        leg_extension_distance = np.linalg.norm(hip_joint - ankle_joint)

        # Collect all features for this frame
        features.append([
            pelvic_tilt_angle,
            knee_flexion_angle,
            hip_flexion_angle,
            leg_elevation_angle,
            trunk_inclination_angle,
            spine_flexion_angle,
            pelvic_rotation_angle,
            head_stability,
            foot_height,
            leg_extension_distance
        ])

    # Convert to NumPy array for easy handling
    return np.array(features), feature_names


def m07(frames_tensor, side='left'):
    """
    Feature extractor for m07: Standing Shoulder Abduction exercise.
    Extracts key features to assess exercise quality based on non-optimal movement criteria and additional features.

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 3) with (x, y, z) positions.
        side (str): 'left' or 'right' to specify the raised arm.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    features = []

    # Define the feature names, starting with non-optimal indicators, followed by general assessment features
    feature_names = [
        "Trunk Inclination Angle",                  # 0 - Non-optimal: Measures upright trunk posture
        f"{side.capitalize()} Arm Abduction Angle", # 1 - Non-optimal: Abduction angle should reach at least 160°
        f"{side.capitalize()} Arm Plane Deviation", # 2 - Non-optimal: Arm should remain in the plane of motion
        f"{side.capitalize()} Shoulder Flexion Angle", # 3 - Measures alignment of shoulder
        "Pelvic Stability (Tilt Angle)",            # 4 - Checks stability of the pelvis
        "Spine Flexion Angle",                      # 5 - Monitors forward bending of spine
        "Head Stability",                           # 6 - Monitors head position for balance
        "Symmetry in Shoulder Position",            # 7 - Checks symmetry between both shoulders
        f"{side.capitalize()} Arm Extension",       # 8 - Measures reach of raised arm
        "Pelvic Rotation"                           # 9 - Assesses any rotation of the pelvis
    ]

    # Define indices based on side for raised arm tracking
    if side == 'left':
        shoulder = 7
        elbow = 8
        hand = 10
        opposite_shoulder = 11
    elif side == 'right':
        shoulder = 11
        elbow = 12
        hand = 14
        opposite_shoulder = 7
    else:
        raise ValueError("Side must be 'left' or 'right'")

    for i, frame in enumerate(frames_tensor):
        # Extract key joints
        waist = frame[0]
        spine = frame[1]
        chest = frame[2]
        neck = frame[3]
        shoulder_joint = frame[shoulder]
        elbow_joint = frame[elbow]
        hand_joint = frame[hand]
        opposite_shoulder_joint = frame[opposite_shoulder]
        head = frame[5]

        # Feature calculations

        # 0. Trunk Inclination Angle
        # Measures the angle between the Waist -> Neck line and the vertical axis to assess upright posture.
        trunk_inclination_angle = calculate_angle(waist, neck, np.array([waist[0], waist[1] + 1, waist[2]]))

        # 1. Arm Abduction Angle
        # Measures the abduction angle of the raised arm (Shoulder, Elbow, Hand).
        # Should reach at least 160°.
        arm_abduction_angle = calculate_angle(shoulder_joint, elbow_joint, hand_joint)

        # 2. Arm Plane Deviation
        # Measures the deviation of the arm from the expected plane of motion.
        # Checks if the arm remains aligned with the body.
        arm_plane_deviation = calculate_angle(spine, shoulder_joint, elbow_joint)

        # 3. Shoulder Flexion Angle
        # Measures the shoulder's alignment in relation to the torso.
        shoulder_flexion_angle = calculate_angle(waist, shoulder_joint, elbow_joint)

        # 4. Pelvic Stability (Tilt Angle)
        # Measures the tilt of the pelvis to ensure stability.
        pelvic_stability = calculate_angle(waist, shoulder_joint, opposite_shoulder_joint)

        # 5. Spine Flexion Angle
        # Measures the forward bending of the spine by checking the angle (Spine, Chest, Neck).
        spine_flexion_angle = calculate_angle(spine, chest, neck)

        # 6. Head Stability
        # Measures the stability of the head position by aligning it with the neck.
        head_stability = calculate_angle(neck, head, np.array([neck[0], neck[1] + 1, neck[2]]))

        # 7. Symmetry in Shoulder Position
        # Measures horizontal alignment between the two shoulders.
        shoulder_symmetry = np.abs(shoulder_joint[0] - opposite_shoulder_joint[0])

        # 8. Arm Extension
        # Measures the reach of the raised arm relative to the shoulder.
        arm_extension = np.linalg.norm(shoulder_joint - hand_joint)

        # 9. Pelvic Rotation
        # Measures the rotation of the pelvis for stability.
        pelvic_rotation = calculate_angle(waist, shoulder_joint, opposite_shoulder_joint)

        # Collect all features for this frame
        features.append([
            trunk_inclination_angle,
            arm_abduction_angle,
            arm_plane_deviation,
            shoulder_flexion_angle,
            pelvic_stability,
            spine_flexion_angle,
            head_stability,
            shoulder_symmetry,
            arm_extension,
            pelvic_rotation
        ])

    # Convert to NumPy array for easy handling
    return np.array(features), feature_names


def m08(frames_tensor, side='left'):
    """
    Feature extractor for m08: Standing Shoulder Extension exercise.
    Extracts key features to assess exercise quality based on non-optimal movement criteria and additional features.

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 3) with (x, y, z) positions.
        side (str): 'left' or 'right' to specify the arm performing the extension.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    features = []

    # Define the feature names, starting with non-optimal indicators, followed by general assessment features
    feature_names = [
        "Trunk Inclination Angle",                   # 0 - Non-optimal: Ensures upright trunk posture
        "Head Neutral Position",                     # 1 - Non-optimal: Checks if head is in neutral alignment
        f"{side.capitalize()} Arm Sagittal Plane Deviation", # 2 - Non-optimal: Measures if arm stays in sagittal plane
        f"{side.capitalize()} Shoulder Extension Angle",     # 3 - Non-optimal: Should reach at least 45°
        "Spine Alignment",                           # 4 - Ensures proper alignment of spine
        "Pelvic Stability (Tilt Angle)",             # 5 - Assesses lateral tilt of pelvis
        f"{side.capitalize()} Shoulder Stability",   # 6 - Checks alignment of shoulder joint
        f"{side.capitalize()} Arm Extension Distance", # 7 - Measures reach of extended arm
        "Symmetry in Shoulder Position",             # 8 - Checks alignment between shoulders for balance
        "Head Stability"                             # 9 - Assesses head stability
    ]

    # Define indices based on side for shoulder extension tracking
    if side == 'left':
        shoulder = 7
        elbow = 8
        hand = 10
        opposite_shoulder = 11
    elif side == 'right':
        shoulder = 11
        elbow = 12
        hand = 14
        opposite_shoulder = 7
    else:
        raise ValueError("Side must be 'left' or 'right'")

    for i, frame in enumerate(frames_tensor):
        # Extract key joints
        waist = frame[0]
        spine = frame[1]
        chest = frame[2]
        neck = frame[3]
        shoulder_joint = frame[shoulder]
        elbow_joint = frame[elbow]
        hand_joint = frame[hand]
        opposite_shoulder_joint = frame[opposite_shoulder]
        head = frame[5]

        # Feature calculations

        # 0. Trunk Inclination Angle
        # Measures the angle between the Waist -> Neck line and the vertical axis to assess upright posture.
        trunk_inclination_angle = calculate_angle(waist, neck, np.array([waist[0], waist[1] + 1, waist[2]]))

        # 1. Head Neutral Position
        # Ensures that the head remains aligned with the spine by checking the angle (Neck -> Head).
        head_neutral_position = calculate_angle(neck, head, np.array([neck[0], neck[1] + 1, neck[2]]))

        # 2. Arm Sagittal Plane Deviation
        # Measures the deviation of the arm from the sagittal plane to ensure proper shoulder extension.
        arm_sagittal_plane_deviation = calculate_angle(spine, shoulder_joint, elbow_joint)

        # 3. Shoulder Extension Angle
        # Measures the shoulder extension angle (Shoulder, Elbow, Hand).
        # Should reach at least 45° for proper shoulder extension.
        shoulder_extension_angle = calculate_angle(shoulder_joint, elbow_joint, hand_joint)

        # 4. Spine Alignment
        # Measures the alignment of the spine by checking the angle (Spine -> Chest -> Neck).
        spine_alignment = calculate_angle(spine, chest, neck)

        # 5. Pelvic Stability (Tilt Angle)
        # Measures the tilt of the pelvis to ensure stability.
        pelvic_tilt_angle = calculate_angle(waist, shoulder_joint, opposite_shoulder_joint)

        # 6. Shoulder Stability
        # Checks the alignment of the active shoulder to ensure stability during extension.
        shoulder_stability = np.abs(shoulder_joint[0] - opposite_shoulder_joint[0])

        # 7. Arm Extension Distance
        # Measures the reach of the extended arm relative to the shoulder.
        arm_extension_distance = np.linalg.norm(shoulder_joint - hand_joint)

        # 8. Symmetry in Shoulder Position
        # Measures the horizontal alignment between both shoulders.
        shoulder_symmetry = np.abs(shoulder_joint[0] - opposite_shoulder_joint[0])

        # 9. Head Stability
        # Checks the stability of the head by measuring its alignment with the neck in the sagittal plane.
        head_stability = calculate_angle(neck, head, np.array([neck[0], neck[1], neck[2] + 1]))

        # Collect all features for this frame
        features.append([
            trunk_inclination_angle,
            head_neutral_position,
            arm_sagittal_plane_deviation,
            shoulder_extension_angle,
            spine_alignment,
            pelvic_tilt_angle,
            shoulder_stability,
            arm_extension_distance,
            shoulder_symmetry,
            head_stability
        ])

    # Convert to NumPy array for easy handling
    return np.array(features), feature_names


def m09(frames_tensor, side='left'):
    """
    Feature extractor for m09: Standing Shoulder Internal–External Rotation exercise.
    Extracts key features to assess exercise quality based on non-optimal movement criteria and additional features.

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 3) with (x, y, z) positions.
        side (str): 'left' or 'right' to specify the arm performing the rotation.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    features = []

    # Define the feature names, starting with non-optimal indicators, followed by general assessment features
    feature_names = [
        "Trunk Inclination Angle",                   # 0 - Non-optimal: Ensures upright trunk posture
        "Head Neutral Position",                     # 1 - Non-optimal: Checks if head is in neutral alignment
        f"{side.capitalize()} Arm Internal Rotation Angle", # 2 - Non-optimal: Should reach at least 60°
        f"{side.capitalize()} Arm External Rotation Angle", # 3 - Non-optimal: Should reach at least 60°
        f"{side.capitalize()} Elbow Flexion Angle",  # 4 - Ensures elbow is bent at approximately 90°
        "Spine Alignment",                           # 5 - Ensures proper alignment of spine
        "Pelvic Stability (Tilt Angle)",             # 6 - Assesses lateral tilt of pelvis
        "Shoulder Symmetry",                         # 7 - Checks alignment of shoulder joint
        "Head Stability",                            # 8 - Assesses head stability
        f"{side.capitalize()} Arm Stability"         # 9 - Measures stability of arm position
    ]

    # Define indices based on side for shoulder rotation tracking
    if side == 'left':
        shoulder = 7
        elbow = 8
        hand = 10
        opposite_shoulder = 11
    elif side == 'right':
        shoulder = 11
        elbow = 12
        hand = 14
        opposite_shoulder = 7
    else:
        raise ValueError("Side must be 'left' or 'right'")

    for i, frame in enumerate(frames_tensor):
        # Extract key joints
        waist = frame[0]
        spine = frame[1]
        chest = frame[2]
        neck = frame[3]
        shoulder_joint = frame[shoulder]
        elbow_joint = frame[elbow]
        hand_joint = frame[hand]
        opposite_shoulder_joint = frame[opposite_shoulder]
        head = frame[5]

        # Feature calculations

        # 0. Trunk Inclination Angle
        # Measures the angle between the Waist -> Neck line and the vertical axis to assess upright posture.
        trunk_inclination_angle = calculate_angle(waist, neck, np.array([waist[0], waist[1] + 1, waist[2]]))

        # 1. Head Neutral Position
        # Ensures that the head remains aligned with the spine by checking the angle (Neck -> Head).
        head_neutral_position = calculate_angle(neck, head, np.array([neck[0], neck[1] + 1, neck[2]]))

        # 2. Arm Internal Rotation Angle
        # Measures the internal rotation angle of the arm (Shoulder, Elbow, Hand).
        # Should reach at least 60° in internal rotation.
        arm_internal_rotation_angle = calculate_angle(spine, shoulder_joint, elbow_joint)

        # 3. Arm External Rotation Angle
        # Measures the external rotation angle of the arm (Shoulder, Elbow, Hand).
        # Should reach at least 60° in external rotation.
        arm_external_rotation_angle = 180 - arm_internal_rotation_angle

        # 4. Elbow Flexion Angle
        # Measures the flexion of the elbow joint to ensure it is bent at approximately 90°.
        elbow_flexion_angle = calculate_angle(shoulder_joint, elbow_joint, hand_joint)

        # 5. Spine Alignment
        # Measures the alignment of the spine by checking the angle (Spine -> Chest -> Neck).
        spine_alignment = calculate_angle(spine, chest, neck)

        # 6. Pelvic Stability (Tilt Angle)
        # Measures the tilt of the pelvis to ensure stability.
        pelvic_tilt_angle = calculate_angle(waist, shoulder_joint, opposite_shoulder_joint)

        # 7. Shoulder Symmetry
        # Checks the alignment of the shoulder relative to the opposite shoulder for balance.
        shoulder_symmetry = np.abs(shoulder_joint[0] - opposite_shoulder_joint[0])

        # 8. Head Stability
        # Checks the stability of the head by measuring its alignment with the neck in the sagittal plane.
        head_stability = calculate_angle(neck, head, np.array([neck[0], neck[1], neck[2] + 1]))

        # 9. Arm Stability
        # Measures the stability of the arm position during internal and external rotation.
        arm_stability = np.abs(hand_joint[0] - elbow_joint[0])

        # Collect all features for this frame
        features.append([
            trunk_inclination_angle,
            head_neutral_position,
            arm_internal_rotation_angle,
            arm_external_rotation_angle,
            elbow_flexion_angle,
            spine_alignment,
            pelvic_tilt_angle,
            shoulder_symmetry,
            head_stability,
            arm_stability
        ])

    # Convert to NumPy array for easy handling
    return np.array(features), feature_names


def m10(frames_tensor, side='left'):
    """
    Feature extractor for m10: Standing Shoulder Scaption exercise.
    Extracts key features to assess exercise quality based on non-optimal movement criteria and additional features.

    Parameters:
        frames_tensor (np.ndarray): Tensor of shape (frames, joints, 3) with (x, y, z) positions.
        side (str): 'left' or 'right' to specify the arm performing the scaption.

    Returns:
        np.ndarray: Extracted features for each frame.
        list: List of feature names.
    """
    features = []

    # Define the feature names, starting with non-optimal indicators, followed by general assessment features
    feature_names = [
        "Trunk Inclination Angle",                   # 0 - Non-optimal: Ensures upright trunk posture
        "Head Neutral Position",                     # 1 - Non-optimal: Checks if head is in neutral alignment
        f"{side.capitalize()} Arm Plane Deviation",  # 2 - Non-optimal: Measures if arm stays in correct plane
        f"{side.capitalize()} Arm Elevation Angle",  # 3 - Non-optimal: Should reach at least 90°
        f"{side.capitalize()} Elbow Stability",      # 4 - Ensures elbow remains straight
        "Spine Alignment",                           # 5 - Ensures proper alignment of spine
        "Pelvic Stability (Tilt Angle)",             # 6 - Assesses lateral tilt of pelvis
        "Shoulder Symmetry",                         # 7 - Checks alignment of shoulder joint
        "Head Stability",                            # 8 - Assesses head stability
        f"{side.capitalize()} Arm Stability"         # 9 - Measures stability of arm position
    ]

    # Define indices based on side for shoulder scaption tracking
    if side == 'left':
        shoulder = 7
        elbow = 8
        hand = 10
        opposite_shoulder = 11
    elif side == 'right':
        shoulder = 11
        elbow = 12
        hand = 14
        opposite_shoulder = 7
    else:
        raise ValueError("Side must be 'left' or 'right'")

    for i, frame in enumerate(frames_tensor):
        # Extract key joints
        waist = frame[0]
        spine = frame[1]
        chest = frame[2]
        neck = frame[3]
        shoulder_joint = frame[shoulder]
        elbow_joint = frame[elbow]
        hand_joint = frame[hand]
        opposite_shoulder_joint = frame[opposite_shoulder]
        head = frame[5]

        # Feature calculations

        # 0. Trunk Inclination Angle
        # Measures the angle between the Waist -> Neck line and the vertical axis to assess upright posture.
        trunk_inclination_angle = calculate_angle(waist, neck, np.array([waist[0], waist[1] + 1, waist[2]]))

        # 1. Head Neutral Position
        # Ensures that the head remains aligned with the spine by checking the angle (Neck -> Head).
        head_neutral_position = calculate_angle(neck, head, np.array([neck[0], neck[1] + 1, neck[2]]))

        # 2. Arm Plane Deviation
        # Measures the deviation of the arm from the correct plane of motion.
        arm_plane_deviation = calculate_angle(spine, shoulder_joint, elbow_joint)

        # 3. Arm Elevation Angle
        # Measures the elevation angle of the arm (Shoulder, Elbow, Hand).
        # Should reach at least 90° for proper scaption.
        arm_elevation_angle = calculate_angle(shoulder_joint, elbow_joint, hand_joint)

        # 4. Elbow Stability
        # Ensures that the elbow remains straight during the arm lift.
        elbow_stability = calculate_angle(shoulder_joint, elbow_joint, hand_joint)

        # 5. Spine Alignment
        # Measures the alignment of the spine by checking the angle (Spine -> Chest -> Neck).
        spine_alignment = calculate_angle(spine, chest, neck)

        # 6. Pelvic Stability (Tilt Angle)
        # Measures the tilt of the pelvis to ensure stability.
        pelvic_tilt_angle = calculate_angle(waist, shoulder_joint, opposite_shoulder_joint)

        # 7. Shoulder Symmetry
        # Checks the alignment of the active shoulder relative to the opposite shoulder for balance.
        shoulder_symmetry = np.abs(shoulder_joint[0] - opposite_shoulder_joint[0])

        # 8. Head Stability
        # Checks the stability of the head by measuring its alignment with the neck in the sagittal plane.
        head_stability = calculate_angle(neck, head, np.array([neck[0], neck[1], neck[2] + 1]))

        # 9. Arm Stability
        # Measures the stability of the arm position during the lift.
        arm_stability = np.abs(hand_joint[0] - elbow_joint[0])

        # Collect all features for this frame
        features.append([
            trunk_inclination_angle,
            head_neutral_position,
            arm_plane_deviation,
            arm_elevation_angle,
            elbow_stability,
            spine_alignment,
            pelvic_tilt_angle,
            shoulder_symmetry,
            head_stability,
            arm_stability
        ])

    # Convert to NumPy array for easy handling
    return np.array(features), feature_names
import numpy as np

class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def magnitude(self):
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def angle_with(self, other):
        dot_product = self.dot(other)
        magnitudes = self.magnitude() * other.magnitude()
        if magnitudes == 0:
            return 0
        cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)


def calculate_angle_between_vectors(v1, v2):
    # return v1.angle_with(v2)
    return int(v1.angle_with(v2))


def RShAbd_features(joint_data): # right standing shoulder abduction
    num_frames = joint_data.shape[0]
    features = np.zeros((num_frames, 3))  # 3 features: shoulder abduction, elbow flexion, torso inclination

    for i in range(num_frames):
        frame = joint_data[i]
        chest = Vector3D(*frame[2])  # Spine (Chest)
        waist = Vector3D(*frame[0])  # Waist

        shoulder = Vector3D(*frame[11])  # Right Collar (Right Shoulder)
        elbow = Vector3D(*frame[12])  # Right Upper Arm (Right Elbow)
        hand = Vector3D(*frame[13])  # Right Hand

        # Shoulder abduction angle
        torso_to_shoulder_vector = Vector3D(
            shoulder.x - chest.x,
            shoulder.y - chest.y,
            shoulder.z - chest.z
        )

        # vertical vector (assume upright)
        torso_vector = Vector3D(
            chest.x - waist.x,
            chest.y - waist.y,
            chest.z - waist.z
        )

        shoulder_to_elbow_vector = Vector3D(
            elbow.x - shoulder.x,
            elbow.y - shoulder.y,
            elbow.z - shoulder.z
        )

        shoulder_abduction_angle = calculate_angle_between_vectors(torso_vector, shoulder_to_elbow_vector)

        # Elbow flexion angle
        elbow_to_hand_vector = Vector3D(
            hand.x - elbow.x,
            hand.y - elbow.y,
            hand.z - elbow.z
        )

        # collar_to_elbow_vector = Vector3D(
        #     elbow.x - collar.x,
        #     elbow.y - collar.y,
        #     elbow.z - collar.z
        # )

        elbow_flexion_angle = calculate_angle_between_vectors(shoulder_to_elbow_vector, elbow_to_hand_vector)

        # Torso inclination angle
        vertical_vector = Vector3D(0, 1, 0)
        torso_vector = Vector3D(
            chest.x - waist.x,
            chest.y - waist.y,
            chest.z - waist.z
        )

        torso_inclination_angle = calculate_angle_between_vectors(vertical_vector, torso_vector)

        features[i] = [180 - shoulder_abduction_angle, elbow_flexion_angle, torso_inclination_angle]
        # features[i] = [180 - shoulder_abduction_angle, 180 - elbow_flexion_angle, torso_inclination_angle]

    return features

def LShAbd_features(joint_data): # left standing shoulder abduction
    num_frames = joint_data.shape[0]
    features = np.zeros((num_frames, 3))  # 3 features: shoulder abduction, elbow flexion, torso inclination

    for i in range(num_frames):
        frame = joint_data[i]
        chest = Vector3D(*frame[2])  # Spine (Chest)
        waist = Vector3D(*frame[0])  # Waist

        shoulder = Vector3D(*frame[7])  # Left Upper Arm (Left Shoulder)
        elbow = Vector3D(*frame[8])  # Left Forearm (Left Elbow)
        hand = Vector3D(*frame[9])  # Left Hand

        # Shoulder abduction angle
        torso_to_shoulder_vector = Vector3D(
            shoulder.x - chest.x,
            shoulder.y - chest.y,
            shoulder.z - chest.z
        )

        # vertical vector (assume upright)
        torso_vector = Vector3D(
            chest.x - waist.x,
            chest.y - waist.y,
            chest.z - waist.z
        )


        shoulder_to_elbow_vector = Vector3D(
            elbow.x - shoulder.x,
            elbow.y - shoulder.y,
            elbow.z - shoulder.z
        )

        shoulder_abduction_angle = calculate_angle_between_vectors(torso_vector, shoulder_to_elbow_vector)

        # Elbow flexion angle
        elbow_to_hand_vector = Vector3D(
            hand.x - elbow.x,
            hand.y - elbow.y,
            hand.z - elbow.z
        )

        collar_to_elbow_vector = Vector3D(
            elbow.x - collar.x,
            elbow.y - collar.y,
            elbow.z - collar.z
        )

        elbow_flexion_angle = calculate_angle_between_vectors(collar_to_elbow_vector, elbow_to_hand_vector)

        # Torso inclination angle
        vertical_vector = Vector3D(0, 1, 0)
        torso_vector = Vector3D(
            chest.x - waist.x,
            chest.y - waist.y,
            chest.z - waist.z
        )

        torso_inclination_angle = calculate_angle_between_vectors(vertical_vector, torso_vector)

        features[i] = [180 - shoulder_abduction_angle, elbow_flexion_angle, torso_inclination_angle]
        # features[i] = [180 - shoulder_abduction_angle, 180 - elbow_flexion_angle, torso_inclination_angle]

    return features

def calculate_neck_chest_angle(joint_data):
    num_frames = joint_data.shape[0]
    neck_chest_angles = np.zeros(num_frames)

    for i in range(num_frames):
        frame = joint_data[i]
        neck = Vector3D(*frame[3])  # Neck
        chest = Vector3D(*frame[2])  # Chest (Spine)

        # Vector from chest to neck
        chest_to_neck_vector = Vector3D(
            neck.x - chest.x,
            neck.y - chest.y,
            neck.z - chest.z
        )

        # Vector representing vertical direction
        vertical_vector = Vector3D(0, 1, 0)

        # Calculate the angle between chest_to_neck_vector and vertical_vector
        neck_chest_angle = calculate_angle_between_vectors(chest_to_neck_vector, vertical_vector)
        neck_chest_angles[i] = neck_chest_angle

    return neck_chest_angles


def calculate_joint_angle_with_vertical(joint_data, joint1_index, joint2_index):
    num_frames = joint_data.shape[0]
    angles = np.zeros(num_frames)

    for i in range(num_frames):
        frame = joint_data[i]
        joint1 = Vector3D(*frame[joint1_index])
        joint2 = Vector3D(*frame[joint2_index])

        # Vector from joint1 to joint2
        joint_vector = Vector3D(
            joint2.x - joint1.x,
            joint2.y - joint1.y,
            joint2.z - joint1.z
        )

        # Vertical vector
        vertical_vector = Vector3D(0, 1, 0)

        # Calculate the angle between the joint vector and the vertical vector
        angle = calculate_angle_between_vectors(joint_vector, vertical_vector)
        angles[i] = angle

    return angles


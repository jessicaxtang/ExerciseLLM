import math as m
import numpy as np

def Rotx(theta):
    return np.matrix([[1, 0, 0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta), m.cos(theta)]])


def Roty(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])


def Rotz(theta):
    return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                      [m.sin(theta), m.cos(theta), 0],
                      [0, 0, 1]])


def eulers_2_rot_matrix(x):
    gamma_x = x[0]
    beta_y = x[1]
    alpha_z = x[2]
    return Rotz(alpha_z) * Roty(beta_y) * Rotx(gamma_x)


# convert the data from relative coordinates to absolute coordinates
def rel2abs(p, a, num_kp, num_axes, num_frames):
    skel = np.zeros((num_kp, num_axes, num_frames))
    for i in range(num_frames):
        """
        1 Waist (absolute)
        2 Spine
        3 Chest
        4 Neck
        5 Head
        6 Head tip
        7 Left collar
        8 Left upper arm 
        9 Left forearm
        10 Left hand
        11 Right collar
        12 Right upper arm 
        13 Right forearm
        14 Right hand
        15 Left upper leg 
        16 Left lower leg 
        17 Left foot 
        18 Left leg toes
        19 Right upper leg 
        20 Right lower leg 
        21 Right foot
        22 Right leg toes
        """

        # extract joint pos and angles for current frame
        joint = p[:, :, i]
        joint_ang = a[:, :, i]

        # chest, neck, head
        rot = eulers_2_rot_matrix(joint_ang[0, :] * np.pi / 180) # initialize waist rotation matrix
        for j in range(1, 6): # spine to head tip
            rot = rot @ eulers_2_rot_matrix(joint_ang[j, :] * np.pi / 180)
            joint[j, :] = rot @ joint[j, :] + joint[j - 1, :]

        # left-arm
        rot = eulers_2_rot_matrix(joint_ang[2, :] * np.pi / 180) 
        joint[6, :] = rot @ joint[6, :] + joint[2, :] # left collar rotation matrix
        for j in range(7, 10):
            rot = rot @ eulers_2_rot_matrix(joint_ang[j - 1, :] * np.pi / 180)
            joint[j, :] = rot @ joint[j, :] + joint[j - 1, :]

        # right-arm
        rot = eulers_2_rot_matrix(joint_ang[2, :] * np.pi / 180)
        joint[10, :] = rot @ joint[10, :] + joint[2, :] # right collar rotation matrix
        for j in range(11, 14):
            rot = rot @ eulers_2_rot_matrix(joint_ang[j - 1, :] * np.pi / 180)
            joint[j, :] = rot @ joint[j, :] + joint[j - 1, :]

        # left-leg
        rot = eulers_2_rot_matrix(joint_ang[0, :] * np.pi / 180)
        joint[14, :] = rot @ joint[14, :] + joint[0, :] # left upper leg rotation matrix
        for j in range(15, 18):
            rot = rot @ eulers_2_rot_matrix(joint_ang[j - 1, :] * np.pi / 180)
            joint[j, :] = rot @ joint[j, :] + joint[j - 1, :]

        # right-leg
        rot = eulers_2_rot_matrix(joint_ang[0, :] * np.pi / 180)
        joint[18, :] = rot @ joint[18, :] + joint[0, :]
        for j in range(19, 22):
            rot = rot @ eulers_2_rot_matrix(joint_ang[j - 1, :] * np.pi / 180)
            joint[j, :] = rot @ joint[j, :] + joint[j - 1, :]

        skel[:, :, i] = joint
    return skel      

def transform_data(data, num_kp, num_axes):
    data = data.T
    data = data.reshape(num_kp, num_axes, -1)
    return data
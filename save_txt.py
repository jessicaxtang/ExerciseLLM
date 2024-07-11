import numpy as np

sliced_motion = np.load('positions_correct_m07_s01_e01.npy')
motion_name = 'right shoulder abduction'

prompt = "Q: is the " + motion_name + " exercise performed correctly, can you provide feedback to help me correct it? Here is position data of the right upper arm, forearm, and hand?\n" + str(sliced_motion) + "A:"
# print(prompt)

filename = "positions_correct_m07_s01_e01.txt"
with open(filename, 'w', encoding='unicode-escape') as f:
    f.write(str(sliced_motion))
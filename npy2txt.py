
# Program to save a NumPy array to a text file
 
# Importing required libraries
import numpy as np
 
 # loading in numpy array
motion_data = np.load('motion.npy')
motion_shape = motion_data.shape
motion_data = motion_data.reshape(motion_data.shape[1], 22, 3)
joints = [16, 18, 20] # shoulder, elbow, wrist
sliced_motion = motion_data[:, joints, :]

# Displaying the array
print('sliced_motion:\n', sliced_motion)
file = open("file1.txt", "w+")
 
# Saving the array in a text file
content = str(sliced_motion)
file.write(content)
file.close()
 
# Displaying the contents of the text file
file = open("file1.txt", "r")
content = file.read()
 
print("\nContent in file1.txt:\n", content)
file.close()
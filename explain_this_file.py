# This code just analyzes some of the hdf5 stuff and lets you poke around a bit

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Replace with the path to your HDF5 file

f = h5py.File('Experiment_moving_target_forward_data.hdf5')
print(list(f.keys()))   #h5py.File works like a python dictionary
                        #we can check the keys

sensor_dataset = f['Sensors']

# calling "sensor_dataset.shape" in terminal bounces an error because it is a group
# in the hierarchy, not an actual dataset

print(sensor_dataset.name) # is the name of whats in sensor_dataset


print("This is everything in here\n_______________________________________________")
def printname(name):
    print(name)
#f.visit(printname) # This will print all of the hierarcies and files in the file

display_all = input("Do you want to skip the blurb [y/N]?\n")

if display_all == "n" or display_all == "N":
    f.visit(printname)

# until this point, everything is from https://docs.h5py.org/en/stable/quick.html#quick
# looking now at https://docs.h5py.org/en/stable/high/group.html#group

frame_data = f['Sensors/TI_Radar/Data/Frame_100/frame_data']

# samples x chirps x channels
print(frame_data.dtype)
print(frame_data[:,0,0])


f.close()


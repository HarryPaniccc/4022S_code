# This notebook identifies a ball in radar data
# range_doppler_notebook has the basics of how to get the basics (what was I thinking when I wrote this)

import h5py
import numpy as np
import matplotlib.pyplot as plt
from radar_functions import get_measurement_parameters, get_data_files, range_doppler_map, cfar_map, make_map, save_map

# This was before creating the new functions
# from radar_functions import range_doppler_map, get_measurement_parameters, cfar_map, save_range_doppler_map, save_cfar_map, get_data_files


c = 299792458 # metres per second - need this

session5_directory = ['../4022S_data/session5/rob_config/test1_calibration/',
                      '../4022S_data/session5/rob_config/test2_motion_calibration/',
                      '../4022S_data/session5/rob_config/test3_basketball_throw/',
                      '../4022S_data/session5/rob_config/test4_tennis_throw/',
                      '../4022S_data/session5/rob_config/test5_golf_throw/',
                      '../4022S_data/session5/rob_config/test6_soccer_roll/',
                      '../4022S_data/session5/rob_config/test7_tennis_roll/',
                      '../4022S_data/session5/rob_config/test8_golf_roll/',
                      '../4022S_data/session5/rob_config/test9_tennis_golf_similar_roll/']

session5_data = []

for i in range(len(session5_directory)):
    loaded_data = get_data_files(session5_directory[i])
    session5_data.append(loaded_data)

session5_frames_directory = ['frames/test1_calibration/',
                             'frames/test2_motion_calibration/',
                             'frames/test3_basketball_throw/',
                             'frames/test4_tennis_throw/',
                             'frames/test5_golf_throw/',
                             'frames/test6_soccer_roll/',
                             'frames/test7_tennis_roll/',
                             'frames/test8_golf_roll/',
                             'frames/test9_tennis_golf_similar_roll/']

tests_in_question = session5_data

_, _, _, _, _, range_bin_size, velocity_resolution = get_measurement_parameters(tests_in_question[0][0])

# Need to take all frames of a target and generate all the maps
frame_number = 0
test_number = 0

#Session 5 is broken up into a number of experiments, each containing a number of tests

for experiment_number in range(len(session5_data)):
    for test_being_saved in session5_data[experiment_number]:
        while True:
            try:
                print(f'Saving range-doppler map {test_number} frame {frame_number} into {session5_frames_directory[experiment_number]}')
                rd_map = range_doppler_map(test_being_saved, frame_number, False)
                save_map(rd_map,
                         range_bin_size,
                         velocity_resolution,
                         False,
                         f'range-doppler map test {test_number} frame {frame_number}',
                         session5_frames_directory[experiment_number])
                
                print(f'Saving cfar map {test_number} frame {frame_number} into {session5_frames_directory[experiment_number]}')
                cf_map = cfar_map(rd_map, range_bin_size, velocity_resolution, False)
                save_map(cf_map,                                                        # Target map
                         range_bin_size,                                                # range bins
                         velocity_resolution,                                           #
                         True, f'cfar map test {test_number} frame {frame_number}',     # name
                         session5_frames_directory[experiment_number])                  # directory saved to
                
                frame_number += 1
            except:
                print("Onto the next test")
                frame_number = 0
                test_number += 1
                break
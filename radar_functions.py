# A collection of functions useful the generation of range-doppler maps, CFAR maps, as well as
# generating images and other niceties in tracking golfballs.
# Developed by Harry Papanicolaou for EEE4022S - Semester 2 2024
# NOT to be confused for functions.py - terribly named but does other things

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift # Might not need this
from radar_ffts import range_doppler_fft, range_doppler_sum
from cfar import cfar, clean_cfar
import os

# Gets data about the h5py file, and stores it
def get_measurement_parameters(hdf5_file_path):

    """Outputs in order: freq_slope_const, number_of_samples_per_chirp, sample_rate, Tdata, bandwidth, range_bin_size,, velocity_resolution (doppler bin size in m/s)"""

    c = 299792458 # metres per second - need this to be declared within the function I think

    freq_slope_const = hdf5_file_path['Sensors/TI_Radar/Parameters/profileCfg/freqSlopeConst'][()] # In MHz per microsecond
    chirp_start_index = hdf5_file_path['Sensors/TI_Radar/Parameters/frameCfg/chirpStartIndex'][()]
    chirp_end_index = hdf5_file_path['Sensors/TI_Radar/Parameters/frameCfg/chirpEndIndex'][()]
    frame_period = hdf5_file_path['Sensors/TI_Radar/Parameters/frameCfg/framePeriod'][()] # Probably in ms
    ramp_end_time = hdf5_file_path['Sensors/TI_Radar/Parameters/profileCfg/rampEndTime'][()]
    start_freq = hdf5_file_path['Sensors/TI_Radar/Parameters/profileCfg/startFreq'][()] # In GHz
    number_of_samples_per_chirp = hdf5_file_path['Sensors/TI_Radar/Parameters/profileCfg/numAdcSamples'][()]
    sample_rate = hdf5_file_path['Sensors/TI_Radar/Parameters/profileCfg/digOutSampleRate'][()] # in ks per second

    Tdata = number_of_samples_per_chirp * 1/(sample_rate*1e3)
    # From this we can find bandwidth with B = Tdata * frequency_slope_constant
    bandwidth = Tdata * freq_slope_const * 1e6 * 1e6 # This leaves it in Hz

    center_frequency = start_freq * 1e9 + bandwidth * 0.5
    velocity_resolution = (c / center_frequency)/(2 * frame_period*1e-3)

    # From these we know c and can calculate the range bin size

    range_bin_size = c / (2 * bandwidth)
    return freq_slope_const, number_of_samples_per_chirp, sample_rate, Tdata, bandwidth, range_bin_size, velocity_resolution



def get_data_files(data_directory):

    """data_directory = a SINGLE string directory of the location of the files of a test set that wants to be stored together"""
    files = []
    hdf5_data = []

    for path in os.listdir(data_directory):
        if os.path.isfile(os.path.join(data_directory, path)):
            files.append(path)

    for data_file in files:
        measurement = h5py.File(f'{data_directory}{data_file}','r')
        hdf5_data.append(measurement)

    return hdf5_data



def range_doppler_map(hdf5_file_path, frame, make_map_check):
    
    """Generates a range doppler map of hdf5 radar data. Can generate a plot (make_map = 1) or just the data (make_map = 0)
        -> make_map = 1 plots the heatmap, heat_map = 0 skips it
        -> save_map = 1 saves the map as a png, save_map = 0 skips it
        Its important to note that the input to this must be in the default orientation as defined by the radar when it takes data. 
        Rotating it before the transforms could break everything"""
    
    frame_data = hdf5_file_path[f'Sensors/TI_Radar/Data/Frame_{frame}/frame_data']
    range_pad = 0
    doppler_pad = 0

    _, _, _, _, _, range_bin_size, velocity_resolution = get_measurement_parameters(hdf5_file_path)

    fftd_frame_data = range_doppler_fft(frame_data, range_pad, doppler_pad)
    plotted_fftd_frame_data = range_doppler_sum(fftd_frame_data)
    plotted_fftd_frame_data=np.flip(plotted_fftd_frame_data, 0)

    if make_map_check:
        make_map(np.rot90(plotted_fftd_frame_data), range_bin_size, velocity_resolution, False) # NOTE: Rotation is done here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
    return plotted_fftd_frame_data



def cfar_map(range_doppler_data, range_bin_size, velocity_resolution, make_map_check):
    
    """Makes a cfar map from range-doppler map"""
    
    cfar_output = cfar((10**(range_doppler_data/20)), 5, 5, 3, 3, 1e-4,0) # <<<<<<<<<<<<<<<<<<<<<< This window is what edits your visibility

    if make_map_check:
        make_map(np.rot90(cfar_output), range_bin_size, velocity_resolution, True) # NOTE: Rotation is done here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    return cfar_output



def make_map(map_data, range_bin_size, velocity_resolution, cfar_map):

    """Takes a 2d heatmap and plots it correctly scaled based on range bin size and velocity resolution
    cfar_map = true -> its a cfar map, etc
    Important to note that rotation is NOT DONE HERE, it needs to be done on the data BEFORE inputting it here"""

    # First we get dimensions of the graphs
    num_doppler_bins, num_range_bins = map_data.shape
    maximum_range = num_range_bins * range_bin_size
    maximum_velocity = num_doppler_bins * velocity_resolution / 2

    # Next we start to plot it
    plt.figure()

    if cfar_map == True:
        plt.imshow(map_data, aspect='auto', cmap='binary', extent=[0, maximum_range,-maximum_velocity, maximum_velocity])
    else:
        plt.imshow(map_data, aspect='auto', cmap='jet', extent=[0, maximum_range,-maximum_velocity, maximum_velocity])

    plt.title('Range-Doppler Map')
    plt.xlabel('Range')
    plt.ylabel('Doppler')
    plt.colorbar(label='Power (dB)')
    
    plt.show()



def save_map(map_data, range_bin_size, velocity_resolution, cfar_map, image_name):
    """Takes in a cfar map or a range-doppler map frame and saves it as an image
    if cfar_map = true it will save it as a greyscaled cfar map rather than a heatmap"""

    # First we get dimensions of the graphs
    num_doppler_bins, num_range_bins = map_data.shape
    maximum_range = num_range_bins * range_bin_size
    maximum_velocity = num_doppler_bins * velocity_resolution / 2

    # Next we start to plot it
    plt.figure()

    if cfar_map == True:
        plt.imshow(map_data, aspect='auto', cmap='binary', extent=[0, maximum_range,-maximum_velocity, maximum_velocity])
    else:
        plt.imshow(map_data, aspect='auto', cmap='jet', extent=[0, maximum_range,-maximum_velocity, maximum_velocity])

    plt.title('Range-Doppler Map')
    plt.xlabel('Range')
    plt.ylabel('Doppler')
    plt.colorbar(label='Power (dB)')

    plt.draw() # Forces a figure redraw just in case
    plt.savefig(f'frames/{image_name}.png', format = 'png')



# def range_doppler_map(hdf5_file_path, frame, range_bin_size, make_map): 
    
#     """Generates a range doppler map of hdf5 radar data. Can generate a plot (make_map = 1) or just the data (make_map = 0)
#         -> make_map = 1 plots the heatmap, heat_map = 0 skips it
#         -> save_map = 1 saves the map as a png, save_map = 0 skips it"""
    
#     frame_data = hdf5_file_path[f'Sensors/TI_Radar/Data/Frame_{frame}/frame_data']
#     range_pad = 0
#     doppler_pad = 0

# #    _, _, _, _, _, range_bin_size = get_measurement_parameters(hdf5_file_path) # Think this is cleaner but slower way of getting it
     

#     fftd_frame_data = range_doppler_fft(frame_data, range_pad, doppler_pad)
#     plotted_fftd_frame_data = range_doppler_sum(fftd_frame_data)
#     plotted_fftd_frame_data=np.flip(plotted_fftd_frame_data, 0)

#     if make_map:
#         plt.figure()
#         plt.imshow(plotted_fftd_frame_data, aspect='auto', cmap='jet')
#         plt.title('Range-Doppler Map')
#         plt.xlabel('Doppler')
#         plt.ylabel('Range')
#         plt.colorbar(label='Power (dB)')
#         # Get current y-ticks and labels
#         y_ticks = plt.gca().get_yticks()
#         plt.gca().set_yticklabels(y_ticks[::1]*range_bin_size) #TODO: Figure out how to relabel the data, not just the ticks
#         plt.show()
        
#     return plotted_fftd_frame_data



# def save_range_doppler_map(range_doppler_data, range_bin_size, image_name):
    
#     """Takes a range-doppler map, a range bin size and the desired name and saves the image of the graph created"""
    
#     plt.figure()
#     plt.imshow(range_doppler_data, aspect='auto', cmap='jet')
#     plt.title('Range-Doppler Map')
#     plt.xlabel('Doppler')
#     plt.ylabel('Range')
#     plt.colorbar(label='Power (dB)')
#     # Get current y-ticks and labels
#     y_ticks = plt.gca().get_yticks()
#     plt.gca().set_yticklabels(y_ticks[::1]*range_bin_size) #TODO: Figure out how to relabel the data, not just the ticks

#     plt.draw() # Forces a figure redraw to update y ticksThey

#     plt.savefig(f'frames/{image_name}.png', format = 'png')
#     return 0


def save_cfar_map(cfar_map_data, range_bin_size, image_name):
    
    """Takes in a cfar map and saves it, using the range bin size, as well as a file name, to a file
    Takes LINEAR input from the output of the cfar_map function"""
    
    plt.figure()
    plt.imshow(cfar_map_data, aspect = 'auto', cmap = 'binary')
    plt.title('CFAR Map')
    plt.xlabel('Doppler')
    plt.ylabel('Range')
    plt.colorbar(label='Power (dB)')
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels(y_ticks[::1]*range_bin_size) #TODO: Figure out how to relabel the data, not just the ticks
    plt.savefig(f'frames/{image_name}.png', format = 'png')

# A collection of functions useful the generation of range-doppler maps, CFAR maps, as well as
# generating images and other niceties in tracking golfballs.
# Developed by Harry Papanicolaou for EEE4022S - Semester 2 2024

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift # Might not need this
from radar_ffts import range_doppler_fft, range_doppler_sum
from cfar import cfar, clean_cfar

c = 299792458 # metres per second - need this

# Gets data about the h5py file, and stores it
def get_measurement_parameters(hdf5_file_path):
    freq_slope_const = hdf5_file_path['Sensors/TI_Radar/Parameters/profileCfg/freqSlopeConst'][()] # In MHz per microsecond
    chirp_start_index = hdf5_file_path['Sensors/TI_Radar/Parameters/frameCfg/chirpStartIndex'][()]
    chirp_end_index = hdf5_file_path['Sensors/TI_Radar/Parameters/frameCfg/chirpEndIndex'][()]
    frame_period = hdf5_file_path['Sensors/TI_Radar/Parameters/frameCfg/framePeriod'][()]
    ramp_end_time = hdf5_file_path['Sensors/TI_Radar/Parameters/profileCfg/rampEndTime'][()]
    start_freq = hdf5_file_path['Sensors/TI_Radar/Parameters/profileCfg/startFreq'][()]
    number_of_samples_per_chirp = hdf5_file_path['Sensors/TI_Radar/Parameters/profileCfg/numAdcSamples'][()]
    sample_rate = hdf5_file_path['Sensors/TI_Radar/Parameters/profileCfg/digOutSampleRate'][()] # in ks per second

    Tdata = number_of_samples_per_chirp * 1/(sample_rate*1000)
    # From this we can find bandwidth with B = Tdata * frequency_slope_constant
    bandwidth = Tdata * freq_slope_const * 1e6 * 1e6 # This leaves it in Hz

    # From these we know c and can calculate the range bin size
    range_bin_size = c/(2*bandwidth)
    return freq_slope_const, number_of_samples_per_chirp, sample_rate, Tdata, bandwidth, range_bin_size


def range_doppler_map(hdf5_file_path, frame, range_bin_size, make_map): 
    """Generates a range doppler map of hdf5 radar data. Can generate a plot (make_map = 1) or just the data (make_map = 0)
        -> make_map = 1 plots the heatmap, heat_map = 0 skips it
        -> save_map = 1 saves the map as a png, save_map = 0 skips it"""
    
    frame_data = hdf5_file_path[f'Sensors/TI_Radar/Data/Frame_{frame}/frame_data']
    range_pad = 0
    doppler_pad = 0

    fftd_frame_data = range_doppler_fft(frame_data, range_pad, doppler_pad)
    plotted_fftd_frame_data = range_doppler_sum(fftd_frame_data)
    plotted_fftd_frame_data=np.flip(plotted_fftd_frame_data, 0)

    if make_map:
        plt.figure()
        plt.imshow(plotted_fftd_frame_data, aspect='auto', cmap='jet')
        plt.title('Range-Doppler Map')
        plt.xlabel('Doppler')
        plt.ylabel('Range')
        plt.colorbar(label='Power (dB)')
        # Get current y-ticks and labels
        y_ticks = plt.gca().get_yticks()
        plt.gca().set_yticklabels(y_ticks[::1]*range_bin_size) #TODO: Figure out how to relabel the data, not just the ticks
        plt.show()
        
    return plotted_fftd_frame_data

def save_range_doppler_map(range_doppler_data, range_bin_size, image_name):
    """Takes a range-doppler map, a range bin size and the desired name and saves the image of the graph created"""
    plt.figure()
    plt.imshow(range_doppler_data, aspect='auto', cmap='jet')
    plt.title('Range-Doppler Map')
    plt.xlabel('Doppler')
    plt.ylabel('Range')
    plt.colorbar(label='Power (dB)')
    # Get current y-ticks and labels
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels(y_ticks[::1]*range_bin_size) #TODO: Figure out how to relabel the data, not just the ticks
    plt.savefig(f'frames/{image_name}.png', format = 'png')
    return 0

def cfar_map(range_doppler_data, make_map):
    """Makes a cfar map from range-doppler map"""
    cfar_output = cfar((10**(range_doppler_data/20)), 5, 5, 3, 3, 1e-3,0) # <<<<<<<<<<<<<<<<<<<<<< This window is what edits your visibility

    if make_map:
        plt.figure()
        plt.imshow(cfar_output, aspect='auto', cmap='jet')
        #plt.imshow(20 * np.log10(np.abs(after_slow_time_fft)), aspect='auto', cmap='jet')
        plt.title('CFAR Map')
        plt.xlabel('Doppler')
        plt.ylabel('Range')
        plt.colorbar(label='Power (dB)')
        plt.show()
    return cfar_output

def save_cfar_map(cfar_map_data, range_bin_size, image_name):
    """Takes in a cfar map and saves it, using the range bin size, as well as a file name, to a file
    Takes LINEAR input from the output of the cfar_map function"""
    plt.figure()
    plt.imshow(cfar_map_data, aspect = 'auto', cmap = 'jet')
    plt.title('CFAR Map')
    plt.xlabel('Doppler')
    plt.ylabel('Range')
    plt.colorbar(label='Power (dB)')
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels(y_ticks[::1]*range_bin_size) #TODO: Figure out how to relabel the data, not just the ticks
    plt.savefig(f'frames/{image_name}.png', format = 'png')


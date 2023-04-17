"""
Helper visialization functions for display ultrasound A/B-scans
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.fft import fft, fftshift
from scipy.signal import iirfilter, lfilter
from envelop_detection import *

# CONST
SF = 65e6           # Sampling Frequency
SOS = 5920          # Speed of Sound for L-wave in steel


def display_Ascan(data, norm=False, sample_offset=0):
    """
    Display A-scan with/without normalization to the max. 
    ARGS:
        data - 1D vector of the A-scan
        norm - True/False normalization
        sample_offset - scanline time axis offest in samples
    """
    _, axs = plt.subplots(figsize=(12, 6))
    if norm:
        axs.plot(data[:] / np.max(data[:]))
    else:
        axs.plot(data[:])
    axs.set_title('RF scan-line')
    axs.set_xlabel('depth [mm]', loc='right')
    ticks_x = ticker.FuncFormatter(
        lambda x, pos: '{0:g}'.format(1000 * SOS * (x + sample_offset) / (2 * SF)))
    axs.xaxis.set_major_formatter(ticks_x)


def display_Bmode_from_RF(data, dynamic_range=100, sample_offset=0):
    """ Display B-mode image from RF signal. """
    fig, axs = plt.subplots(figsize=(18, 13))
    imB = axs.imshow(data.copy().T, cmap='viridis', aspect=0.05)
    fig.colorbar(imB)
    imB.set_clim(vmin=-dynamic_range, vmax=dynamic_range)
    axs.set_title('B-mode image')
    axs.set_xlabel('number of scan line')
    axs.set_ylabel('depth [mm]')
    ticks_y = ticker.FuncFormatter(
        lambda y, pos: '{0:g}'.format(1000 * SOS * (y + sample_offset) / (2 * SF)))
    axs.yaxis.set_major_formatter(ticks_y)


def display_video_env_detect(data, selected_func, dynamic_range=500, n_scan_display=1, sample_offset=0):
    """ 
    Display signal after Envelop Detection (aka 'Video')
    Input:
        data                    matrix with RF signal 
        selected_func           function selected from envelop_functions: possible only without local oscilators,
                                synchronous_real, asynchronous_complex_osci V1 and V2 are forbidden
        dynamic range           max value to visualize (necessary for colorbar)
        n_scan_display          number of scan line which will be plotted
        sample_offset             sample from which signal should be cut to remove noise
    """
    arrV = data.copy()
    for n_scan in range(data.shape[0]):
        t = np.arange(len(data[n_scan, :]))
        # TO DO: check if frequency is correct, how to interpret
        f_s = 1/np.diff(t)[0]

        if n_scan == n_scan_display:
            output = selected_func(
                data[n_scan, :], t, f_s, cutoff_freq=0.2, display=1)
        else:
            output = selected_func(
                data[n_scan, :], t, f_s, cutoff_freq=0.2, display=0)

        if n_scan == n_scan_display:
            plt.figure(figsize=[6, 4])
            plt.plot(t, data[n_scan, :], label="Raw signal")
            plt.plot(t, output, label="Detected envelope")
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],
                       ncol=2, fontsize="smaller")

        arrV[n_scan, :] = output

    display_Bmode_from_RF(
        arrV, dynamic_range=dynamic_range, sample_offset=sample_offset)

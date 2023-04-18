"""
Envelop Detection functions.

Code based on Rick Lyons, Digital Envelope Detection: The Good, the Bad, and the Ugly
(https://www.dsprelated.com/showarticle/938.php)
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import iirfilter, lfilter, hilbert
from scipy.ndimage import shift


def LP_filtration(sig, t, fs, cutoff_freq, disp_option=0, raw_sig=[]):
    # spectrum analysis (to help to determine the appropriate cutoff frequency)

    if disp_option:
        L = len(t)
        fft_sig = fftshift(fft(sig))
        freqs = np.arange(-L/2, L/2)*fs/L
        ps = np.abs(fft_sig)**2
        fig, axs = plt.subplots(figsize=[6, 4])
        axs.plot(freqs, ps)
        axs.set_title('Spectrum analysis')
        axs.set_ylabel('Power')
        axs.set_xlabel('Frequency [Hz]')

    # 3.order IIR low-pass filter
    b, a = iirfilter(3, Wn=cutoff_freq, fs=fs, btype="low", ftype="butter")
    y_filtered = lfilter(b, a, sig)

    if len(raw_sig) and disp_option:
        plt.figure(figsize=[6.4, 3.4])
        plt.plot(t, raw_sig, label="Raw signal")
        plt.plot(t, y_filtered, label="LP filter (detected envelope)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],
                   ncol=2, fontsize="smaller")

    return y_filtered


def asynchronous_half_wave(AM, t, fs, cutoff_freq, display=1):
    sig = AM.copy()

    # 1. thresholding to get half-wave rectified sinusoid
    sig[sig < 0] = 0
    if display == 1:
        fig, axs = plt.subplots(figsize=[6, 4])
        axs.plot(t, sig)
        axs.set_title('Half-wave rectified sinusoid')
        axs.set_xlabel('Time [s]')

    # 2. determine the appropriate cutoff frequency for LP filter and
    # 3. third-order IIR low-pass filter
    return LP_filtration(sig, t, fs, cutoff_freq, display, sig)


def asynchronous_full_wave(AM, t, fs, cutoff_freq, display=1):
    sig = AM.copy()

    # 1. absolute value to get full-wave rectified sinusoid
    sig = np.abs(sig)
    if display == 1:
        fig, axs = plt.subplots(figsize=[6, 4])
        axs.plot(t, sig)
        axs.set_title('Full-wave rectified sinusoid')
        axs.set_xlabel('Time [s]')

    # 2. determine the appropriate cutoff frequency for LP filter and
    # 3. third-order IIR low-pass filter
    return LP_filtration(sig, t, fs, cutoff_freq, display, sig)


def asynchronous_real_square_law(AM, t, fs, cutoff_freq, display=1):
    sig = AM.copy()

    # 1. RF squared
    sig = sig**2
    if display == 1:
        fig, axs = plt.subplots(figsize=[6, 4])
        axs.plot(t, sig)
        axs.set_title('RF squared')
        axs.set_xlabel('Time [s]')

    # 2. determine the appropriate cutoff frequency for LP filter and
    # 3. third-order IIR low-pass filter
    y_filtered = LP_filtration(sig, t, fs, cutoff_freq, display)

    # 4. square root
    output = np.sqrt(y_filtered)

    if display == 1:
        plt.figure(figsize=[6.4, 3.4])
        plt.plot(t, sig, label="Raw signal")
        plt.plot(t, output, label="Detected envelope")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],
                   ncol=2, fontsize="smaller")

    return output


def asynchronous_complex_hilbert(AM, t, fs, cutoff_freq, display=1):
    sig = AM.copy()

    # 1. delay + absolute value
    s1 = shift(sig, 10, mode='wrap')
    if display == 1:
        fig, axs = plt.subplots(figsize=[6, 4])
        axs.plot(t, sig, 'red', label='original signal')
        axs.plot(t, s1, label='delayed signal')
        axs.legend(loc="best")
        axs.set_xlabel('Time [s]')

    # 2. FIR Hilbert transformer + absolute value
    s1 = np.abs(s1)
    s2 = np.imag(hilbert(sig))
    s2 = np.abs(s2)

    # 3. add 2 components + LP filter
    return LP_filtration(s1+s2, t, fs, cutoff_freq, display, sig)


def asynchronous_complex_square_law(AM, t, fs, cutoff_freq, display=1):
    sig = AM.copy()

    # 1. delay
    s1 = shift(sig, 10, mode='wrap')
    if display == 1:
        fig, axs = plt.subplots(figsize=[6, 4])
        axs.plot(t, sig, 'red', label='original signal')
        axs.plot(t, s1, label='delayed signal')
        axs.legend(loc="best")
        axs.set_xlabel('Time [s]')

    # 2. FIR Hilbert transformer + value to the power
    s1 = s1**2
    s2 = np.imag(hilbert(sig))
    s2 = s2**2

    # 3. add 2 components + fet square root
    sig_sqroot = np.sqrt(s1+s2)

    # 4. LP filter
    return LP_filtration(sig_sqroot, t, fs, cutoff_freq, display, sig)


def synchronous_real(Ac, fc, m, ym, Am, t, fs, cutoff_freq, display=1):

    n = 2
    phi = math.pi/2 + n*math.pi
    yc_temp = Ac*np.cos(2*math.pi*fc*t+phi)
    AM_temp = yc_temp*m*ym/Am

    # 1. multiplication by a local oscillator signal
    sig = AM_temp*np.cos(2*math.pi*fc*t+phi)

    # 2. LP filter
    return LP_filtration(sig, t, fs, cutoff_freq, display, AM_temp)


def asynchronous_complex_V1_osci(Ac, fc, m, ym, Am, t, fs, cutoff_freq, display=1):

    n = 2
    phi = math.pi/2 + n*math.pi
    yc_temp = Ac*np.cos(2*math.pi*fc*t+phi)
    AM_temp = yc_temp*m*ym/Am

    # 1. extract 2 components
    f0 = fc/5
    s1 = AM_temp*np.cos(2*math.pi*f0*t)
    s2 = AM_temp*np.sin(2*math.pi*f0*t)

    # 2. LP filter
    s1_filtered = LP_filtration(s1, t, fs, cutoff_freq, display)
    s2_filtered = LP_filtration(s2, t, fs, cutoff_freq, display)

    # 3. add squared component and extract envelope
    sig = s1_filtered ** 2 + s2_filtered**2
    output = np.sqrt(sig)
    if display == 1:
        plt.figure(figsize=[6.4, 3.4])
        plt.plot(t, AM_temp, label="Raw signal")
        plt.plot(t, output, label="Detected envelope)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],
                   ncol=2, fontsize="smaller")

    return output


def asynchronous_complex_V2_osci(Ac, fc, m, ym, Am, t, fs, cutoff_freq, display=1):

    n = 2
    phi = math.pi/2 + n*math.pi
    yc_temp = Ac*np.cos(2*math.pi*fc*t+phi)
    AM_temp = yc_temp*m*ym/Am

    # 1. extract 2 components
    f0 = fc/5
    theta = math.pi/4
    s1 = AM_temp*np.cos(2*math.pi*f0*t + theta)
    s2 = AM_temp*np.sin(2*math.pi*f0*t + theta)

    # 2. add squared component
    sig = s1**2 + s2**2
    sig = np.sqrt(sig)

    # 3. LP filter
    return LP_filtration(sig, t, fs, cutoff_freq, display, AM_temp)

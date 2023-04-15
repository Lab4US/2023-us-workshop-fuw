"""
Ultrasound processing/manipulation helper functions.

Code inspired from PICMUS (https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/home)
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
import time


def extents(f):
    delta = f[2] - f[1]
    return [f[1] - delta / 2, f[-1] + delta / 2]


def py_imagesc(ax, x, y, data):
    """ Wrapper for PyPlot's `imshow` to imitate Matlab-style IMAGESC. """
    im = ax.imshow(data, aspect='auto', interpolation='none',
                   extent=extents(x) + extents(y), origin='lower')
    return im


def py_ind2sub(array_shape, ind):
    """ Wrapper for Matlab-style IND2SUB. """
    rows = int((ind / array_shape[1]))
    cols = int((ind % array_shape[1]))
    return (rows, cols)


class US_RecoImage:
    """
    Class containing the DAS (delay-and-sum) beamformed data (source: PICMUS)
    Input:
        file_path:                  full path of recontructed image  
    Public properties:
        x_axis                      vector defining the x coordinates (from scan)
        z_axis                      vector defining the z coordinates (from scan)
        number_plane_waves          vector containing number of plane waves used in each reconstructed frame
        data                        matrix containing the envelope of the reconstructed signal 
        transmit_f_number           scalar of the F-number used on transmit
        transmit_apodization_window string describing the transmit apodization window
        receive_f_number            scalar of the F-number used on receive
        receive_apodization_window  string describing the receive apodization window 
    """
    def __init__(self, file_path):
        data = h5py.File(file_path, "r")

        # read scan
        self.x_axis = data['US']['US_DATASET0000']['scan']['x_axis'][:]
        self.z_axis = data['US']['US_DATASET0000']['scan']['z_axis'][:]
        self.x_matrix, self.z_matrix = np.meshgrid(self.x_axis, self.z_axis)

        #  F-numbers
        self.transmit_f_number = data['US']['US_DATASET0000']['transmit_f_number']
        self.receive_f_number = data['US']['US_DATASET0000']['receive_f_number']

        # Apodization window
        self.transmit_apodization_window = data['US']['US_DATASET0000']['transmit_apodization_window']
        self.receive_apodization_window = data['US']['US_DATASET0000']['receive_apodization_window']

        # read data
        self.real_part = data['US']['US_DATASET0000']['data']['real']
        self.imag_part = data['US']['US_DATASET0000']['data']['imag']
        self.number_plane_waves = data['US']['US_DATASET0000']['number_plane_waves']
        # adding [:] returns a numpy array
        self.data = self.real_part[:] + 1j*self.imag_part[:]


class US_Phantom:
    """
    Class defining a standard way to generate numerical phantom in ultrasound (source: PICMUS)
    Input:
        file_path:  full path of phantom data
    """
    def __init__(self, file_path):
        data = h5py.File(file_path, "r")

        self.occlusionCenterX = data['US']['US_DATASET0000']['phantom_occlusionCenterX'][:]
        self.occlusionCenterZ = data['US']['US_DATASET0000']['phantom_occlusionCenterZ'][:]
        self.occlusionDiameter = data['US']['US_DATASET0000']['phantom_occlusionDiameter'][:]
        self.axialResolution = data['US']['US_DATASET0000']['phantom_axialResolution'][:]
        self.lateralResolution = data['US']['US_DATASET0000']['phantom_lateralResolution'][:]


class US_Contrast:
    """
    Class defining a testing procedure to assess contrast performance (CNR) of beamforming techniques in ultrasound imaging (source: PICMUS)
    Input:
        pht:                US_Phantom object
        image:              US_RecoImage object
        flagDisplay:        flag for displaying beamformed images
    Public properties:
        pht:                as input
        image:              as input
        flagDisplay:        as input 
        score:              CNR (contrast to noise ratio) values for all beamformings per image
        dynamic_range:      max scale value to visualize [in dB]
    """
    padding = 1

    def __init__(self, pht, image, flagDisplay=1):
        self.pht = pht
        self.image = image
        self.flagDisplay = flagDisplay
        self.score = 0
        self.dynamic_range = 60

    def evaluate(self):
        """ Main method call to generate phantom and evaluate contrast """
        # Define parameters / variables
        nb_frames = len(self.image.number_plane_waves[:])
        frame_list = range(nb_frames)
        self.score = np.zeros((nb_frames, len(self.pht.occlusionDiameter)))

        # Setting axis limits (mm)
        x_lim = (np.min(self.image.x_matrix)*1e3,
                 np.max(self.image.x_matrix)*1e3)
        z_lim = (np.min(self.image.z_matrix)*1e3,
                 np.max(self.image.z_matrix)*1e3)
        x = self.image.x_matrix[:]
        z = self.image.z_matrix[:]

        # # Ploting image reconstruction
        if (self.flagDisplay == 1):
            fig, ax = plt.subplots(2, 2, figsize=[20, 18])

        # Loop over frames
        for f in frame_list:
            # Compute dB values
            env = np.transpose(self.image.data[f, :, :])
            bmode = 20*np.log10(env/np.max(env))

            # # Ploting image reconstruction
            if (self.flagDisplay == 1):
                (ix1, ix2) = py_ind2sub(ax.shape, f)
                im = py_imagesc(ax[ix1, ix2], self.image.x_axis*1e3,
                                self.image.z_axis*1e3, abs(bmode))
                im.set_cmap('gray_r')
                im.set_clim(vmin=0, vmax=self.dynamic_range)
                ccb = fig.colorbar(im, ax=ax[ix1, ix2])
                step = 10
                ccb.set_ticks(np.arange(0, self.dynamic_range+step, step),
                              labels=np.flip(np.arange(-self.dynamic_range, step, step)), fontsize=16)
                ax[ix1, ix2].set_xlabel('x [mm]', fontsize=16)
                ax[ix1, ix2].set_xlim(x_lim)
                ax[ix1, ix2].set_ylabel('z [mm]', fontsize=16)
                ax[ix1, ix2].set_ylim(z_lim)
                ax[ix1, ix2].set_title(
                    f'Beamforming for {round(self.image.number_plane_waves[:][f])} plane waves', fontsize=20)
                ax[ix1, ix2].invert_yaxis()
                time.sleep(0.1)

            # Main loop
            for k in range(len(self.pht.occlusionDiameter)):
                r = self.pht.occlusionDiameter[k] / 2
                rin = r - self.padding * self.pht.lateralResolution
                rout1 = r + self.padding * self.pht.lateralResolution
                rout2 = 1.2*math.sqrt(rin**2+rout1**2)
                xc = self.pht.occlusionCenterX[k]
                zc = self.pht.occlusionCenterZ[k]
                maskOcclusion = (((x-xc)**2 + (z-zc)**2) <= r**2)
                maskInside = (((x-xc)**2 + (z-zc)**2) <= rin**2)
                maskOutside = np.logical_and(
                    (((x-xc)**2 + (z-zc)**2) >= rout1**2), (((x-xc)**2 + (z-zc)**2) <= rout2**2))

                # Ploting image reconstruction
                if (self.flagDisplay == 1):
                    ax[ix1, ix2].contour(self.image.x_axis*1e3, self.image.z_axis*1e3,
                                         maskOcclusion, levels=2, colors='yellow', linewidths=2.5)
                    ax[ix1, ix2].contour(self.image.x_axis*1e3, self.image.z_axis*1e3,
                                         maskInside, levels=2, colors='red', linewidths=2.5)
                    ax[ix1, ix2].contour(self.image.x_axis*1e3, self.image.z_axis*1e3,
                                         maskOutside, levels=2, colors='green', linewidths=2.5)
                    time.sleep(0.1)

                # gray level inside the anechoic cystic region
                inside = bmode[maskInside]
                outside = bmode[maskOutside]    # gray level outside

                # CNR value
                value = 20 * math.log10(abs(np.mean(inside)-np.mean(outside)) /
                                        math.sqrt((np.var(inside)+np.var(outside))/2))
                self.score[f, k] = np.round(value*10) / 10

            score_per_image = np.mean(self.score[f, :])
            print(
                f'\n DAS Beamforming for {round(self.image.number_plane_waves[:][f])} plane waves:')
            print(f'Mean image contrast score (dB): {score_per_image} \n')

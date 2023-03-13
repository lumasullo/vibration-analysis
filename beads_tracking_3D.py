"""
    Bead tracking and mechanical noise analysis
    ~~~~~~~~~~~~~~~~
    -
    
    :authors: Luciano A. Masullo 2021
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib as mpl
from scipy.stats import norm
from matplotlib import colors
from matplotlib import cm

from scipy.signal import savgol_filter

import warnings

def moving_average(x, w):
    
    """
    Function to smooth the position curves. The parameter w should be set to
    match a DNA-PAINT acquisition frame, (e.g. w = 20 for frame = 5 ms)
    """
    return np.convolve(x, np.ones(w), 'valid') / w

warnings.filterwarnings("ignore")
plt.close('all')

save_data = False

px_size = 130 # pixel size in nm
n_frames = 10000
dt = 20e-3 # in s, frame real time of the acquisition (might be different from exp time!)
w = 5 # how many frames are we going to average for the low-freq drift (w*dt should roughly correspond to a DNA-PAINT frame)

# directory = "/Users/masullo/Dropbox/Jungmann Lab/vibration analysis"
# in_path = "/Users/masullo/Dropbox/Jungmann Lab/vibration analysis/38mW_1_MMStack_Pos0.ome_locs_picked.hdf5"
# file = "/50k_frames_5ms_exposure_100mW_laser_multipos_1_MMStack_Pos0.ome_locs_picked.hdf5"
# in_path = "/Users/masullo/Dropbox/Jungmann Lab/vibration analysis/200nm_beads_10mW_2_MMStack_Pos0.ome_locs_picked.hdf5"

# directory = r'/Volumes/pool-miblab4/users/masullo/zz.microscopy_raw/2021-12-17 new Voyager drift Fusion BT and Quest/'
# file = r'200_nm_beads_BT_5ms_5mW_288px_box_1/200_nm_beads_BT_5ms_5mW_288px_box_1_MMStack_Default.ome_locs_picked.hdf5'
# file = r'200_nm_beads_Quest_5ms_5mW_288px_box_2/200_nm_beads_Quest_5ms_5mW_288px_box_2_MMStack_Default.ome_locs_picked.hdf5'
# file = r'200_nm_beads_Quest_5ms_5mW_288px_1/200_nm_beads_Quest_5ms_5mW_288px_1_MMStack_Default.ome_locs_picked.hdf5'

# directory = r'/Volumes/pool-miblab4/users/masullo/zz.microscopy_raw/2021-12-21 vibration analysis Fusion BT and Zyla/'
# file = r'200nm_beads_Zyla_10mW_128px_1/200nm_beads_Zyla_10mW_128px_1_MMStack_Default.ome_locs_picked.hdf5'
# file = r'200nm_beads_Zyla_10mW_128px_2/200nm_beads_Zyla_10mW_128px_2_MMStack_Default.ome_locs_picked.hdf5'
# file = r'200nm_beads_BT_5mW_288px_3/200nm_beads_BT_5mW_288px_3_MMStack_Default.ome_locs_picked.hdf5'

# directory = r'/Volumes/pool-miblab4/users/masullo/zz.microscopy_raw/2022-01-31 vibration analysis Fusion BT w cooling and Zyla/'
# file = r'FusionBT_200nm_beads_7mW_5ms_6/FusionBT_200nm_beads_7mW_5ms_6_MMStack_Pos0.ome_locs_picked.hdf5'
# file = r'Zyla_200nm_beads_7mW_5ms_3/Zyla_200nm_beads_7mW_5ms_3_MMStack_Pos0.ome_locs_picked.hdf5'

# directory = r'/Volumes/pool-miblab4/users/masullo/zz.microscopy_raw/2022-02-03 vibration analysis Fusion BT w cooling and Zyla (2)/'
# file = r'FusionBT_200nm_beads_7mW_5ms_2/FusionBT_200nm_beads_7mW_5ms_2_MMStack_Pos0.ome_locs_picked.hdf5'

directory = r'/Volumes/pool-miblab/users/masullo/z_raw/22-11-14 Gemini PFS test/beads_200nm_15mW_3D_1/'
file = r'beads_200nm_15mW_3D_1_MMStack_Pos0.ome_locs_picked.hdf5'

in_path = directory + file
fulltable = pd.read_hdf(in_path, key = 'locs')
# fulltable.groupby('group')

fulltable['x'] = fulltable['x']*px_size
fulltable['y'] = fulltable['y']*px_size
fulltable['z'] = fulltable['z']

number_of_beads = fulltable['group'].max()+1
print("Number of nanoparticles = ", number_of_beads)

fig, ax = plt.subplots(3, figsize = (8,12))
fig.tight_layout()
color_map = cm.get_cmap('plasma', number_of_beads)

n = 1 # interval at which display points in the plot

x_list = []
y_list = []
z_list = []

d_beads = {}

for i in range(number_of_beads):
    bead = fulltable.loc[fulltable['group'] == i]
    
    if len(bead['x'])==n_frames:
    
        x_0 = bead['x'].iloc[0]
        y_0 = bead['y'].iloc[0]
        z_0 = bead['z'].iloc[0]
    
        bead['x'] = bead['x'] - x_0
        bead['y'] = bead['y'] - y_0
        bead['z'] = bead['z'] - z_0
        
        ax[0].plot(bead['frame'][::n], bead['x'][::n], color = "red", alpha=0.05) # displaying every point would overload the plot
        ax[1].plot(bead['frame'][::n], bead['y'][::n], color = "red", alpha=0.05)
        ax[2].plot(bead['frame'][::n], bead['z'][::n], color = "red", alpha=0.05)


        x_list.append(bead["x"].to_numpy())
        y_list.append(bead["y"].to_numpy())
        z_list.append(bead["z"].to_numpy())
        
        keyx = 'x'+str(i)
        keyy = 'y'+str(i)
        keyz = 'z'+str(i)
        
        if save_data:

            d_beads[keyx] = bead["x"].to_numpy()
            d_beads[keyy] = bead["y"].to_numpy()
            d_beads[keyz] = bead["z"].to_numpy()
        
if save_data:
    
    df = pd.DataFrame(d_beads)
    df.to_csv(directory +  r'/'+ 'individual_beads_' + 'data.csv', sep='\t', encoding='utf-8')
    df.to_excel(directory + r'/' + 'individual_beads_' + ' data.xlsx')        
            
ax[0].plot(np.arange(n_frames), x_list[-1], color="red", alpha=0.05, label='single beads')        
        
ax[0].set_xlabel("Frame")
ax[0].set_ylabel("x (nm)")
            
ax[1].set_xlabel("Frame")
ax[1].set_ylabel("y (nm)")

ax[2].set_xlabel("Frame")
ax[2].set_ylabel("y (nm)")
            
number_of_good_beads = len(x_list)
print("Number of completely tracked nanoparticles = ", number_of_good_beads)

x_array = np.array(x_list)
y_array = np.array(y_list)

pos_array = np.zeros((x_array.shape[0], x_array.shape[1], 2))

pos_array[:, :, 0] = x_array
pos_array[:, :, 1] = y_array

av_array = np.mean(pos_array, axis=0)

ax[0].plot(av_array[:, 0], linewidth=0.3, label='average of the positions of the beads')
ax[1].plot(av_array[:, 1], linewidth=0.3, label='average of the positions of the beads')

if save_data:

    d = {'x_av':list(av_array[:, 0]), 'y_av':list(av_array[:, 1])}
        
    df = pd.DataFrame(d)
    df.to_csv(directory +  r'/'+ 'av_pos_' + 'data.csv', sep='\t', encoding='utf-8')
    df.to_excel(directory + r'/' + 'av_pos_' + 'data.xlsx')      

x_smooth = moving_average(av_array[:,0], w)
ax[0].plot(x_smooth, linewidth=0.8, color='green', label='rolling time average - 100 ms')

y_smooth = moving_average(av_array[:,1], w)
ax[1].plot(y_smooth, linewidth=0.8, color='green', label='rolling time average - 100 ms')

if save_data:

    d = {'roll_time_av_x':list(x_smooth), 'roll_time_av_y':list(y_smooth)}
        
    df = pd.DataFrame(d)
    df.to_csv(directory +  r'/'+ 'roll_time_av_pos_' + 'data.csv', sep='\t', encoding='utf-8')
    df.to_excel(directory + r'/' + 'roll_time_av_pos_' + ' data.xlsx')    

dev_array = np.zeros((len(x_smooth), 2))

dev_array[:, 0] = av_array[:, 0][:-w+1] - x_smooth # smooth array is (w-1) smaller than original data array because of the moving average
dev_array[:, 1] = av_array[:, 1][:-w+1] - y_smooth

ax[0].plot(dev_array[:, 0], linewidth=0.8, markersize=1, color='black', label='drift corrected')
ax[1].plot(dev_array[:, 1], linewidth=0.8, markersize=1, color='black', label='drift corrected')

ax[0].legend()

plt.tight_layout()

if save_data:

    d = {'high_freq_noise_x':list(dev_array[:, 0]), 'high_freq_noise_y':list(dev_array[:, 1])}
        
    df = pd.DataFrame(d)
    df.to_csv(directory +  r'/'+ 'high_freq_noise_pos_' + 'data.csv', sep='\t', encoding='utf-8')
    df.to_excel(directory + r'/' + 'high_freq_noise_pos_' + 'data.xlsx')   

print("Mean of the deviation (x,y): ", np.mean(dev_array, axis = 0), "nm")
print("Standard deviation (of the deviation from the smoothed fit) (x,y): ", np.std(dev_array, axis = 0), "nm")

fig1, ax1 = plt.subplots()
ax1.hist(dev_array[:, 0], bins=400, alpha=0.5, label='x')
ax1.hist(dev_array[:, 1], bins=400, alpha=0.5, label='y')
ax1.set_ylabel('Counts')
ax1.set_xlabel('Position (nm)')
ax1.legend()

plt.tight_layout()

# Fourier transform

fmax = 1.0/(2*dt)
t = np.arange(n_frames) * dt
freq = np.linspace(0.0, fmax, n_frames//2)
df = freq[1] - freq[0]

# f1 = 20
# sim_signal = 3*np.sin(2*np.pi*f1*t) 

Y = np.fft.fft(dev_array[:, 0])
C = n_frames/2 * dt

fft_data_x = np.abs(Y[np.arange(n_frames // 2)]) / C # normalized fft

# cum_data = np.cumsum(fft_data_x) * df # cumulative noise

fig2, ax2 = plt.subplots()
ax2.plot(freq, fft_data_x, alpha=0.7, label='x') 
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Noise density (nm/Hz)')

# fig3, ax3 = plt.subplots()
# ax3.plot(freq, cum_data, alpha=0.7, label='x') 
# ax3.set_xlabel('Frequency')
# ax3.set_ylabel('Cumulative noise (nm)')

plt.tight_layout()

Y = np.fft.fft(dev_array[:, 1]) 

fft_data_y = np.abs(Y[np.arange(n_frames // 2)]) / C # fft normalized

# cum_data = np.cumsum(fft_data) * df # cumulative noise

ax2.plot(freq, fft_data_y, alpha=0.7, label='y') 
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Noise density (nm/Hz)')

ax2.legend()

# ax3.plot(freq, cum_data, alpha=0.7, label='y') 
# ax3.set_xlabel('Frequency (Hz)')
# ax3.set_ylabel('Cumulative noise (nm)')

# ax3.legend()

if save_data:

    d = {'fft_x':list(fft_data_x), 'fft_y':list(fft_data_y)}
        
    df = pd.DataFrame(d)
    df.to_csv(directory +  r'/'+ 'fft_x_' + 'data.csv', sep='\t', encoding='utf-8')
    df.to_excel(directory + r'/' + 'fft_y_' + 'data.xlsx')   

# plt.tight_layout()




import glob as glob
import numpy as np
import matplotlib.pyplot as plt
import os as os
import pandas as pd
import sys
import seaborn as sb
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
#import numba as numba
import tqdm
import time
from multiprocessing import Pool
from functools import partial


#@numba.autojit
def lowpass1_array(signal, tau):
    ''' first order low-pass '''
    N   = signal.shape[0]
    tau = float(tau)
    out = np.zeros(signal.shape)
    alpha = (tau / (tau + 1))

    out[0,:] = signal[0,:] # initial condition
    for i in np.arange(1,N):
        out[i,:] = signal[i,:] * (1. - alpha) + out[i-1,:] * alpha

    return out


def gen_tqdm_range(N):
    try:
        tqdm_range = tqdm.tqdm_notebook(range(N))
    except:
        tqdm_range = tqdm.tqdm(range(N))
    return tqdm_range



def get_1st_order_LP_baseline(data, fs, time_constant):

    pad_length     = 1
    pad_thingA     = np.ones(pad_length)*data.values[:10*fs].mean()  # mean of first 10 sec
    pad_thingB     = np.ones(pad_length)*data.values[-10*fs:].mean() # mean of last 10 sec

    filter_this  = np.concatenate((pad_thingA, data.values, pad_thingB))

    lp_filter     = lowpass1_array(filter_this[:,np.newaxis], time_constant*fs)[:,0]
    baseline      = pd.Series(lp_filter[pad_length:-pad_length], index = data.index)

    return baseline


def recover_BinaryNoise_DiscreteTime(duration, N_phi, N_z, delta_s):
    N_t = int(duration*60.0/delta_s)
    noise_shape = (N_t, N_z, N_phi)
    np.random.seed(0) # seed at zero, to make sure it will be always the same random numbers generated
    my_binary_noise = np.random.randint(0,2,noise_shape)
    return my_binary_noise


def recover_GaussNoise_General(duration, N_phi, N_z, tau):

    duration = duration
    N_t   = int(duration*60.0) # number of frames
    N_phi = N_phi
    N_z   = N_z
    noise_shape = (N_t, N_z, N_phi)

    tau = tau
    tau_fs = tau*60.0
    alpha = (tau_fs / (tau_fs + 1))
    scaling_f = np.sqrt((1-alpha)/(1+alpha))

    # re-generate underlying random numbers
    np.random.seed(0) # seed at zero, to make sure it will be always the same random numbers generated
    my_gauss_noise = np.random.normal(0, 1, noise_shape)

    # re-generate initial state
    np.random.seed(0) # seed at zero, to make sure it will be always the same random numbers generated
    init_state = np.random.normal(0, scaling_f, noise_shape[1:])
    my_gauss_noise = np.concatenate((init_state[np.newaxis,:,:], my_gauss_noise), 0)

    # recover_low-pass_filtered noise
    my_gauss_noise = lowpass1_array(my_gauss_noise, tau_fs)/scaling_f
    my_gauss_noise = my_gauss_noise[1:,:,:]

    # remap to arena luminance values
    my_gauss_noise[my_gauss_noise<-2] = -2
    my_gauss_noise[my_gauss_noise>2]  = 2
    my_gauss_noise = (my_gauss_noise + 2)*255/4.
    my_gauss_noise = my_gauss_noise.astype(np.uint8) # simulate discretization by arena screen

    return my_gauss_noise



def reverse_correlation_time_single_core(data_df, noise_stimulus, fs_stimulus, T_past, T_futu):
    # data_df hast to be a pandas DataFrame with only the time as index and the signals (from differents cells/ROIs) as columns

    time_axis = data_df.index.values
    min_t     = np.floor((T_past*fs_stimulus)).astype(np.int)/fs_stimulus # minimum time so that there is enough time "in the past"
    max_t     = np.floor((noise_stimulus.shape[0]/fs_stimulus - T_futu)*fs_stimulus).astype(np.int)/fs_stimulus # maximum time so that there is enough data "in the future"

    select_mask = np.logical_and(time_axis > min_t, time_axis < max_t)
    time_axis   = time_axis[select_mask] # restrict time axis
    data_values = data_df.values[select_mask,:]

    S_futu = int(T_futu*fs_stimulus)
    S_past = int(T_past*fs_stimulus)
    K      = np.zeros((S_futu+S_past,)+ noise_stimulus.shape[1:] + (data_values.shape[1],))# initialize kernel
    #print S_futu, S_past

    #for i in range(len(time_axis)):
    for i in gen_tqdm_range(len(time_axis)):
        t = time_axis[i]
        frame_in_stimulus = int(t*fs_stimulus)

        history = noise_stimulus[frame_in_stimulus-S_past:frame_in_stimulus+S_futu, :, :]

        K += history[:,:,:,np.newaxis] * data_values[i,:]

    K_raw  = K/float(len(time_axis))
    K_norm = (K-K.mean())/K.std()

    return K_raw, K_norm




# define MULTI CORE LOOP function
def revcorr_loop_multicore(partial_index_range, time_axis, fs_stimulus, noise_stimulus, data_values, S_past, S_futu):
    K_sub      = np.zeros((S_futu+S_past,)+ noise_stimulus.shape[1:] + (data_values.shape[1],)) # initialize kernel

    #for i in range(len(partial_time_axis)):
    #for i in gen_tqdm_range(len(partial_time_axis)):
    for i in partial_index_range:
        t = time_axis[i]
        frame_in_stimulus = int(t*fs_stimulus)

        history = noise_stimulus[frame_in_stimulus-S_past:frame_in_stimulus+S_futu, :, :]

        K_sub += history[:,:,:,np.newaxis] * data_values[i,:]

    return K_sub


def reverse_correlation_time_multi_core(data_df, noise_stimulus, fs_stimulus, T_past, T_futu, N_pool = -1, N_chunks = 100):

    time_axis = data_df.index.values
    min_t     = np.floor((T_past*fs_stimulus)).astype(np.int)/fs_stimulus # minimum time so that there is enough time "in the past"
    max_t     = np.floor((noise_stimulus.shape[0]/fs_stimulus - T_futu)*fs_stimulus).astype(np.int)/fs_stimulus # maximum time so that there is enough data "in the future"
    #print min_t, max_t

    select_mask = np.logical_and(time_axis > min_t, time_axis < max_t)
    time_axis   = time_axis[select_mask] # restrict time axis
    data_values = data_df.values[select_mask,:]

    S_futu = int(T_futu*fs_stimulus)
    S_past = int(T_past*fs_stimulus)

    ##################################################
    # compute reverse correlation using multiple cores

    time_chunks = np.array_split(range(len(time_axis)), N_chunks) # split time axis into N_chunks chunks

    # initialize Pool
    if N_pool == -1:
        pool = Pool()
    else:
        pool = Pool(processes = N_pool)

    # check if this is a python or an ipython script
    try:
        #tqdm.tqdm_notebook(range(0))
        __IPYTHON__ # checks if in ipython actually
        notebook_script = True
    except:
        notebook_script = False

    # define partial function, that takes only the time axis chunks as argument
    loop_function = partial(revcorr_loop_multicore, time_axis = time_axis, noise_stimulus = noise_stimulus, fs_stimulus = fs_stimulus, data_values = data_values, S_past = S_past, S_futu = S_futu)

    # calculate the reverse correlation function for each chunk on the different cores
    if notebook_script:
        pool_iterator = tqdm.tqdm_notebook(pool.imap(loop_function, time_chunks), total=N_chunks)
    else:
        pool_iterator = tqdm.tqdm(pool.imap(loop_function, time_chunks), total=N_chunks)

    K_sub_list = list(pool_iterator) # merge results

    # sum results up and get z_score curve
    K      = np.zeros((S_futu+S_past,)+ noise_stimulus.shape[1:] + (data_values.shape[1],)) # initialize kernel
    z_score_list = []
    for i in range(len(K_sub_list)):
        K += K_sub_list[i]
        z_score_list.append( np.abs( (K - K.mean())/K.std() ).max() )

    K_raw  = K/float(len(time_axis)) # divide by number of time steps
    K_norm = (K-K.mean())/K.std()    # normalize to z-score


    return K_raw, K_norm, z_score_list


#### THE FOLLOWING TWO FUNCTIONs ARE HEAVILY ADAPTED TO THE SPECIFIC ARENA SYSTEM (180 degree cylindric arena, 13 cm height)
def pre_analyse_rev_corr_2D(R, printMode = True, already_normalized = False):

    '''
    takes an array 4-dim array R (x,y,time,ROI#) containing the reverse correlation output and do first analysis:
    1. normalize to z-score
    2. find extremum
    3. fit gaussian and find center with subpixel precision

    returns
    R_normalized,  maxima, maxima_i, maxima_j, maxima_t, centers_i, centers_j, sigma_i, sigma_j, a_i, a_j

    R_normalized: the input receptive fields normalized in z-score
    maxima: the extreme (min or max) value within each RF
    maxima_i: the i-coordinate of the maxima
    maxima_j: the j-coordinate of the maxima
    maxima_t: the time-coordinate of the maxima
    centers_i: the subpixel precision center of the fitted gaussian in the first dimension
    centers_j: the subpixel precision center of the fitted gaussian in the second dimension
    sigma_i: the standard deviation of the fitted gaussian in the first dimension
    sigma_j: the standard deviation of the fitted gaussian in the second dimension
    a_i: the amplitude of the fitted gaussian in the first dimension
    a_j: the amplitude of the fitted gaussian in the second dimension

    NOTE: fitting both dimensions seperately seemed to be a more robust method to locate the center instead of fitting
    a two-dimensional Gaussian which was more sensitive to noise in the neighbouring pixels
    '''

    # normalize receptive field to z-score and find maxima
    R_normalized = np.zeros(R.shape)
    maxima_i = np.zeros(R.shape[3])
    maxima_j = np.zeros(R.shape[3])
    maxima_t = np.zeros(R.shape[3])
    maxima   = np.zeros(R.shape[3])

    ind = 0
    for ind in range(R.shape[3]):
        R_temp = R[:,:,:,ind]
        if already_normalized:
            pass
        else:
            R_temp = (R_temp - R_temp.mean())/R_temp.std()
        R_normalized[:,:,:,ind] = R_temp

        minimum = np.absolute(R_temp).argmax()
        i,j,z = np.unravel_index(minimum, R_temp.shape)
        max_value = R_temp[i,j,z]

        maxima_i[ind] = i
        maxima_j[ind] = j
        maxima_t[ind] = z
        maxima[ind]   = max_value


    # fit gaussian and find center of RF
    from scipy.optimize import curve_fit
    def func(x, a, x0, sigma): # gauss distribution
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    centers_i = np.zeros(R.shape[3])
    centers_j = np.zeros(R.shape[3])
    a_i = np.zeros(R.shape[3])
    a_j = np.zeros(R.shape[3])
    sigma_i = np.zeros(R.shape[3])
    sigma_j = np.zeros(R.shape[3])
    for i in range(len(maxima_i)):

        # fit gauss in i-direction
        ydata = R_normalized[:,int(round(maxima_j[i])),int(round(maxima_t[i])),i]
        ydata = ydata - ydata.mean()

        xdata = np.arange(len(ydata))
        try:
            popt, pcov = curve_fit(func, xdata, ydata, p0 = (maxima[i], maxima_i[i], 1.), maxfev = 2000)
            centers_i[i] = popt[1]
            a_i[i] = popt[0]
            sigma_i[i] = popt[2]
        except:
            if printMode:
                print("WARNING: Gaussian fit for ROI #" + str(int(i)) + " failed.")
            else:
                pass

        # fit gauss in j-direction
        ydata = R_normalized[int(round(maxima_i[i])),:,int(round(maxima_t[i])),i]
        ydata = ydata - ydata.mean()
        xdata = np.arange(len(ydata))
        try:
            popt, pcov = curve_fit(func, xdata, ydata, p0 = (maxima[i], maxima_j[i], 1.), maxfev = 2000)
            centers_j[i] = popt[1]
            a_j[i] = popt[0]
            sigma_j[i] = popt[2]
        except:
            if printMode:
                print("WARNING: Gaussian fit for ROI #" + str(int(i)) + " failed.")
            else:
                pass


    return R_normalized, maxima, maxima_i, maxima_j, maxima_t, centers_i, centers_j, sigma_i, sigma_j, a_i, a_j



def plot_pre_analysis_2D(R_normalized, mean_image, roi_masks, maxima, maxima_i, maxima_j, maxima_z, centers_i, centers_j, sigma_i, sigma_j, a_i, a_j, roi_bins = [], cmap = 'coolwarm'):

    for ind_ROI in range(R_normalized.shape[3]):

        with sb.axes_style("white"), sb.axes_style("ticks"):
            fig = plt.figure(figsize = (10,5))
            gs = gridspec.GridSpec(2,2)
            ax = plt.subplot(gs[0:2,0])

        #plt.imshow(np.rot90(np.flipud(R_normalized[:,:,maxima_z[ind_ROI],ind_ROI])), cmap = plt.get_cmap('coolwarm'), interpolation = 'nearest', origin = 'lower', vmin = -np.abs(maxima[ind_ROI]), vmax = np.abs(maxima[ind_ROI]))
        plt.imshow(np.rot90(np.flipud(np.fliplr(R_normalized[:,:,int(round(maxima_z[ind_ROI])),ind_ROI]))), cmap = plt.get_cmap(cmap), interpolation = 'nearest', origin = 'lower', vmin = -np.abs(maxima[ind_ROI]), vmax = np.abs(maxima[ind_ROI]))
        #ax.axhline(R_normalized.shape[1] - 1 - centers_j[ind_ROI], color = 'k', linewidth = 1, linestyle = '--')
        ax.axhline(0.5 + centers_j[ind_ROI], color = 'k', linewidth = 1, linestyle = '--')
        ax.axvline(R_normalized.shape[0] - 0.5 - centers_i[ind_ROI], color = 'k', linewidth = 1, linestyle = '--')

        # azimuth
        ax.plot(np.arange(R_normalized.shape[0]), np.flipud(R_normalized[:,int(round(maxima_j[ind_ROI])),int(round(maxima_z[ind_ROI])),ind_ROI]), color = 'r')
        ax.plot(np.arange(0, R_normalized.shape[0]), np.flipud(func_gaussian(np.arange(0, R_normalized.shape[0]), a_i[ind_ROI], centers_i[ind_ROI], sigma_i[ind_ROI])),  color = 'k')

        # elevation
        #ax.plot(np.flipud(R_normalized[maxima_i[ind_ROI],:,maxima_z[ind_ROI],ind_ROI]), np.arange(R_normalized.shape[1]),  color = 'r')
        #ax.plot(np.flipud(func_gaussian(np.arange(0, R_normalized.shape[1]),  a_j[ind_ROI], centers_j[ind_ROI], sigma_j[ind_ROI])), np.arange(0, R_normalized.shape[1]), color = 'k')
        ax.plot(R_normalized[int(round(maxima_i[ind_ROI])),:,int(round(maxima_z[ind_ROI])),ind_ROI], np.arange(R_normalized.shape[1]),  color = 'r')
        ax.plot(func_gaussian(np.arange(0, R_normalized.shape[1]),  a_j[ind_ROI], centers_j[ind_ROI], sigma_j[ind_ROI]), np.arange(0, R_normalized.shape[1]), color = 'k')


        #
        phi = (R_normalized.shape[0] - 1 - centers_i[ind_ROI])*(180./R_normalized.shape[0]) + 180./R_normalized.shape[0]/2.
        z   = -1 * ((R_normalized.shape[1] - 1 - centers_j[ind_ROI])*(13./R_normalized.shape[1]) + 13./R_normalized.shape[1]/2. - 6.5)

        ax.text(0.65,0.85, 'phi = ' + str(np.round(phi,2)), transform = ax.transAxes, ha = 'left', va = 'top', fontsize = 'large')
        ax.text(0.65,0.78, 'z   = ' + str(np.round(z,2)), transform = ax.transAxes, ha = 'left', va = 'top', fontsize = 'large')


        ax.set_xticks([0,R_normalized.shape[0]/2.,R_normalized.shape[0]])
        ax.set_yticks([0,R_normalized.shape[1]/2.,R_normalized.shape[1]])
        ax.set_xticklabels([0,90,180], fontsize = 'large')
        ax.set_yticklabels([-6.5,0,6.5], fontsize = 'large')
        ax.set_xlabel('azimuth [deg]')
        ax.set_ylabel('elevation [cm]')
        sb.despine(ax = ax)
        ax.set_xlim([-5,R_normalized.shape[0]+5])
        ax.set_ylim([-5,R_normalized.shape[1]+5])

        if roi_bins:
            bin_txt = '\nfile: ' + roi_bins[ind_ROI]
        else:
            bin_txt = '\nfile: '
        ax.set_title('ind_ROI = ' + str(int(ind_ROI)) + bin_txt)


        with sb.axes_style("white"):
            ax = plt.subplot(gs[0,1])
            try:
                ax.imshow(mean_image, cmap = plt.get_cmap('Greys_r'), vmin = 0, vmax = 0.5, interpolation = 'nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                ROI = roi_masks[:,:,ind_ROI]
                ROI = ROI.astype(np.bool)
                ROI_inner = (np.roll(ROI,1,1) & np.roll(ROI,-1,1) & np.roll(ROI,1,0) & np.roll(ROI,-1,0))
                ROI[ROI_inner] = False
                ROI_color = np.dstack((np.zeros(ROI.shape),np.zeros(ROI.shape),np.zeros(ROI.shape),np.zeros(ROI.shape)))
                ROI_color[ROI, :] = [1,0,0,1]
                ax.imshow(ROI_color, interpolation = 'nearest')
            except:
                print("WARNING: No mean image or ROI mask given.")

        with sb.axes_style("white"), sb.axes_style("ticks"):
            ax = plt.subplot(gs[1,1])
            plt.plot(R_normalized[int(round(maxima_i[ind_ROI])),int(round(maxima_j[ind_ROI])),:,ind_ROI], 'k')
            ax.set_xlim([0,R_normalized.shape[2]])
            ax.set_xticks(np.arange(0,R_normalized.shape[2]+60,60))
            ax.set_xticklabels(np.arange(0,R_normalized.shape[2]+60,60)/60., fontsize = 'large')
            sb.despine(ax = ax)
            ax.set_xlabel('time [s]', fontsize = 'large')
            ax.set_ylabel('z-score', fontsize = 'large')
            plt.yticks(fontsize = 'large')
            ax.text(0.10,0.9, 'max   = ' + str(np.round(maxima[ind_ROI],2)), transform = ax.transAxes, ha = 'left', va = 'top', fontsize = 'large')
            ax.axhline(0.0, color = 'k', linestyle = '--')




def func_gaussian(x, a, x0, sigma): # gauss distribution
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
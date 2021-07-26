""" Provide any additional functionality for setiml demo.

This module is currently not meant to be run as a script.

Python Version
--------------
Requires Python 3
    Tested with Python 3.6.5


Authors
-------
|    Paul Pinchuk (ppinchuk@physics.ucla.edu)


Jean-Luc Margot UCLA SETI Group.
University of California, Los Angeles.
Copyright 2021. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt


def show_batch(batch, plot_shape, snr_thresh=None):
    image_batch = batch['IMAGES']
    labels = batch['LABEL'].numpy()
    ids = batch['ID'].numpy()
    oids  = batch['OTHER_ID'].numpy()
    shifts = batch['SHIFT'].numpy()
    p_rows, p_cols = plot_shape
    
    plt.figure(figsize=(12, 12))
    for n in range(0, p_rows*p_cols//2):
        ax = plt.subplot(p_rows, p_cols, n+1+np.floor(n/p_cols)* 4)
        
        if snr_thresh is None:
            snr_thresh = np.inf
        elif isinstance(snr_thresh, bool):
            if snr_thresh:
                snr_thresh = image_batch[n, :, :, 0].mean() + 3 * image_batch[n, :, :, 0].std()  # set to 3*std
            else:
                snr_thresh = np.inf
        
        plt.imshow(
            np.minimum(image_batch[n, :, :, 0], snr_thresh), 
            aspect='auto', cmap='gray'
        )
        plt.axis('off')
        
        plt.title(f"LABEL: {labels[n]}")
        
        ax = plt.subplot(p_rows, p_cols, n+1+np.floor(n/p_cols)*4+p_cols)
        
        if snr_thresh is None:
            snr_thresh = np.inf
        elif isinstance(snr_thresh, bool):
            if snr_thresh:
                snr_thresh = image_batch[n, :, :, 1].mean() + 3 * image_batch[n, :, :, 1].std()  # set to 3*std
            else:
                snr_thresh = np.inf
        
        plt.imshow(
            np.minimum(image_batch[n, :, :, 1], snr_thresh), 
            aspect='auto', cmap='gray'
        )
        plt.axis('off')


def _shift_and_calc_scores(vdata, shift, model):
    data = vdata.copy()
    data[:, :, :, 1] = np.roll(data[:, :, :, 1], shift)
    
    return model.predict(
        data,
        verbose=0,
        batch_size=None
    )
    
    
def __calc_scores(val_data, model, shifts=None):
    shifts = shifts or range(-10, 11)
    scores = [_shift_and_calc_scores(val_data, shift, model) for shift in shifts]
    scores = np.r_[scores][:, :, 0].T
    return np.max(scores, axis=1), np.argmax(scores, axis=1)


def calculate_scores(val_data, model, snr_mult=None, shifts=None, return_inds=False):
    inds_fixed_by_snr = []
    scores, shift_inds = __calc_scores(val_data, model, shifts=shifts)
    
    recheck_inds = np.where(scores < 0.5)[0]
    if recheck_inds.shape[0] > 0:
        recheck_data = val_data[recheck_inds].copy()
        if snr_mult is None:
            mults = recheck_data.max(axis=(1, 2)).mean(axis=1)
        else:
            mults = snr_mult
        recheck_data[:, :, :, 0] = (recheck_data[:, :, :, 0].T - recheck_data[:, :, :, 0].min(axis=(1, 2))).T
        recheck_data[:, :, :, 1] = (recheck_data[:, :, :, 1].T - recheck_data[:, :, :, 1].min(axis=(1, 2))).T

        recheck_data[:, :, :, 0] = (recheck_data[:, :, :, 0].T / recheck_data[:, :, :, 0].max(axis=(1, 2))).T
        recheck_data[:, :, :, 1] = (recheck_data[:, :, :, 1].T / recheck_data[:, :, :, 1].max(axis=(1, 2))).T
        recheck_data = (recheck_data.T * mults).T
        
        snr_scores, snr_shift_inds = __calc_scores(recheck_data, model, shifts=shifts)
        
        for ind, new_score, new_shift in zip(recheck_inds, snr_scores, snr_shift_inds):
            if new_score > scores[ind]:
                scores[ind] = new_score
                shift_inds[ind] = new_shift
                inds_fixed_by_snr.append(ind)
    if return_inds:
        return scores, shift_inds, inds_fixed_by_snr
    else:
        return scores

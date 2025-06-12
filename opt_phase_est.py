#!/bin/python3

import os
import sys
import h5py as h5
import math
import shutil
import numpy as np
from scipy.stats import chi2
from multiprocessing import Pool
from numba import njit
from concurrent.futures import ProcessPoolExecutor

'''
Optimal Phase Estimation Script

This script performs optimal interferometric phase estimation by 
integrating homogeneous multi-looking and EMI-based phase linking, 
implemented with a patch-based processing strategy. The selection of 
statistically homogeneous pixels (SHP) is based on a local phase similarity 
statistic [1], while phase linking is achieved using the EMI method [2].

Input File:
inputs_fln      An HDF5 file containing the following datasets:
                <dates>:
                    Array of acquisition dates for n SAR scenes.
                    Shape: (n,)
                    Example: array([20201205, 20210106, ..., 20210904])
                <imgSize>:
                    Size of the SAR image [w(idth), l(ength)].
                    Shape: (2,)
                <slc>:
                    Single-look complex (SLC) data.
                    Shape: (n, w * l)
                <diffPhase>:
                    Single-master interferometric phase expressed as exp(j*\phi) 
                    Shape: (n, w * l)

⸻
Parameters:
alpha_phs       (default: 0.2) significance level (\alpha) for selecting SHP.
rng_win         (default: 9) window size in range for local pixel selection.
azi_win         (default: 9) window size in azimuth for local pixel selection.
ifg_cnsc_num    (default: 5) sequentially connected number to reformulate
                interferometric phases for assessing phase similarity.
patch_size      (default: 200) Patch size for block-wise processing.
crtShpNum       (default: 20) critical number of homogeneous pixels to designate 
                as a DS.
num_cores       (default: 4) Number of CPU cores used for parallel computation.
⸻

An HDF5 file named "pl_prods_overview.h5" is generated containing the following datasets:
<linkedPhase>           EMI-linked interferometric phase.
                        Shape: (n, w * l)
<gammaPta>              posterior coherence for each pixel.
                        Shape: (w * l,)
<shpNum>                Number of statistically homogeneous pixels associated with each target pixel.
                        Shape: (w * l,)
<psMask>                Boolean mask identifying PS.
                        Shape: (w * l,)
<dates>                 Repeated input date array.
                        Shape: (n,)
<imgSize>               Image size as provided in input.
                        Shape: (2,)

Contact:
For any inquiries, please contact: guoqiang.shi@polyu.edu.hk

'''


# parameter setting panel
inputs_fln = 'inputs.h5'
out_fln = "phaseLink.h5"
rng_win = 9
azi_win = 9
alpha_phs = 0.2
ifg_cnsc_num = 5

patch_size = 200
crtShpNum = 20
num_cores = 20

# functions
def get_box(centr, box_span, width, length):
  '''
  get box coordinates (XY) of a window under centroid
  '''
  box = np.tile(centr, 2) + box_span
  
  if box[0] <= 0:
      box[0] = 0
  if box[1] <= 0:
      box[1] = 0
  if box[2] > width - 1:
      box[2] = width - 1
  if box[3] > length - 1:
      box[3] = length - 1
  
  box_w_l = (box[2] - box[0] + 1,
             box[3] - box[1] + 1)
  return box, box_w_l


def consecutive_interferogram_indices(sceneNum, consecNum):
  """
  Construct indices of <n> consecutive inferograms from <sceneNum> SLCs
  """
  
  # Initial indices of all interf. pairs
  m_indices, s_indices = np.triu_indices(sceneNum, k=1)

  # Redundant index pairs to be removed
  m_indices_rdn, s_indices_rdn = np.triu_indices(sceneNum, k=consecNum+1)


  # Create a mask that filters out the pairs to remove
  rdn_indices_set = set(zip(m_indices_rdn, s_indices_rdn))
  mask = [(i, j) not in rdn_indices_set for i, j in zip(m_indices, s_indices)]

  return m_indices[mask], s_indices[mask]


def get_xy_coords(x_vec, y_vec, snx, sny):
  '''
  Function: return the coordinates of a 2d array
  Variables:
  x_vec = x coordinates in vector
  y_vec = y coordinates in vector
  snx = sample number in x direction
  sny = sample number of y direction
  '''
  return np.vstack((np.tile(x_vec, sny), np.repeat(y_vec, snx)))

@njit
def test_non_uniformity_nb(dirc_dt):
  """
  Implement rayleigh test to cluster pixels being non-uniformity under von mises distribution 
  """
  num_obs, num_pts = dirc_dt.shape
  
  # Calculate the mean resultant vector
  z_bar = np.empty(num_pts, dtype=np.complex128)
  for i in range(num_pts):
      tmp_sum = 0
      for j in range(num_obs):
          tmp_sum += dirc_dt[j, i]
      z_bar[i] = tmp_sum / num_obs
  
  # Calculate the mean direction (mu_hat)
  mu_hat = np.angle(z_bar)
  
  # Calculate the resultant length (R_hat)
  R_hat = np.abs(z_bar)
  
  var_hat = 1 - R_hat

  R_bar2 = R_hat**2
  
  rayleigh_statistic = R_bar2 * num_obs * 2 
  return rayleigh_statistic, var_hat, mu_hat


@njit
def idf_shp_phs_sim_nb(wnSmIfg, wnInd, ind_ifg_m, ind_ifg_s, criticalValue):
  
  wnCnsIfg = wnSmIfg[ind_ifg_s, :] * wnSmIfg[ind_ifg_m, :].conj()
  
  wnDphsRef = wnCnsIfg[:, 0].ravel()
  wnDltDphs = wnCnsIfg[:, 1:] * wnDphsRef[:, np.newaxis].conj()
  
  stats_phs, var_hat, mu_hat = test_non_uniformity_nb(wnDltDphs)
  shpPhs = stats_phs >= criticalValue
  
  sps = np.zeros(wnSmIfg.shape[1], dtype=np.bool)
  sps[1:] = shpPhs; sps[0] = True
  
  shpInd = wnInd[sps]
  shpNum = len(shpInd)

  # return shpInd, shpNum, mu_hat[shpPhs], var_hat[shpPhs], kappa_hat[shpPhs]
  return shpInd, shpNum, var_hat[shpPhs], mu_hat[shpPhs]



@njit(fastmath=True)
def cal_weight1_numba(mu, var):
    R_bar = 1 - var
    term = 2 * R_bar * (np.sin(mu / 2))**2
    D = var + term
    return 1 / D

@njit
def detect_zero_columns(arr):
  num_cols = arr.shape[1]
  mask = np.ones(num_cols, dtype=np.bool_)  

  for j in range(num_cols):
      for i in range(arr.shape[0]):
          if arr[i, j] != 0:
              mask[j] = False  
              break
  
  return mask


def search_shp_phs(pxl, args):
    # prepare window indices
    pPhs, boxSpan, pxy0, pxy1, poff, pind, pw1, pl1, cv, ind_ifg_m, ind_ifg_s = args

    xyoff = pxy0[:, pxl] + poff[0:2]
    wn_bx, bwl = get_box(xyoff, boxSpan, pw1, pl1)

    wn_ind = pind[wn_bx[1] : wn_bx[3] + 1, 
                wn_bx[0] : wn_bx[2] + 1].ravel()
    wn_ind_ref = pind[xyoff[1], xyoff[0]]

    tl_crn = pxy1[0, wn_ind].min(), pxy1[1, wn_ind].min()
    x_ref, y_ref = xyoff[0] - tl_crn[0], xyoff[1] - tl_crn[1]
    indRefInWn = y_ref * bwl[0] + x_ref

    wnInd = np.concatenate(([wn_ind[indRefInWn]], wn_ind[:indRefInWn], wn_ind[indRefInWn+1:]))

    samp_phs = pPhs[:, wnInd]
    mask0 = detect_zero_columns(samp_phs)
    wnInd1 = wnInd[~mask0]
    samp_phs1 = pPhs[:, wnInd1]
  
    if mask0[0]:
        return wn_ind_ref, 1, 1
    else:  
        shpInd, shpNum, var_hat, mu_hat = idf_shp_phs_sim_nb(samp_phs1, wnInd1, ind_ifg_m, ind_ifg_s, cv)
        if shpNum != 1:
            vmVar_min = np.min(var_hat)
            vmVar = np.insert(var_hat, 0, vmVar_min)
            vmMu = np.insert(mu_hat, 0, 0)

            weights = cal_weight1_numba(vmMu, vmVar)

        else:
            weights = 1

        # if pxl == 31:
        #     print("plx == 31")
        return shpInd, shpNum, weights


def patch_slice(width, length, patch_size=200):
    """
    Slice the image into patches of size patch_size
    box: (x0 y0 x1 y1) = (left, top, right, bottom) for each patch with respect to the whole image
    Returns box list, number of boxes
    """
    patchAzTop = np.arange(0, length, patch_size, dtype='int64')
    patchAzBtm = np.arange(patch_size - 1, length, patch_size, dtype='int64')
    patchRgLft = np.arange(0, width, patch_size, dtype='int64')
    patchRgRht = np.arange(patch_size - 1, width, patch_size, dtype='int64')

    if patchAzBtm.size == patchAzTop.size - 1:
        patchAzBtm = np.append(patchAzBtm, length - 1)
    if patchRgRht.size == patchRgLft.size - 1:
        patchRgRht = np.append(patchRgRht, width - 1)

    np_az = patchAzTop.size
    np_rg = patchRgLft.size
    num_patches = np_az * np_rg

    ind_az = np.repeat(np.arange(0, np_az), np_rg)
    ind_rg = np.tile(np.arange(0, np_rg), np_az)

    lft = patchRgLft[ind_rg]
    rgt = patchRgRht[ind_rg]
    top = patchAzTop[ind_az]
    btm = patchAzBtm[ind_az]
    patch_list = np.stack((lft, top, rgt, btm), axis=0).T

    print(f'{num_patches} patches created')

    return patch_list, num_patches

def get_big_box(patch_corners, range_radius, azimuth_radius, width, nlines):
    patch_corners_el = np.arange(4, dtype=np.int64)
    patch_corners_el[0] = patch_corners[0] - range_radius
    patch_corners_el[1] = patch_corners[1] - azimuth_radius
    patch_corners_el[2] = patch_corners[2] + range_radius
    patch_corners_el[3] = patch_corners[3] + azimuth_radius

    if patch_corners_el[0] <= 0:
        patch_corners_el[0] = 0
    if patch_corners_el[1] <= 0:
        patch_corners_el[1] = 0
    if patch_corners_el[2] > width - 1:
        patch_corners_el[2] = width - 1
    if patch_corners_el[3] > nlines - 1:
        patch_corners_el[3] = nlines - 1

    return patch_corners_el

def display_progress_bar(title, current, total, bar_length=None, update_interval=0.01):
    """
    Displays a progress bar in the console.

    Parameters:
    - od: Output directory string to be displayed on the first line.
    - current: Current iteration number.
    - total: Total number of iterations.
    - bar_length: The length of the progress bar in characters. If None, it adapts to terminal width.
    """
    # Get the terminal window width if bar_length is not provided
    if bar_length is None:
        terminal_width = shutil.get_terminal_size().columns
        # Reserve some space for the percentage and extra text
        bar_length = terminal_width - 20
        # Ensure the bar length is not negative
        bar_length = max(10, bar_length)

    # Calculate the progress
    progress = current / total
    block = int(round(bar_length * progress))
    
    # Construct progress bar string
    progress_bar = f"{title}: [{'>' * block}{'.' * (bar_length - block)}] {progress * 100:.1f}%"   
    # Print the output directory and progress bar only at intervals
    if current % max(1, int(total * update_interval)) == 0:
        print(f"\r{progress_bar}", end='', flush=True)


@njit
def outer_product(vec, dim):
    out = np.empty((dim, dim), dtype=vec.dtype)
    for i in range(dim):
        for j in range(dim):
            out[i, j] = vec[i] * vec[j]
    return out

@njit
def outer_product_cpx(cpx_vec, dim):
    out = np.zeros((dim, dim), dtype=cpx_vec.dtype)
    for i in range(dim):
        for j in range(dim):
            out[i, j] = cpx_vec[i] * np.conj(cpx_vec[j])
    return out

@njit
def cal_wcorr_mat_cpx(cpx_vec, w, dim0, dim1):
    mat0 = np.zeros((dim0, dim1), dtype=cpx_vec.dtype)
    mat1 = np.zeros((dim0, dim0), dtype=cpx_vec.dtype)
    for i in range(dim0):
        for j in range(dim1):
            mat0[i, j] =  cpx_vec[i, j] * w[j]

    for i in range(dim0):
        for j in range(dim0):
            for k in range(dim1):
                mat1[i, j] += mat0[i, k] * np.conj(cpx_vec[j, k])

    return mat1

@njit
def cal_weighted_sample_corr_matrix_numba(pxAmpShp, pxDphShp, weights):
    dim0, dim1 = pxAmpShp.shape

    # denominatorNorm = np.zeros((dim0, dim0), dtype=np.float64)
    sampvec = pxAmpShp * pxDphShp
    
    scm0 = cal_wcorr_mat_cpx(sampvec, weights, dim0, dim1)
    
    pwrdiag = np.ones(dim0, dtype=pxAmpShp.dtype)
    for i in range(dim0):
        tmp = np.abs(scm0[i, i])
        pwrdiag[i] = tmp
    
    denominatorNorm0 = outer_product(pwrdiag, dim0)
    denominatorNorm = np.sqrt(denominatorNorm0)
    scm = scm0 / denominatorNorm

    return scm

@njit
def cal_sample_corr_matrix_numba(pxAmpShp, pxDphShp):
    dim0 = pxAmpShp.shape[0]

    # denominatorNorm = np.zeros((dim0, dim0), dtype=np.float64)
    sampvec = pxAmpShp * pxDphShp
    
    scm0 = sampvec @  sampvec.conj().T
    
    pwrdiag = np.ones(dim0, dtype=pxAmpShp.dtype)
    for i in range(dim0):
        tmp = np.abs(scm0[i, i])
        pwrdiag[i] = tmp
    
    denominatorNorm0 = outer_product(pwrdiag, dim0)
    denominatorNorm = np.sqrt(denominatorNorm0)
    scm = scm0 / denominatorNorm

    return scm

# @njit
def cal_sample_corr_matrix_1d(pxAmpShp, pxDphShp):
    dim0 = pxAmpShp.shape[0]

    # denominatorNorm = np.zeros((dim0, dim0), dtype=np.float64)
    sampvec = pxAmpShp * pxDphShp
    
    scm0 = sampvec @  sampvec.conj().T
    # scm0 = np.zeros((dim0, dim0), dtype=sampvec.dtype)
    # for i in range(dim0):
    #     for j in range(dim0):
    #         scm0[i, j] = sampvec[i, 0] * np.conj(sampvec[j, 0])
    
    pwrdiag = np.ones(dim0, dtype=pxAmpShp.dtype)
    for i in range(dim0):
        tmp = np.abs(scm0[i, i])
        pwrdiag[i] = tmp
    
    denominatorNorm0 = outer_product(pwrdiag, dim0)
    denominatorNorm = np.sqrt(denominatorNorm0)
    scm = scm0 / denominatorNorm

    return scm

@njit
def est_gamma_pta_nb(phase_series, scm):
    '''
    <Description>:
    Estimate temporal coherence (gamma PTA) from optimized phase series 
    and observed differential phase.
    <Description>
    
    <input>:
    phase_series (1D complex vec): linked phase 
    dph_obs (complex matrix): observation of difference phase of all interferogram pairs
    <input> 
    '''
    num_sce = len(phase_series)    
    dph_opt = outer_product_cpx(phase_series, num_sce)
    diff = scm * np.conj(dph_opt)
    angles = np.angle(diff)
    distance = np.cos(angles)    
    
    tmp = 0.0
    for i in range(num_sce):
        for j in range(i):
            tmp += distance[i, j]
    
    temp_coh = tmp * 2 / (num_sce**2 - num_sce)
    
    if temp_coh < 0:
        temp_coh = 0
    return temp_coh

@njit
def est_phase_series_EMI_nb(scm, coh_abs, ref_sce=0):
    coh_hat_inv = np.linalg.inv(coh_abs)
    ci_odot_scm = coh_hat_inv * scm
    
    # Compute eigenvalues and eigenvectors using cheevd
    egn_val, egn_vec = np.linalg.eigh(ci_odot_scm)
    
    # compute estimate of phase series referred at the image `ref_img`
    zeta = egn_vec[:, 0]
    phase_series = zeta.copy()
    
    temp_coh = est_gamma_pta_nb(phase_series, scm)
    
    return phase_series*np.conj(phase_series[ref_sce]), temp_coh


@njit
def is_pos_def_nb(M):
    """
    Check if a matrix is positive definite (PD) using eigenvalues.

    Parameters:
    ----------
    M : 2D NumPy array of float64 or complex128
        The input matrix to check.

    Returns:
    -------
    bool
        True if M is positive definite, False otherwise.
    """
    eigenvalues = np.linalg.eigvalsh(M)
    return eigenvalues[0] > 0  # Strictly positive for PD

@njit
def regularize_matrix_nb(M, inc):
    """
    Regularizes a matrix to make it positive semi-definite (PSD) by iteratively
    adding a value to its diagonal elements until all eigenvalues are non-negative.

    Parameters:
    ----------
    M : 2D NumPy array of float64 or complex128
        The input matrix to regularize.
    inc : float64 or complex128
        The initial increment value to add to the diagonal.

    Returns:
    -------
    N : 2D NumPy array of the same dtype as M
        The regularized positive semi-definite matrix.
    """
    # Create a deep copy to avoid modifying the original matrix
    N = np.copy(M)
    t = 0
    max_iterations = 100

    while t < max_iterations:
        if is_pos_def_nb(N):
            break
        else:
            # Manually add 'inc' to each diagonal element
            for i in range(N.shape[0]):
                N[i, i] += inc
            # Double the increment for the next iteration
            inc *= 2.0
            t += 1

    return N

def phase_linking_nb(px, args):
    epsilon = 1e-6
    
    amp = args['amp']
    diff = args['diff']
    crtShpNum = args['crtShpNum']
    pxshp = args['shp']
    pxns = args['shpNum']
    pxwgt = args['weights']

    # Initialization
    phase_series = np.empty(amp.shape[0], dtype='complex128')
    tempCoh = 0
    psmsk = 0
        
    pxAmpShp = amp[:, pxshp]
    pxDphShp = diff[:, pxshp]
    try:
        if pxns < crtShpNum:
            # processed as a Ps
        
            # get indexes in row and col of patch
            amps_ps = amp[:, px]

            amp_disp = np.std(amps_ps)/np.mean(amps_ps)

            # check using shp from amp and phs similarity or not
            if amp_disp <= 0.4:
                phase_series = diff[:, px].ravel()
                tempCoh = 1
                psmsk = 1
            else:
                if pxns == 1:
                    phase_series = diff[:, px].ravel()
                    tempCoh = 0
                    psmsk = 0  
                else:
                    pxscm = cal_weighted_sample_corr_matrix_numba(pxAmpShp, pxDphShp, pxwgt)
                    pxscoh = regularize_matrix_nb(np.abs(pxscm), epsilon)
                    phase_series, tempCoh = est_phase_series_EMI_nb(pxscm, pxscoh) 
        else:
            pxscm = cal_weighted_sample_corr_matrix_numba(pxAmpShp, pxDphShp, pxwgt)
            pxscoh = regularize_matrix_nb(np.abs(pxscm), epsilon)
            phase_series, tempCoh = est_phase_series_EMI_nb(pxscm, pxscoh)
    except Exception as exp:
        phase_series = diff[:, px].ravel()
        tempCoh = 0
        psmsk = 0  
    
    return phase_series, tempCoh, psmsk

def patch_process_pl(id, args):
    od, boxSpan, alpha_phs, ind_ifg_m, ind_ifg_s = args

    patchId = 'PATCH' + str(id)

    outDir = od[id]

    pldir = os.path.join(outDir, out_fln) # output filename
    
    print(f"\r{patchId}: Processing...", end='', flush=True)

    dtDir = os.path.join(outDir, 'dataStackPatch.h5')
    dtHdl = h5.File(dtDir, 'r')

    pAmp = dtHdl['amplitude'][()] + 1e-6
    pPhs = dtHdl['diffPhase'][()]
    # pAmp = np.where(pAmp==0., 1e-5, pAmp)

    patchBox = dtHdl['patchBox'][()]
    patchBoxEl = dtHdl['patchBoxEl'][()]
    # patchEnlInd = dtHdl['patchIndicesEnl'][()]

    poff = patchBox - patchBoxEl  # Inner patch offset

    pw0, pl0 = patchBox[2:4] + 1 - patchBox[0:2] 
    pw1, pl1 = patchBoxEl[2:4] + 1 - patchBoxEl[0:2] 
    pnum = pw0 * pl0

    pxy0 = get_xy_coords(np.arange(0, pw0), np.arange(0, pl0), pw0, pl0)
    pxy1 = get_xy_coords(np.arange(0, pw1), np.arange(0, pl1), pw1, pl1)
    pind = np.arange(0, pl1 * pw1).reshape(pl1, pw1)

    # Initialize arrays to hold the results
    numscn = pAmp.shape[0]

    shpNum = np.empty(pnum, dtype='int64')
    phLk = np.zeros((numscn, pnum), dtype='complex128')
    gmPta = np.zeros_like(shpNum, dtype='float32')
    psmsk = np.zeros_like(shpNum, dtype='int8')

    cv_phs = chi2.ppf(1 - alpha_phs, df=2)
    args1 = [pPhs, boxSpan, pxy0, pxy1, poff, pind, pw1, pl1, cv_phs, ind_ifg_m, ind_ifg_s]

    for pxl in range(pnum):
        display_progress_bar(patchId, pxl, pnum) 

        results = search_shp_phs(pxl, args1)
        
        shp, sn, weights = results
        shpNum[pxl] = sn

        args_pl = {
            'shp': shp,
            'shpNum': sn,
            'weights': weights,
            'amp': pAmp,
            'diff': pPhs,
            'crtShpNum': crtShpNum}
        
        results1 = phase_linking_nb(pxl, args_pl)
        phLk[:, pxl] = results1[0]
        gmPta[pxl] = results1[1]
        psmsk[pxl] = results1[2]
    
    with h5.File(pldir, 'w') as f_out:
        # Save arrays to the HDF5 file
        f_out.create_dataset('linkedPhase', data=phLk, compression='gzip', chunks=True)
        f_out.create_dataset('gammaPta', data=gmPta, compression='gzip', chunks=True)
        f_out.create_dataset('psmsk', data=psmsk, compression='gzip', chunks=True)
        f_out.create_dataset('shpNum', data=shpNum, compression='gzip', chunks=True)

    return 1

def filter_unprocessed_patches(patchNum, od):
  unprocessed_ids = []
  for id in range(patchNum):
    outDir = od[id]
    shpOd = os.path.join(outDir, out_fln)
 
    if not os.path.exists(shpOd):
      unprocessed_ids.append(id)
  return unprocessed_ids

if __name__ == '__main__':

    # load input data
    Input = h5.File(inputs_fln, 'r')

    slc_hdl = Input['slc']
    diff_hdl = Input['diffPhase']
    scenes = Input['dates'][()]
    imgSize = Input['imgSize'][()]

    width, length = imgSize
    numScn = len(scenes)

    # initialization
    rng_sw = rng_win // 2 # semi-width in range
    azi_sl = azi_win // 2 # semi-length in azimuth

    numPxl = width * length

    boxSpan = np.array([-rng_sw, -azi_sl, rng_sw, azi_sl]) # foundamenttal window box bounding bottomLeft and topRight
    ind_ifg_m, ind_ifg_s = consecutive_interferogram_indices(numScn, ifg_cnsc_num) # n-consecutive interferograms

    xy_img = get_xy_coords(np.arange(0, width), np.arange(0, length), width, length)
    indImg = np.arange(0, length*width).reshape(length, width)

    # dividing data in patches
    cwd = os.getcwd()
    patchList, patchNum = patch_slice(width, length, patch_size=patch_size)
    patchListDir = os.path.join(cwd, 'patchList.h5')
    if os.path.exists(patchListDir):
        os.remove(patchListDir)
    with h5.File(patchListDir, 'w') as inputhf:
        inputhf.create_dataset('patchList', data=patchList, dtype='int64')
        inputhf.create_dataset('patchNum', data=patchNum, dtype='int64')

    outDir = cwd + os.sep +'patches'
    od = [os.path.join(outDir, 'PATCH_{:03.0f}'.format(id)) for id in range(patchNum)] 

    for Id in range(patchNum):
        OD = od[Id]
        if not os.path.exists(OD):
            print(f'\r({Id+1}/{patchNum})making directory: {OD}', end='', flush=True)
            os.makedirs(OD, exist_ok=True)
        else:
            print(f'\r({Id+1}/{patchNum})directory: {OD} exists', end='', flush=True)
            continue
        
        patchBox = patchList[Id, :]
        patchBoxEl = get_big_box(patchBox, rng_sw, azi_sl, width, length)
        patchInd = indImg[patchBox[1]:patchBox[3] + 1, patchBox[0]:patchBox[2] + 1]
        patchEnlInd = indImg[patchBoxEl[1]:patchBoxEl[3] + 1, patchBoxEl[0]:patchBoxEl[2] + 1]
        
        pAmp = np.abs(slc_hdl[:, patchEnlInd.ravel()])
        pPhs = diff_hdl[:, patchEnlInd.ravel()]

        dtOd = os.path.join(OD, 'dataStackPatch.h5')
        if not os.path.exists(dtOd):
            with h5.File(dtOd, 'w') as inputhf:
                inputhf.create_dataset('amplitude', data=pAmp, dtype='float64', compression='gzip', chunks=True)
                inputhf.create_dataset('diffPhase', data=pPhs, dtype='complex128', compression='gzip', chunks=True)
                inputhf.create_dataset('patchIndices', data=patchInd, dtype='int64', compression='gzip', chunks=True)
                inputhf.create_dataset('patchIndicesEnl', data=patchEnlInd, dtype='int64', compression='gzip', chunks=True)
                inputhf.create_dataset('patchBox', data=patchBox, dtype='int64', compression='gzip', chunks=True)
                inputhf.create_dataset('patchBoxEl', data=patchBoxEl, dtype='int64', compression='gzip', chunks=True)
    
    unprocessed_ids = filter_unprocessed_patches(patchNum, od)
    print(f"\nTotal unprocessed patches: {len(unprocessed_ids)}")

    args = (od, boxSpan, alpha_phs, ind_ifg_m, ind_ifg_s)
    # for id in unprocessed_ids:
    #    patch_process_pl(id, args)
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        executor.map(patch_process_pl, unprocessed_ids, [args]*len(unprocessed_ids))
    executor.shutdown()
    print('PL done!')


    # integrate 
    flnPatchList = 'patchList.h5'
    flnPdt = 'dataStackPatch.h5'
    flnpl = out_fln

    patchList = h5.File(flnPatchList, 'r')['patchList'][()]
    patchNum = h5.File(flnPatchList, 'r')['patchNum'][()]

    cwd = os.getcwd()
    pPatch = os.path.join(cwd, 'patches')
    PP = [os.path.join(pPatch, 'PATCH_{:03.0f}'.format(id)) for id in range(patchNum)]
    numPxl = imgSize[0] * imgSize[1]

    shpNum = np.empty(numPxl, dtype='float32')
    psmsk = np.empty(numPxl, dtype='float32')

    dph = np.empty((numScn, numPxl), dtype='complex128')
    gmPta = np.empty(numPxl, dtype='float32')
    for pcnt in range(patchNum):
        print(f'\rintegrate patch ({pcnt+1}/{patchNum})', end='', flush=True)
        pp = PP[pcnt]
        dirDt = os.path.join(pp, flnPdt)

        dirpl = os.path.join(pp, flnpl)        
        pdt = h5.File(dirDt, 'r')

        pInd = pdt['patchIndices'][()].ravel()
        pDphOri =  pdt['diffPhase'][-1, :]

        ppl = h5.File(dirpl, 'r')
        pLkPhs = ppl['linkedPhase']
        pGmPta = ppl['gammaPta']
        pPsmsk = ppl['psmsk']
        pShpNum = ppl['shpNum']
        dph[:, pInd] = pLkPhs[()]
        gmPta[pInd] = pGmPta[()]
        psmsk[pInd] = pPsmsk[()]
        shpNum[pInd] = pShpNum[()]

    with h5.File('pl_prods_overview.h5', 'w') as hdlShp:
        hdlShp.create_dataset('linkedPhase', data=dph)
        hdlShp.create_dataset('gammaPta', data=gmPta)
        hdlShp.create_dataset('shpNum', data=shpNum)
        hdlShp.create_dataset('psMask', data=psmsk)
        hdlShp.create_dataset('dates', data=scenes)
        hdlShp.create_dataset('imgSize', data=imgSize)
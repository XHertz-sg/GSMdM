# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:22:46 2025

@author: e0718973
"""

import h5py
import tkinter as tk
from tkinter import filedialog
import scipy.optimize
import matplotlib
import scipy.io
import matplotlib.cm as cm
import cupy as cp
import cupyx.scipy.ndimage as cpx
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ctypes as _ctypes
from ctypes import POINTER, c_float, c_size_t
from ctypes import *
import pickle
import scipy.optimize as opt
from scipy.spatial import cKDTree
import tifffile
import cv2
import glob

# 使用 tkinter 打开对话框选择文件夹路径
def select_folder():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title="选择一个文件夹")  # 打开选择文件夹对话框
    return folder_path

x_offset = 377
x_offset1 = x_offset+373
x_offset2 = x_offset1+371
y_offset = 5
y_offset2 = y_offset+5
y_offset3 = y_offset2+6
roi1 = [50,380,50,380]
roi2 = [roi1[0]+x_offset,roi1[1]+x_offset,roi1[2]+y_offset,roi1[3]+y_offset]
roi3 = [roi1[0]+x_offset1,roi1[1]+x_offset1,roi1[2]+y_offset2,roi1[3]+y_offset2]
roi4 = [roi1[0]+x_offset2,roi1[1]+x_offset2,roi1[2]+y_offset3,roi1[3]+y_offset3]
folder_path= select_folder()

print('1/7 loading and cropping')
# 获取所有 .tif 文件的路径列表
if folder_path:
    # 遍历文件夹及其所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        all_fit_results = []
        tif_files = sorted(glob.glob(os.path.join(root, "*.tif")))
        imgstack2 = tifffile.imread(tif_files[0:len(tif_files)])
        output1 = imgstack2[:,int(roi1[2]):int(roi1[3]),int(roi1[0]):int(roi1[1])]
        output2 = imgstack2[:,int(roi2[2]):int(roi2[3]),int(roi2[0]):int(roi2[1])]
        output3 = imgstack2[:,int(roi3[2]):int(roi3[3]),int(roi3[0]):int(roi3[1])]
        output4 = imgstack2[:,int(roi4[2]):int(roi4[3]),int(roi4[0]):int(roi4[1])]

print('2/7 corp finished, start finding peaks')
img1 = cp.array(output1).astype(cp.float32)  
img2 = cp.array(output2).astype(cp.float32)
img3 = cp.array(output3).astype(cp.float32)
img4 = cp.array(output4).astype(cp.float32)


def extract_local_patches(image, sigma=1.4, threshold=0.6, sub_image_size=7):
    """
    在 `cupy` GPU 上计算局部最大值并提取 7×7 子图像，同时输出峰值坐标 (y, x) 并存储到 `pandas.DataFrame`。

    参数:
    - image: `cupy.ndarray`，形状 (num_images, height, width) (1000, 330, 330)
    - sigma: float，高斯拉普拉斯 (LoG) 滤波的标准差
    - threshold: float，局部最大值的强度阈值
    - sub_image_size: int，裁剪子图像的大小（默认 7×7）

    返回:
    - DataFrame，包含 ["sub_image", "image_index", "y", "x"]
    """
    num_images, height, width = image.shape
    half_size = sub_image_size // 2

    # **计算 LoG 并寻找局部最大值**
    log_image = -cpx.gaussian_laplace(image, sigma)
    peaks = (log_image == cpx.maximum_filter(log_image, size=(1, sub_image_size, sub_image_size))) & (log_image > threshold)

    # **提取峰值坐标 (image, y, x)**
    peak_coords = cp.argwhere(peaks)  # shape=(N, 3)
    if peak_coords.shape[0] == 0:  # 如果没有峰值点，返回空 DataFrame
        return pd.DataFrame(columns=["sub_image", "image_index", "y", "x"])

    image_idx, y_coords, x_coords = peak_coords.T  # 转置以获得单独数组

    # **计算子图像的索引边界**
    y_min = cp.maximum(0, y_coords - half_size)
    y_max = cp.minimum(height, y_coords + half_size + 1)
    x_min = cp.maximum(0, x_coords - half_size)
    x_max = cp.minimum(width, x_coords + half_size + 1)

    # **创建网格索引，批量提取子图像**
    y_grid, x_grid = cp.meshgrid(cp.arange(sub_image_size), cp.arange(sub_image_size), indexing="ij")
    y_indices = (y_min[:, None, None] + y_grid[None, :, :]).clip(0, height - 1)
    x_indices = (x_min[:, None, None] + x_grid[None, :, :]).clip(0, width - 1)

    # **利用 `cupy` 批量索引提取子图像**
    sub_images = image[image_idx[:, None, None], y_indices, x_indices]

    # **将数据转换为 `pandas.DataFrame`**
    sub_images_flat = cp.asnumpy(sub_images.reshape(sub_images.shape[0], -1))  # 展平成 (N, 49)
    df = pd.DataFrame({
        "sub_image": list(sub_images_flat),  # `pandas` 处理变长数组时必须使用 `list`
        "image_index": cp.asnumpy(image_idx),
        "y": cp.asnumpy(y_coords),
        "x": cp.asnumpy(x_coords)
    })

    return df
sig_set = 1.4
thres_set = 0.2
df1 = extract_local_patches(img1,sig_set,thres_set,7)
df2 = extract_local_patches(img2,sig_set,thres_set,7)
df3 = extract_local_patches(img3,sig_set,thres_set,7)
df4 = extract_local_patches(img4,sig_set,thres_set,7)

# =============================================================================
# dataframes = {"df1": df1, "df2": df2, "df3": df3, "df4": df4}
# with open("peaks.pkl", "wb") as f:
#     pickle.dump(dataframes, f)
# =============================================================================

print('3/7 peaks find finished, start fitting')

def gpu_fit(subimages, psf_sigma=1.1, iterations=100, fit_mode=4):
    """
    Runs GPU-based fitting on input subimages using pystormbasic.dll.

    Parameters:
    - subimages: np.ndarray of shape (num_images, size, size)
    - psf_sigma: float, PSF sigma value for fitting
    - iterations: int, number of iterations for fitting
    - fit_mode: int, mode for fitting (default: 4 for MLE x/y/n/bg/sigma)

    Returns:
    - fitresult1: np.ndarray of shape (num_images, num_fitresult)
    - fiterror1: np.ndarray of shape (num_images, num_fiterror)
    """
    # Convert data to float32
    datanew = np.float32(subimages)
    
    # Load GPU library
    hgpu = _ctypes.CDLL('pystormbasic.dll')
    pygpufit = hgpu.pystorm
    pygpufit.argtypes = [POINTER(c_float), c_size_t, c_size_t, c_size_t, c_float, POINTER(c_float), POINTER(c_float), c_size_t]
    
    # Prepare data pointers
    data_p = datanew.ctypes.data_as(POINTER(c_float))
    c_psfSigma = float(psf_sigma)
    
    fitraw = datanew.shape[0]  # Number of subimages
    c_fitraw = int(fitraw)
    c_sz = int(datanew.shape[1])  # Size of each subimage
    c_iterations = int(iterations)
    c_fitmode = int(fit_mode)
    
    # Define output arrays
    num_fiterror = 6  # x, y, n, bg, sigma_x, sigma_y (or 6 for general case)
    num_fitresult = 5  # x, y, n, bg, sigma
    
    fiterror = np.zeros(fitraw * num_fiterror, dtype=np.float32)
    fitresult = np.zeros(fitraw * num_fitresult, dtype=np.float32)
    
    fiterror_p = fiterror.ctypes.data_as(POINTER(c_float))
    fitresult_p = fitresult.ctypes.data_as(POINTER(c_float))
    
    # Call GPU fitting function
    pygpufit(data_p, c_fitraw, c_sz, c_iterations, c_psfSigma, fitresult_p, fiterror_p, c_fitmode)
    
    # Reshape results
    fiterror1 = np.reshape(fiterror, (fitraw, num_fiterror))
    fitresult1 = np.reshape(fitresult, (fitraw, num_fitresult))
    
    return fitresult1, fiterror1

fitresult1, fiterror1 = gpu_fit(np.array(df1['sub_image'].to_list()).reshape(-1,7,7), psf_sigma=1.1, iterations=50, fit_mode=4)
fitresult2, fiterror2 = gpu_fit(np.array(df2['sub_image'].to_list()).reshape(-1,7,7), psf_sigma=1.1, iterations=50, fit_mode=4)
fitresult3, fiterror3 = gpu_fit(np.array(df3['sub_image'].to_list()).reshape(-1,7,7), psf_sigma=1.1, iterations=50, fit_mode=4)
fitresult4, fiterror4 = gpu_fit(np.array(df4['sub_image'].to_list()).reshape(-1,7,7), psf_sigma=1.1, iterations=50, fit_mode=4)
print('4/7 fitting finished, start filtering')

# =============================================================================
# arrays_dict = {
#     'fit1': fitresult1,
#     'fit2': fitresult2,
#     'fit3': fitresult3,
#     'fit4': fitresult4
# }
# pd.to_pickle(arrays_dict, 'fits.pkl')
# =============================================================================

def process_fit_results(fitres, df):
    # 筛选满足条件的行
    condition = (fitres[:, 4] < 2.8) & (fitres[:, 4] > 0.6)
    fit_filtered = fitres[condition, :]
    
    # 提取峰值数据
    peak = np.array([df["x"].to_numpy(), df["y"].to_numpy(), df["image_index"].to_numpy()]).T
    peak_filtered = peak[condition, :]
    
    # 修正坐标
    fit_filtered[:, 0:2] = fit_filtered[:, 0:2] + peak_filtered[:, 0:2] - 3
    
    # 添加 image_index 列
    fit_result = np.hstack((fit_filtered, peak_filtered[:, 2].reshape(-1, 1)))
    
    return fit_result

fit1 = process_fit_results(fitresult1, df1)
fit2 = process_fit_results(fitresult2, df2)
fit3 = process_fit_results(fitresult3, df3)
fit4 = process_fit_results(fitresult4, df4)
fit2[:,0] =fit2[:,0] -0.46498210
fit2[:,1] =fit2[:,1] +0.86601648
fit3[:,0] =fit3[:,0] -1.62871212
fit3[:,1] =fit3[:,1] +2.15867046
fit4[:,0] =fit4[:,0] -4.1158322
fit4[:,1] =fit4[:,1] +2.96733141

print('5/7 fit filtered, start displacing')

def best_displacement(xy1, xy2, max_distance,frame_index,fov_index):

    # 创建 KDTree
    tree_b = cKDTree(xy2)
    
    # 记录每个点的当前最佳配对
    used_b_indices = set()  # 记录已经配对过的 b 中的点
    best_pairs = {}
    
    for i in range(len(xy1)):
        point_a = xy1[i]
        
        # 查询 b 中离 point_a 最近的点
        dist, idx = tree_b.query(point_a, k=1)
        
        if dist > max_distance:
            continue  # 超过最大距离，跳过
        
        # 更新配对信息
        if idx not in used_b_indices:
            best_pairs[i] = (idx, dist)
            used_b_indices.add(idx)
        else:
            # 如果这个点已经被使用，检查是否有更近的点
            for existing_index, (existing_b_index, existing_dist) in list(best_pairs.items()):
                if existing_b_index == idx:
                    if dist < existing_dist:
                        best_pairs.pop(existing_index)
                        best_pairs[i] = (idx, dist)
                    break
    
    # 转换为更易读的形式，包括距离
    result_pairs = [(xy1[i], xy2[best_pairs[i][0]], best_pairs[i][1]) for i in best_pairs]
    displace=[]
    for pair in result_pairs:
        xmid = np.mean((pair[0][0],pair[1][0]))
        dx = pair[1][0]-pair[0][0]
        ymid = np.mean((pair[0][1],pair[1][1]))
        dy = pair[1][1]-pair[0][1]
        dist = pair[2]
        displace.append(np.array([xmid,ymid,dx,dy,dist,frame_index,fov_index]))
    return displace

max_distance = 15
dout = []
for fs in range(0,int(np.max(fit1[:,5]))):
    xy1 = fit1[fit1[:,5]==fs,0:2]
    xy2 = fit2[fit2[:,5]==fs,0:2]
    xy3 = fit3[fit3[:,5]==fs,0:2]
    xy4 = fit4[fit4[:,5]==fs,0:2]
    d1 = best_displacement(xy1, xy2, max_distance,fs,1)
    d2 = best_displacement(xy2, xy3, max_distance,fs,2)
    d3 = best_displacement(xy1, xy2, max_distance,fs,3)
    dout= dout+d1 +d2+d3
dout = np.array(dout)
# =============================================================================
# plt.figure,plt.hist(dout[:,4])
# =============================================================================
print('6/7 displaceing finished, start mapping')

def pdf(r, r_max, delta_t, D, b):
    """
    PDF for fitting issues
    """
    a = 4 * delta_t * D
    density = ((2 * r) / a) * np.exp(-(r ** 2) / a) + (b * r)
    normalization = (1 - np.exp(-(r_max ** 2) / a)) + ((r_max ** 2) * b) / 2
    return density / normalization
def likelihood(r, r_max, delta_t, D, b):
    """
    Getting summs of the Logs for MLE 
    """
    return np.sum(np.log(pdf(r, r_max, delta_t, D, b)))  
def minimization_function(x, *args):
    """
    Minimization of the MLE 
    Returns reversed value of likelihhod
    """
    D, b = x
    r, r_max, delta_t = args
    return -likelihood(r, r_max, delta_t, D, b)
def maximum_likelihood(r, r_max, delta_t, D, b=0):
    """
    Calculating parameters of the distribution (Diffusion coefficient and background correction coefficient)
    Maximum of the likelihood function is evaluated by minimization of - likelihood function "minimization_function(x, *args)"
    """

    bounds = [(0, None), (0, None)]
    initial = (D, b)

        
    res = scipy.optimize.minimize(minimization_function, initial, (r, r_max, delta_t), method='Nelder-Mead', bounds=bounds)
    
    return res.x


disp = dout[:,4]*4.6*2/150
dt = 4e-3
initialD = 1
initialb = 1
rmax = np.max(disp)
D_all, b_all = maximum_likelihood(disp, rmax, delta_t = dt, D = initialD, b= initialb)

plt.figure(1)
h_g = 15
plt.hist(disp,h_g)
amp = 10
x0 = np.linspace(0,rmax,h_g*amp)
y0 = pdf(x0, rmax, dt,D_all, b_all)
y00 = y0/np.sum(y0)*disp.shape[0]*amp
plt.plot(x0, y00)
plt.title(D_all)

reso = 2
imgx =330
imgy =330
x_grid = int(imgx/reso)
y_grid = int(imgy/reso)
xfix = np.fix(dout[:,0]/reso)
yfix = np.fix(dout[:,1]/reso)

D_local = np.zeros((x_grid,y_grid))
b_local = np.zeros((x_grid,y_grid))
rmaxdd = np.zeros((x_grid,y_grid))
pmap = np.zeros((x_grid,y_grid))
radi = imgx/reso/2-2
for i in range(0,x_grid):
    for j in range(0,y_grid):
        d = np.sqrt((i-imgx/reso/2)**2+(j-imgx/reso/2)**2)
        if d<radi:
            local_disp = disp[(xfix[:] == i) & (yfix[:] == j)]
            pmap[i,j] = local_disp.shape[0]
            if pmap[i,j] > 0:
                rmaxdd[i,j] = np.max(local_disp)
                D_local[i,j], b_local[i,j] = maximum_likelihood(local_disp, rmaxdd[i,j], delta_t = dt, D = initialD, b= initialb)

D_local = D_local.T
b_local = b_local.T
D_b_p = np.zeros((b_local.shape[1], b_local.shape[0], 3))
D_b_p[:,:,0] = D_local
D_b_p[:,:,1] = b_local
D_b_p[:,:,2] = pmap

cmap = cm.get_cmap('jet', 256)
cmap.set_under('black')
cmap.set_over('red')

dd1 = np.where(D_local<20,D_local,0)
####### blur
d_blur = scipy.signal.medfilt(dd1,3)
# =============================================================================
dmin =0.0001
dmax =5
norm = matplotlib.colors.Normalize(vmin = dmin, vmax =dmax)
plt.figure(2)
plt.imshow(d_blur , cmap, norm = norm)
plt.colorbar(fraction = 0.06)
plt.show()

dm = np.mean(dd1[dd1>0])
plt.figure(3)
plt.hist(dd1[dd1>0],10,alpha=0.5)
plt.figure(4)
plt.imshow(pmap.T, cmap = 'CMRmap')
plt.colorbar(fraction = 0.06)
plt.show()

print('7/7 Mapping Done, saving')

directory = os.path.join(folder_path, 'Dmap')
# Create the directory
os.makedirs(directory, exist_ok=True)
save_path1 = os.path.join(directory, 'Dmap.npy')
np.save(save_path1, D_b_p)


# 将 DataFrame 存入字典
save_path2 = os.path.join(directory, 'PEAKS and fits.pkl')
pfdata = {'df1': df1, 'df2': df2, 'df3': df3, 'df4': df4, 
        'fit1': fit1,'fit2': fit2,'fit3': fit3,'fit4': fit4,}
# 保存到 pkl 文件
with open(save_path2, "wb") as f:
    pickle.dump(pfdata, f)
print('Finished')
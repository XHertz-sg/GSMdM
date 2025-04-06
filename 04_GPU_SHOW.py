# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import tifffile
import cv2
import matplotlib.patches as patches
import matplotlib
import matplotlib.cm as cm
import scipy
import os
import tkinter as tk
from tkinter import filedialog

cmap = cm.get_cmap('jet', 256)
cmap.set_under('black')
cmap.set_over('red')
# =============================================================================
# data = np.load('//toast.mbi.nus.edu.sg/scratch/kwong/e0718973/2024/AAAAZX_1026GSMDM_NLS/1026ARPE_NLS/NLE_K/002/Dmap/Dmap.npy')
# =============================================================================
# =============================================================================
# data = np.load('E:/2023/0926 RPE1 ER-SG/b0613new/b_all_dmap0613_reso2.npy')
# =============================================================================
def select_file():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename()  # 打开文件选择对话框
    return file_path

# reading all npy in selected file folder
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="请选择一个文件夹")
all_data = []
if folder_path:
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    for npy_file in npy_files:
        file_path = os.path.join(folder_path, npy_file)
        data0 = np.load(file_path)
        print(f"读取文件: {npy_file}, 数据形状: {data0.shape}")
        all_data.append(data0)
    combined_matrix = np.vstack(all_data)
    print(f"合并后的矩阵形状: {combined_matrix.shape}")
else:
    print("未选择文件夹。")
    
data = combined_matrix
Dmap = data[:,:,0]
# =============================================================================
# Dmap = data[60:115,40:95,0]
# =============================================================================
dd = Dmap
Pmap = data[:,:,2].T
reso =2
dd1 = np.where(dd<100,dd,0)
pp1 = np.where(dd<100,Pmap,0)
all_p = np.sum(pp1) 

####### blur

d_blur = scipy.signal.medfilt(dd1,3)
p_blur = scipy.signal.medfilt(pp1,3)
# =============================================================================
dmin =0.01
dmax =10
norm = matplotlib.colors.Normalize(vmin = dmin, vmax =dmax)
plt.figure(1)
plt.imshow(d_blur, cmap, norm = norm)
plt.colorbar(fraction = 0.06)
plt.show()

unit_size = reso * 4600/150
plt.figure(2)
plt.hist(np.log10(dd1[dd1>0]),50,alpha=0.5)

dm = np.mean(dd1[dd1>0])
dm2 = np.median(dd1[dd1>0])
plt.title(dm)

plt.figure(3)
# =============================================================================
# norm2 = matplotlib.colors.Normalize(vmin = 0, vmax =5000)
# =============================================================================
plt.imshow(p_blur, cmap = 'inferno')
plt.colorbar(fraction = 0.06)

plt.show()

# =============================================================================
# dataK = np.load('E:/2024/0513 LBR GSMDM//K005/K005-1-dmap.npy')
# dataE = np.load('E:/2024/0513 LBR GSMDM/E003/E003-12-dmap.npy')
# dataN = np.load('E:/2024/0513 LBR GSMDM/N002/N002-34-dmap.npy')
# 
# ddk = dataK[:,:,0]
# ddn = dataN[:,:,0]
# dde = dataE[:,:,0]
# dd1k = np.where(ddk<20,ddk,0)
# dd1n = np.where(ddn<20,ddn,0)
# dd1e = np.where(dde<20,dde,0)
# d_blurk = scipy.signal.medfilt(dd1k,3)
# d_blurn = scipy.signal.medfilt(dd1n,3)
# d_blure = scipy.signal.medfilt(dd1e,3)
# 
# dmin = 1e-15
# dmax =15
# norm = matplotlib.colors.Normalize(vmin = dmin, vmax =dmax)
# 
# plt.figure(1)
# plt.imshow(d_blurk, cmap, norm = norm)
# plt.colorbar(fraction = 0.06)
# plt.show()
# dmk = np.median(dd1k[dd1k>0])
# 
# plt.figure(2)
# plt.imshow(d_blurn, cmap, norm = norm)
# plt.colorbar(fraction = 0.06)
# plt.show()
# dmn = np.median(dd1n[dd1n>0])
# 
# plt.figure(3)
# plt.imshow(d_blure, cmap, norm = norm)
# plt.colorbar(fraction = 0.06)
# plt.show()
# dme = np.median(dd1e[dd1e>0])
# =============================================================================



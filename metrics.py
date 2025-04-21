import lpips
import torch
import ignite.metrics.regression
import numpy as np
import pandas as pd
import ignite.metrics.regression.pearson_correlation 
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

t = transforms.ToTensor()
trips_imgs = os.listdir("TRIPS")
results = []
for i in range(0,len(trips_imgs),2):
    ncc = ignite.metrics.regression.pearson_correlation.PearsonCorrelation()
    ssim = ignite.metrics.SSIM(data_range=1.0)
    psnr = ignite.metrics.PSNR(data_range=1.0)
    gt = Image.open(f".\\TRIPS\\{trips_imgs[i]}")
    render = Image.open(f".\\TRIPS\\{trips_imgs[i+1]}")
    gt_d,r_d = t(gt),t(render)
    gt_d,r_d = torch.reshape(gt_d,(1,gt_d.shape[0],gt_d.shape[1],gt_d.shape[2])),torch.reshape(r_d,(1,r_d.shape[0],r_d.shape[1],r_d.shape[2]))
    ncc_gt,ncc_r = torch.reshape(gt_d,(-1,)),torch.reshape(r_d,(-1,))
    ncc.update((ncc_gt,ncc_r))
    ssim.update((gt_d,r_d))
    psnr.update((gt_d,r_d))
    results.append((ncc.compute(),ssim.compute(),psnr.compute()))
pd.DataFrame(results,columns=["NCC","SSIM","PSNR"]).to_csv("TRIPS_metrics_no_lpips.csv")

# Get corresponding images from GS dataset
img_numbers = [x[5:10] for x in os.listdir("TRIPS")]
img_numbers = set(img_numbers)
gt_dirs = os.listdir("train/ours_30000/gt")
gt_dirs = [x for x in gt_dirs if x[:5] in img_numbers]
t = transforms.ToTensor()

# Calculate GS metrics
results = []
for dir in gt_dirs:
    ncc = ignite.metrics.regression.pearson_correlation.PearsonCorrelation()
    ssim = ignite.metrics.SSIM(data_range=1.0)
    psnr = ignite.metrics.PSNR(data_range=1.0)
    gt = Image.open(f"train/ours_30000/gt/{dir}")
    render = Image.open(f"train/ours_30000/renders/{dir}")
    gt_d,r_d = t(gt),t(render)
    gt_d,r_d = torch.reshape(gt_d,(1,gt_d.shape[0],gt_d.shape[1],gt_d.shape[2])),torch.reshape(r_d,(1,r_d.shape[0],r_d.shape[1],r_d.shape[2]))
    ncc_gt,ncc_r = torch.reshape(gt_d,(-1,)),torch.reshape(r_d,(-1,))
    ncc.update((ncc_gt,ncc_r))
    ssim.update((gt_d,r_d))
    psnr.update((gt_d,r_d))
    results.append((ncc.compute(),ssim.compute(),psnr.compute()))
pd.DataFrame(results,columns=["NCC","SSIM","PSNR"]).to_csv("GS_metrics_no_lpips.csv")

#Calculate LPIPS for GS
results = []
for dir in gt_dirs:
    lpip = lpips.LPIPS(net="vgg")
    gt = Image.open(f"train/ours_30000/gt/{dir}")
    render = Image.open(f"train/ours_30000/renders/{dir}")
    gt_d,r_d = t(gt),t(render)
    gt_d,r_d = torch.reshape(gt_d,(1,gt_d.shape[0],gt_d.shape[1],gt_d.shape[2])),torch.reshape(r_d,(1,r_d.shape[0],r_d.shape[1],r_d.shape[2]))
    results.append(lpip(gt_d,r_d)[0][0][0][0].detach().numpy())
    print(results)
    del gt
    del render
    del lpip
    del gt_d
    del r_d
    

results = []
for i in range(0,len(trips_imgs),2):
    lpip = lpips.LPIPS(net="vgg")
    gt = Image.open(f"./TRIPS/{trips_imgs[i]}")
    render = Image.open(f"./TRIPS/{trips_imgs[i+1]}")
    gt_d,r_d = t(gt),t(render)
    gt_d,r_d = torch.reshape(gt_d,(1,gt_d.shape[0],gt_d.shape[1],gt_d.shape[2])),torch.reshape(r_d,(1,r_d.shape[0],r_d.shape[1],r_d.shape[2]))
    results.append(lpip(gt_d,r_d)[0][0][0][0].detach().numpy())
    print(results)
    del gt
    del render
    del lpip
    del gt_d
    del r_d
    
pd.DataFrame(results,columns=["LPIPS"]).to_csv("TRIPS_LPIPS.csv")
import os
import sys
import glob
import torch
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch.nn.functional as F
from tqdm import tqdm

# 專案路徑設定
MADOS_ROOT = r"C:\Users\楊大正\OneDrive\文件\Runspace"
MADOS_DATA_ROOT = os.path.join(MADOS_ROOT, 'MADOS')

if os.path.join(MADOS_ROOT, 'marinext') not in sys.path:
    sys.path.append(os.path.join(MADOS_ROOT, 'marinext'))

import mmseg.models.backbones 
from models.EfficientNetB2_adapter import EfficientNetB2Adapter
from models.marinext_adapter import MarinextAdapter
from marinext.marinext_wrapper import MariNext
from utils.dataset import DEFAULT_BANDS_MEAN, DEFAULT_BANDS_STD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_band_val(path): return int(path.split('_')[-2])

def load_18ch_tensor_and_gt(roi_name):
    parts = roi_name.rsplit('_', 1)
    scene_folder, patch_idx = parts[0], parts[1]
    search_pattern = os.path.join(MADOS_DATA_ROOT, scene_folder, '*', f"*_L2R_rhorc_*_{patch_idx}.tif")
    all_bands = sorted(glob.glob(search_pattern), key=get_band_val)
    if len(all_bands) < 11: return None, None
    
    img_data = []
    for band_path in all_bands:
        upscale_factor = int(os.path.basename(os.path.dirname(band_path))) // 10
        with rasterio.open(band_path) as src:
            img_data.append(src.read(1, out_shape=(int(src.height * upscale_factor), int(src.width * upscale_factor)), resampling=Resampling.nearest))
    img = np.stack(img_data)
    
    eps = 1e-8
    blue, green, red, re2, nir, swir1 = img[1], img[2], img[3], img[5], img[7], img[9]
    fai = nir - (red + (swir1 - red) * 0.19)
    fdi = nir - (re2 + (swir1 - re2) * 0.1186)
    si  = np.sqrt(np.clip((1-blue)*(1-green)*(1-red), 0, None))
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndmi = (nir - swir1) / (nir + swir1 + eps)
    bsi  = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)
    
    img_18 = np.concatenate([img, np.stack([ndvi, ndwi, ndmi, fai, fdi, si, bsi])], axis=0)
    
    active_mean = DEFAULT_BANDS_MEAN[:18].reshape(-1, 1, 1)
    active_std = DEFAULT_BANDS_STD[:18].reshape(-1, 1, 1)
    nan_mask = np.isnan(img_18)
    img_18[nan_mask] = np.tile(active_mean, (1, img_18.shape[1], img_18.shape[2]))[nan_mask]
    tensor_18 = torch.from_numpy((img_18 - active_mean) / active_std).float().unsqueeze(0)
    
    gt_mask = np.full(img_18.shape[1:], -1, dtype=np.int16)
    cl_path = os.path.join(MADOS_DATA_ROOT, scene_folder, "10", f"{scene_folder}_L2R_cl_{patch_idx}.tif")
    if os.path.exists(cl_path):
        with rasterio.open(cl_path) as cl_src:
            gt_mask = cl_src.read(1).astype(np.int16)
            
    return tensor_18, gt_mask

def main():
    print("🚀 載入 PK 模型中...")
    # 1. 11ch Baseline (MariNeXt 15-class)
    base_model = MariNext(11, 15)
    ckpt = torch.load(os.path.join(MADOS_ROOT, "baseline_weight.pth"), map_location=device)
    base_model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    base_model = base_model.eval().to(device)

    # 2. 18ch Stage 1 & Stage 2
    eb2_cfg = {'experiment': {'input_channels': 18, 'num_classes': 2, 'model_variant': 'efficientnet_b2', 'pretrained': False}}
    model_s1 = EfficientNetB2Adapter(eb2_cfg, model_path=os.path.join(MADOS_ROOT, "EB2_weight.pth")).model.eval().to(device)
    seg_cfg = {'experiment': {'input_channels': 18, 'model_name': 'MariNext'}}
    model_s2 = MarinextAdapter(seg_cfg, model_path=os.path.join(MADOS_ROOT, "MariNeXt_weight.pth")).model.eval().to(device)

    test_txt = os.path.join(MADOS_DATA_ROOT, "splits", "test_X.txt")
    with open(test_txt, 'r', encoding='utf-8') as f:
        rois = [line.strip() for line in f if line.strip()]

    demo_s1_hero = []        # GT 無油，Baseline 誤抓，Stage 1 擋下
    demo_baseline_failed = [] # GT 有油，Baseline 翻車，Hybrid 提升 20% 以上
    demo_all_perfect = []     # GT 有油，Baseline & Hybrid 皆 > 85%

    print(f"🔍 開始雙架構對抗掃描 ({len(rois)} 張)...")
    with torch.no_grad():
        for roi in tqdm(rois):
            tensor_18, gt_mask = load_18ch_tensor_and_gt(roi)
            if tensor_18 is None: continue
            
            tensor_18 = tensor_18.to(device)
            valid = (gt_mask > 0)
            if not valid.any(): continue
            has_oil = np.any(gt_mask[valid] == 6)
            y_true = (gt_mask[valid] == 6).astype(int)
            
            # --- Baseline 11ch 預測 ---
            tensor_11 = tensor_18[:, :11, :, :]
            out_base = base_model(tensor_11)
            out_base = F.interpolate(out_base, size=gt_mask.shape, mode='bilinear', align_corners=True)
            pred_base_15 = out_base.argmax(1)[0].cpu().numpy()
            y_pred_base = np.zeros_like(y_true)
            y_pred_base[pred_base_15[valid] == 5] = 1 # 15 類中的 5 是 Oil
            
            tp_b = int(((y_pred_base == 1) & (y_true == 1)).sum())
            fp_b = int(((y_pred_base == 1) & (y_true == 0)).sum())
            fn_b = int(((y_pred_base == 0) & (y_true == 1)).sum())
            iou_base = (tp_b / (tp_b + fp_b + fn_b + 1e-7)) * 100
            
            # --- Hybrid 18ch 預測 ---
            out_s1 = model_s1(tensor_18)
            prob = F.softmax(out_s1[0] if isinstance(out_s1, tuple) else out_s1, dim=1)[0, 1].item()
            is_candidate = prob >= 0.20
            
            if is_candidate:
                out_s2 = model_s2(tensor_18)
                out_s2 = F.interpolate(out_s2, size=gt_mask.shape, mode='bilinear', align_corners=True)
                y_pred_hyb = out_s2.argmax(1)[0].cpu().numpy()[valid]
            else:
                y_pred_hyb = np.zeros_like(y_true)
                
            tp_h = int(((y_pred_hyb == 1) & (y_true == 1)).sum())
            fp_h = int(((y_pred_hyb == 1) & (y_true == 0)).sum())
            fn_h = int(((y_pred_hyb == 0) & (y_true == 1)).sum())
            iou_hyb = (tp_h / (tp_h + fp_h + fn_h + 1e-7)) * 100
            
            # 💡 英雄榜 (demo_s1_hero)：GT無油污 + Baseline誤抓 + Stage 1成功攔截
            if not has_oil and fp_b > 0 and not is_candidate:
                demo_s1_hero.append(f"{roi} | Baseline 誤抓: {fp_b} px | Stage 1 成功攔截 (Prob: {prob:.3f})")
                
            # 💡 有油污的分流
            if has_oil:
                # 翻車區 (demo_baseline_failed)：2 stage 有效抓取 (IoU 贏 Baseline 20% 以上)
                if (iou_hyb - iou_base) >= 20.0:
                    demo_baseline_failed.append(f"{roi} | Baseline IoU: {iou_base:.1f}% -> Hybrid IoU: {iou_hyb:.1f}% (提升: {iou_hyb - iou_base:.1f}%)")
                
                # 模範生 (demo_all_perfect)：兩者 IoU 皆大於 85%
                elif iou_base > 85.0 and iou_hyb > 85.0:
                    demo_all_perfect.append(f"{roi} | Baseline IoU: {iou_base:.1f}% | Hybrid IoU: {iou_hyb:.1f}%")

    out_dir = os.path.join(MADOS_DATA_ROOT, "splits")
    with open(os.path.join(out_dir, "demo_s1_hero.txt"), "w", encoding='utf-8') as f: f.write("\n".join(demo_s1_hero))
    with open(os.path.join(out_dir, "demo_baseline_failed.txt"), "w", encoding='utf-8') as f: f.write("\n".join(demo_baseline_failed))
    with open(os.path.join(out_dir, "demo_all_perfect.txt"), "w", encoding='utf-8') as f: f.write("\n".join(demo_all_perfect))
    
    print("\n🎉 分析完成！")
    print(f"🛡️ 英雄榜 (Baseline 誤抓，S1 擋下): {len(demo_s1_hero)} 張")
    print(f"📈 翻車區 (Hybrid 提升 20% 以上): {len(demo_baseline_failed)} 張")
    print(f"✨ 模範生 (雙模型皆 > 85%): {len(demo_all_perfect)} 張")

if __name__ == "__main__":
    main()
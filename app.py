import base64
import cv2
import numpy as np
import rasterio
import os
import glob
import sys
import json
import random
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from rasterio.enums import Resampling

# Ensure the system treats 'marinext' as a root module
MADOS_ROOT = r"C:\Users\楊大正\OneDrive\文件\Runspace"
MADOS_DATA_ROOT = os.path.join(MADOS_ROOT, 'MADOS')

if os.path.join(MADOS_ROOT, 'marinext') not in sys.path:
    sys.path.append(os.path.join(MADOS_ROOT, 'marinext'))

import mmseg.models.backbones 

from models.ConvNeXt_adapter import ConvNeXtAdapter
from models.EfficientNetB2_adapter import EfficientNetB2Adapter
from models.marinext_adapter import MarinextAdapter
from marinext.marinext_wrapper import MariNext
from utils.dataset import DEFAULT_BANDS_MEAN, DEFAULT_BANDS_STD

app = FastAPI(title="Marine Eye - Demo Architecture")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Initializing Inference Engine on: {device}")

models = {}
def get_weight_path(filename):
    path1 = os.path.join(MADOS_ROOT, filename)
    path2 = os.path.join(MADOS_ROOT, filename + ".pth")
    if os.path.exists(path1): return path1
    elif os.path.exists(path2): return path2
    else: raise FileNotFoundError(f"Weight file not found: {filename}")

try:
    print("Loading Stage 1: EfficientNetB2 (18ch)...")
    eb2_cfg = {'experiment': {'input_channels': 18, 'num_classes': 2, 'model_variant': 'efficientnet_b2', 'pretrained': False}}
    models['EB2'] = EfficientNetB2Adapter(eb2_cfg, model_path=get_weight_path("EB2_weight")).model.eval().to(device)

    print("Loading Stage 1: ConvNeXt (18ch)...")
    conv_cfg = {'experiment': {'input_channels': 18, 'num_classes': 2, 'model_variant': 'convnext_tiny.fb_in1k', 'pretrained': False}}
    models['ConvNeXt'] = ConvNeXtAdapter(conv_cfg, model_path=get_weight_path("ConvNeXt_weight")).model.eval().to(device)

    print("Loading Stage 2: MariNeXt (18ch)...")
    seg_cfg = {'experiment': {'input_channels': 18, 'model_name': 'MariNext'}}
    models['MariNeXt'] = MarinextAdapter(seg_cfg, model_path=get_weight_path("MariNeXt_weight")).model.eval().to(device)
    
    print("Loading Baseline: MariNeXt (11ch, 15-class)...")
    base_net = MariNext(11, 15)
    ckpt = torch.load(get_weight_path("baseline_weight"), map_location=device)
    base_net.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    models['Baseline'] = base_net.eval().to(device)
    
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"⚠️ Model loading warning: {e}")

def get_band_val(path): return int(path.split('_')[-2])

def load_18ch_tensor_and_gt(roi_name):
    parts = roi_name.rsplit('_', 1)
    scene_folder, patch_idx = parts[0], parts[1]
    search_pattern = os.path.join(MADOS_DATA_ROOT, scene_folder, '*', f"*_L2R_rhorc_*_{patch_idx}.tif")
    all_bands = sorted(glob.glob(search_pattern), key=get_band_val)
    if len(all_bands) < 11: raise ValueError("Incomplete 11 bands.")

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
            
    rgb_img = np.stack([img[1], img[2], img[3]], axis=-1)
    return tensor_18, gt_mask, rgb_img

def generate_mask_b64(mask, gt_mask, mode='gt'):
    h, w = gt_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = (gt_mask > 0) 
    
    color_oil = [153, 204, 255, 255]    
    color_others = [0, 255, 0, 255]     
    color_bg = [102, 25, 25, 255]       
    
    if mode == 'gt':
        rgba[:] = color_bg 
        rgba[(gt_mask > 0) & (gt_mask != 6)] = color_others 
        rgba[gt_mask == 6] = color_oil 
    elif mode == 'pred':
        rgba[:] = color_bg 
        rgba[(valid) & (mask == 0)] = color_others
        rgba[(valid) & (mask == 1)] = color_oil
    elif mode == 'overlay':
        rgba[(valid) & (mask == 0)] = [0, 255, 0, 200]
        rgba[(valid) & (mask == 1)] = [153, 204, 255, 200]
        
    _, buffer = cv2.imencode('.png', rgba)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/")
def serve_index(): return FileResponse("index.html")

@app.get("/api/thumbnail/{roi_name}")
def get_thumbnail(roi_name: str):
    try:
        parts = roi_name.rsplit('_', 1)
        scene_folder, patch_idx = parts[0], parts[1]
        search_pattern = os.path.join(MADOS_DATA_ROOT, scene_folder, '*', f"*_L2R_rhorc_*_{patch_idx}.tif")
        all_bands = sorted(glob.glob(search_pattern), key=get_band_val)
        if len(all_bands) < 4: return Response(status_code=404)
        
        b_path, g_path, r_path = all_bands[1], all_bands[2], all_bands[3]
        with rasterio.open(r_path) as r_src, rasterio.open(g_path) as g_src, rasterio.open(b_path) as b_src:
            r_img = r_src.read(1, out_shape=(64, 64), resampling=Resampling.nearest)
            g_img = g_src.read(1, out_shape=(64, 64), resampling=Resampling.nearest)
            b_img = b_src.read(1, out_shape=(64, 64), resampling=Resampling.nearest)
            
        rgb_img = np.stack([b_img, g_img, r_img], axis=-1)
        p2, p98 = np.percentile(rgb_img, (2, 98))
        rgb_norm = np.clip((rgb_img - p2) / (p98 - p2 + 1e-7), 0, 1) * 255
        
        _, buffer = cv2.imencode('.jpg', rgb_norm.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    except Exception:
        return Response(status_code=404)

@app.get("/api/get_random_selection")
def get_random_selection():
    splits_dir = os.path.join(MADOS_DATA_ROOT, "splits")
    def read_clean(filepath):
        if not os.path.exists(filepath): return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.split('|')[0].strip() for line in f if line.strip()]

    perfect_list = read_clean(os.path.join(splits_dir, "demo_all_perfect.txt"))
    failed_list = read_clean(os.path.join(splits_dir, "demo_baseline_failed.txt"))
    high_fp_list = read_clean(os.path.join(splits_dir, "demo_baseline_fp.txt"))
    s1_list = read_clean(os.path.join(splits_dir, "demo_s1_perfect_block.txt"))

    # 🚨 關鍵隔離：將「高誤報名單」從「常規無油名單」中徹底剔除，保證題庫互斥
    s1_pure_list = [roi for roi in s1_list if roi not in high_fp_list]

    # 🚨 抽取 20 張黃金比例：7 模範生、5 漏判對比、6 高誤報打臉、2 常規無油防守
    sel_perfect = random.sample(perfect_list, min(7, len(perfect_list))) if perfect_list else []
    sel_failed = random.sample(failed_list, min(5, len(failed_list))) if failed_list else []
    sel_high_fp = random.sample(high_fp_list, min(6, len(high_fp_list))) if high_fp_list else []
    sel_s1 = random.sample(s1_pure_list, min(2, len(s1_pure_list))) if s1_pure_list else []

    selected_rois = sel_perfect + sel_failed + sel_high_fp + sel_s1
    random.shuffle(selected_rois) # 徹底洗牌，讓前端看不出規律
    
    return {"selected": selected_rois}

@app.get("/api/get_test_list")
def get_test_list():
    txt_path = os.path.join(MADOS_DATA_ROOT, "splits", "test_demo_pool.txt")
    oil_txt_path = os.path.join(MADOS_DATA_ROOT, "splits", "test_has_oil.txt")
    
    oil_set = set()
    if os.path.exists(oil_txt_path):
        with open(oil_txt_path, 'r', encoding='utf-8') as f:
            oil_set = {line.strip() for line in f if line.strip()}
            
    results = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                roi = line.strip()
                if roi: results.append({"name": roi, "has_oil": roi in oil_set})
    return {"files": results}

@app.get("/api/load_image/{roi_name}")
def load_image(roi_name: str):
    try:
        _, gt_mask, rgb_img = load_18ch_tensor_and_gt(roi_name)
        p2, p98 = np.percentile(rgb_img, (2, 98))
        rgb_norm = np.clip((rgb_img - p2) / (p98 - p2 + 1e-7), 0, 1) * 255
        _, buffer = cv2.imencode('.png', rgb_norm.astype(np.uint8))
        rgb_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {"rgb_base64": rgb_b64, "gt_base64": generate_mask_b64(None, gt_mask, mode='gt')}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/evaluate/{roi_name}")
def evaluate_patch(roi_name: str, stage: str = "hybrid", model: str = "EB2"):
    try:
        tensor_18, gt_mask, _ = load_18ch_tensor_and_gt(roi_name)
        tensor_18 = tensor_18.to(device)
        
        pred_mask = np.full_like(gt_mask, -1) 
        valid = (gt_mask > 0) 
        
        metrics = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        prob = 1.0
        status_msg = ""
        is_candidate = True

        with torch.no_grad():
            if stage == "baseline":
                base_model = models.get('Baseline')
                tensor_11 = tensor_18[:, :11, :, :] 
                out = base_model(tensor_11)
                out = F.interpolate(out, size=gt_mask.shape, mode='bilinear', align_corners=True)
                pred_15 = out.argmax(1)[0].cpu().numpy()
                
                pred_mask[valid] = 0 
                pred_mask[(valid) & (pred_15 == 5)] = 1 
                status_msg = "Baseline (11ch) Completed"
                prob = 1.0
            else:
                cls_model = models.get(model)
                out_cls = cls_model(tensor_18)
                prob = F.softmax(out_cls[0] if isinstance(out_cls, tuple) else out_cls, dim=1)[0, 1].item()
                is_candidate = prob >= 0.20
                status_msg = "Candidate (Pending S2)" if is_candidate else "Excluded (Filtered)"
                
                if stage == "stage1" or not is_candidate:
                    pred_mask[valid] = 1 if is_candidate else 0

                elif stage == "hybrid" and is_candidate:
                    seg_model = models.get('MariNeXt')
                    out_seg = seg_model(tensor_18)
                    out_seg = F.interpolate(out_seg, size=gt_mask.shape, mode='bilinear', align_corners=True)
                    raw_pred = out_seg.argmax(1).cpu().numpy().squeeze()
                    pred_mask[valid] = raw_pred[valid]
                    status_msg = "Two-Stage Completed"

        if valid.any():
            y_true = (gt_mask[valid] == 6).astype(int) 
            y_pred = pred_mask[valid]
            
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            
            metrics.update({"tp": tp, "fp": fp, "tn": tn, "fn": fn})

        mask_pred_b64 = generate_mask_b64(pred_mask, gt_mask, mode='pred')
        mask_overlay_b64 = generate_mask_b64(pred_mask, gt_mask, mode='overlay')
        
        return {"status": status_msg, "prob": prob, "is_candidate": is_candidate, "metrics": metrics, "mask_pred_b64": mask_pred_b64, "mask_overlay_b64": mask_overlay_b64}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
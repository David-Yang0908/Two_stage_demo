# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou (Modified for NCU SAILY Lab)
Description: 高性能多通道資料載入器 - 支援 11/14/18 通道動態切換版。
             🚀 核心更新：支援「外部標準化參數」傳入，解決 Fold 間數據偏移問題。
             🚀 通道配置：11 (原始波段), 14 (11+核心指數), 18 (11+完整指數)。
             🚀 標籤映射：油汙 (ID 6) -> 1, 其他標記 (ID 1-5, 7-15) -> 0, 其餘 -> -1。
'''
import os
import torch
import random
from glob import glob
import numpy as np
from tqdm import tqdm
from osgeo import gdal
import rasterio
from rasterio.enums import Resampling
from torch.utils.data import Dataset

# 固定隨機種子確保實驗可重複性
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# -------------------------------------------------------------------------
# 1. 預設 18 通道標準化參數 (若外部未傳入則使用此組全域平均值)
# -------------------------------------------------------------------------
# 索引順序：[0-10: 原始波段] [11:NDVI, 12:NDWI, 13:NDMI, 14:FAI, 15:FDI, 16:SI, 17:BSI]
DEFAULT_BANDS_MEAN = np.array([
    0.058121473, 0.05210267, 0.04370499, 0.0356187, 0.034042913, 0.03671142, 
    0.039890528, 0.03557705, 0.03955101, 0.026731603, 0.019739248, 
    -0.107859276, 0.24759273, 0.21211773, 0.0016443103, 4.815468e-05, 0.9329618, -0.2540545 
]).astype('float32')

DEFAULT_BANDS_STD = np.array([
    0.03249658, 0.034378916, 0.03550433, 0.037572216, 0.03784507, 0.049894392, 
    0.058804873, 0.055417728, 0.06418743, 0.042080186, 0.030169226, 
    0.16748442, 0.22345243, 0.17628014, 0.036581792, 0.011342162, 0.06623049, 0.16210075 
]).astype('float32')

# 油汙比例 (用於 Loss 權重平衡)
class_distr_binary = torch.Tensor([0.8416, 0.1584])

def get_band(path):
    """ 從檔名中提取波段編號用於排序 """
    return int(path.split('_')[-2])

class MADOS(Dataset): 
    def __init__(self, data_path, split='train', task='seg', input_channels=18, split_prefix=None, mean=None, std=None): 
        """
        Args:
            data_path: 資料集根目錄
            split: 'train', 'val', 或 'test'
            task: 'seg' (分割), 'cls' (分類), 'both' (同時)
            input_channels: 11, 14, 或 18 通道模式
            split_prefix: Fold 前綴 (如 'fold1')
            mean: 外部傳入的平均值列表 (長度應為 18)
            std: 外部傳入的標準差列表 (長度應為 18)
        """
        self.data_path = data_path
        self.split = split
        self.task = task
        self.input_channels = input_channels 
        self.split_prefix = split_prefix     
        
        # 🚀 核心更新：優先採用外部傳入的統計參數，若無則採用全域預設值
        temp_mean = np.array(mean) if mean is not None else DEFAULT_BANDS_MEAN
        temp_std = np.array(std) if std is not None else DEFAULT_BANDS_STD

        # 根據 input_channels 模式自動裁切統計參數
        if self.input_channels == 14:
            # 14ch 模式：原始 11 波段 + 索引 14, 15, 16 (FAI, FDI, SI)
            self.active_mean = np.concatenate([temp_mean[:11], temp_mean[14:17]])
            self.active_std = np.concatenate([temp_std[:11], temp_std[14:17]])
        else:
            # 11ch 或 18ch 模式：直接取對應長度
            self.active_mean = temp_mean[:input_channels]
            self.active_std = temp_std[:input_channels]
        
        if mean is not None:
            print(f"✅ MADOS Dataset: 已套用 Fold 專屬標準化參數 (模式: {self.input_channels}ch)")

        # 動態構建分層抽樣 (Split) 檔案路徑
        filename = f"{split_prefix}_{split}_X.txt" if split_prefix else f"{split}_X.txt"
        split_file = self._locate_split_file(data_path, filename)

        if os.path.exists(split_file):
            print(f"📂 載入 Split 清單: {os.path.basename(split_file)}")
            self.ROIs_split = np.genfromtxt(split_file, dtype='str')
        else:
            raise FileNotFoundError(f"❌ 找不到關鍵 Split 檔案: {filename}")

        self.X, self.y, self.roi_names = [], [], []
        self.tiles = [t for t in glob(os.path.join(data_path, '*')) if os.path.isdir(t)]

        # ---------------------------------------------------------------------
        # 數據預加載 (保持原始邏輯)
        # ---------------------------------------------------------------------
        for tile in tqdm(self.tiles, desc=f'Preloading {split} ({input_channels}ch)'):
            crops = [f.split('_cl_')[-1] for f in glob(os.path.join(tile, '10', '*_cl_*'))]
            
            for crop in crops:
                crop_name = os.path.basename(tile) + '_' + crop.split('.tif')[0]
                
                if crop_name in self.ROIs_split:
                    self.roi_names.append(crop_name)
                    all_bands = sorted(glob(os.path.join(tile, '*', f'*L2R_rhorc*_{crop}')), key=get_band)
        
                    current_image = []
                    for band in all_bands:
                        upscale_factor = int(os.path.basename(os.path.dirname(band))) // 10
                        with rasterio.open(band, mode='r') as src:
                            data = src.read(1, out_shape=(int(src.height * upscale_factor), 
                                                        int(src.width * upscale_factor)),
                                            resampling=Resampling.nearest).copy()
                            current_image.append(data)
                    
                    if len(current_image) == 11:
                        stacked_11 = np.stack(current_image) 
                        
                        # 核心邏輯：根據 input_channels 決定是否疊加光譜指數
                        final_data = self.add_indices(stacked_11) if self.input_channels > 11 else stacked_11
                        self.X.append(final_data)

                        # 標籤映射：對齊 MADOS 標註協議 (ID 6 為油汙)
                        cl_path = os.path.join(tile, '10', f'{os.path.basename(tile)}_L2R_cl_{crop}')
                        with rasterio.open(cl_path) as src:
                            raw_mask = src.read(1).astype(np.int64)
                        
                        binary_mask = np.full_like(raw_mask, -1, dtype=np.int64)
                        binary_mask[raw_mask == 6] = 1 
                        others_mask = (raw_mask >= 1) & (raw_mask <= 15) & (raw_mask != 6)
                        binary_mask[others_mask] = 0
                        self.y.append(binary_mask)
            
        self.X = np.stack(self.X)
        self.y = np.stack(self.y)
        self.length = len(self.y)
        # 用於處理缺失值的平均數矩陣
        self.impute_nan = np.tile(self.active_mean, (self.X.shape[-2], self.X.shape[-1], 1))
        
    def _locate_split_file(self, data_path, filename):
        """ 多重路徑匹配邏輯，確保能找到 splits 檔案 """
        paths = [
            os.path.join(data_path, filename),
            os.path.join(os.path.dirname(data_path), filename),
            os.path.join(data_path, 'splits', filename)
        ]
        for p in paths:
            if os.path.exists(p): return p
        return paths[0]

    def add_indices(self, image):
        """ 🚀 計算光譜指數：嚴格對應 18 通道物理公式 """
        eps = 1e-8
        # 波段解構：Band 2(1), 3(2), 4(3), 6(5), 8(7), 11(9)
        blue, green, red = image[1], image[2], image[3] 
        re2, nir, swir1 = image[5], image[7], image[9]

        # 核心分割指數 (FAI/FDI/SI)
        fai = nir - (red + (swir1 - red) * 0.19)
        fdi = nir - (re2 + (swir1 - re2) * 0.1186)
        si  = np.sqrt(np.clip((1-blue)*(1-green)*(1-red), 0, None))

        if self.input_channels == 14:
            # 14ch 模式：11 波段 + FAI, FDI, SI
            indices = np.stack([fai, fdi, si])
            return np.concatenate([image, indices], axis=0)

        elif self.input_channels == 18:
            # 18ch 模式：完整 7 種環境監測指標
            ndvi = (nir - red) / (nir + red + eps)
            ndwi = (green - nir) / (green + nir + eps)
            ndmi = (nir - swir1) / (nir + swir1 + eps)
            bsi  = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)
            
            # 注意：此堆疊順序必須與 DEFAULT_BANDS_MEAN 索引 11-17 完美對應
            indices = np.stack([ndvi, ndwi, ndmi, fai, fdi, si, bsi])
            return np.concatenate([image, indices], axis=0)
        
        return image

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.X[index]
        mask = self.y[index]
        roi_name = self.roi_names[index] 

        # 轉換為 (H, W, C) 以利於增強操作
        image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]).astype('float32')
        
        # 缺失值填充
        nan_mask = np.isnan(image)
        image[nan_mask] = self.impute_nan[nan_mask]
        
        # 訓練集數據增強 (隨機翻轉與旋轉)
        if self.split == 'train':
            mask_expanded = mask[:, :, np.newaxis]
            image, mask_expanded = self.join_transform(image, mask_expanded)
            mask = mask_expanded.squeeze()

        image, mask = image.copy(), mask.copy()

        # 🚀 關鍵：根據當前 Fold 專屬參數進行標準化
        image_final = (image.transpose(2, 0, 1) - self.active_mean.reshape(-1, 1, 1)) / self.active_std.reshape(-1, 1, 1)
        image_final = torch.from_numpy(image_final).float()

        if self.task == 'cls':
            label = 1 if np.any(mask == 1) else 0
            return image_final, torch.tensor(label, dtype=torch.long), roi_name
        elif self.task == 'seg':
            return image_final, torch.tensor(mask, dtype=torch.long), roi_name
        elif self.task == 'both':
            label = 1 if np.any(mask == 1) else 0
            return image_final, torch.tensor(label, dtype=torch.long), \
                   torch.tensor(mask, dtype=torch.long), roi_name

    def join_transform(self, image, target):
        """ 實作隨機旋轉與翻轉 """
        f = [1, 0, -1, 2, 2][np.random.randint(0, 5)]
        if f != 2:
            image, target = np.flip(image, f), np.flip(target, f)
        if np.random.random() < 0.8:
            k = np.random.randint(0, 4)
            image, target = np.rot90(image, k, (1, 0)), np.rot90(target, k, (1, 0))
        return image, target

def gen_weights(class_distribution=class_distr_binary, c=1.02):
    """ 基於類別分佈生成平衡權重 (可用於 CrossEntropyLoss) """
    return 1 / torch.log(c + class_distribution)
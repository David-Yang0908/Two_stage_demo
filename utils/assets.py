# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou (Modified for NCU SAILY Lab)
Description: 包含 MADOS 資料集的映射工具、標準化參數與排程器。
             🚀 核心更新：對齊最新 dataset.py 之 18 通道實測統計數據。
             🚀 邏輯鎖定：採用論文 Table 3 標記像素比例，解決背景稀釋問題。
'''
import numpy as np
import torch 

# -------------------------------------------------------------------------
# 1. 18 通道標準化參數 (由 calculate_stats.py 實測訓練集得出)
# -------------------------------------------------------------------------
# 前 11 通道為 Sentinel-2 原始波段，後 7 通道為計算出的光譜指數 (NDVI ~ BSI)
# 🚀 這些數值必須與 dataset.py 保持絕對一致，以確保推理與訓練環境對齊
bands_mean = np.array([
    0.058121473, 0.05210267, 0.04370499, 0.0356187, 0.034042913, 0.03671142, 
    0.039890528, 0.03557705, 0.03955101, 0.026731603, 0.019739248, # 原始 11 波段
    -0.107859276, 0.24759273, 0.21211773, 0.0016443103, 4.815468e-05, 0.9329618, -0.2540545 # 7 指數實測均值
]).astype('float32')

bands_std = np.array([
    0.03249658, 0.034378916, 0.03550433, 0.037572216, 0.03784507, 0.049894392, 
    0.058804873, 0.055417728, 0.06418743, 0.042080186, 0.030169226, # 原始 11 波段
    0.16748442, 0.22345243, 0.17628014, 0.036581792, 0.011342162, 0.06623049, 0.16210075 # 7 指數實測標差
]).astype('float32')

# -------------------------------------------------------------------------
# 2. 🚀 二元任務映射 (對齊論文「標記像素對抗」邏輯)
# -------------------------------------------------------------------------
# 映射說明：
# - 索引 1: Oil Spill (油汙，原始 ID 6)
# - 索引 0: Other Annotated (除了油汙以外的其他 14 類標記像素)
# - 索引 -1: Ignored (未標記之海水區域，Loss 不進行計算)

binary_labels = ['Other_Annotated', 'Oil_Spill']
binary_cat_mapping = {'Other_Annotated': 0, 'Oil_Spill': 1}

# 🚀 根據論文 Table 3 數據精算之標記像素分佈 (Annotated-only) [cite: 158, 201]：
# 總標記像素 (Total Annotated): 1,481,155
# 油汙像素 (Oil Spill): 234,568 (佔 15.84%)
# 其他標記物 (Other Annotated): 1,246,587 (佔 84.16%)
# 這樣做能讓油汙權重回歸正常範圍，不再被 3,700 萬海水稀釋到 0.1%
class_distr_binary = torch.Tensor([0.8416, 0.1584])

# 相容性別名 (供評估與混合模組調用)
labels_3class = binary_labels
class_distr_3class = class_distr_binary

def gen_weights(class_distribution=class_distr_binary, c=1.02):
    """ 
    計算加權 CrossEntropy 的權重參數。
    c 值建議維持 1.02 (對齊論文)，若模型對油汙仍不夠敏感可調至 1.01。
    """
    return 1 / torch.log(c + class_distribution)

# -------------------------------------------------------------------------
# 3. 原始 15 類別映射 (MADOS 官方定義，僅供邏輯參考)
# -------------------------------------------------------------------------
mados_cat_mapping = {
    'Marine Debris': 1, 'Dense Sargassum': 2, 'Sparse Floating Algae': 3,
    'Natural Organic Material': 4, 'Ship': 5, 'Oil Spill': 6,
    'Marine Water': 7, 'Sediment-Laden Water': 8, 'Foam': 9,
    'Turbid Water': 10, 'Shallow Water': 11, 'Waves & Wakes': 12,
    'Oil Platform': 13, 'Jellyfish': 14, 'Sea snot': 15
}

labels = list(mados_cat_mapping.keys())

# -------------------------------------------------------------------------
# 4. 視覺化顏色映射 (用於產出預測圖)
# -------------------------------------------------------------------------
mados_color_mapping = {
    'Marine Debris': 'red', 'Dense Sargassum': 'green', 'Sparse Floating Algae': 'limegreen',
    'Natural Organic Material': 'brown', 'Ship': 'orange', 'Oil Spill': 'thistle',
    'Marine Water': 'navy', 'Sediment-Laden Water': 'gold', 'Foam': 'purple',
    'Turbid Water': 'darkkhaki', 'Shallow Water': 'darkturquoise', 'Waves & Wakes': 'bisque',
    'Oil Platform': 'dimgrey', 'Jellyfish': 'hotpink', 'Sea snot': 'yellow'
}

# -------------------------------------------------------------------------
# 5. 波段索引映射 (18 通道完整版)
# -------------------------------------------------------------------------
s2_mapping = {
    'nm440': 0, 'nm490': 1, 'nm560': 2, 'nm665': 3, 'nm705': 4, 'nm740': 5,
    'nm783': 6, 'nm842': 7, 'nm865': 8, 'nm1600': 9, 'nm2200': 10,
    'NDVI': 11, 'NDWI': 12, 'NDMI': 13, 'FAI': 14, 'FDI': 15, 'SI': 16, 'BSI': 17
}

# -------------------------------------------------------------------------
# 6. 訓練工具函式
# -------------------------------------------------------------------------
def bool_flag(s):
    """ 解析命令行布林參數 """
    import argparse
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS: return False
    elif s.lower() in TRUTHY_STRINGS: return True
    else: raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """ 
    餘弦退火學習率排程器。
    確保學習率在訓練末期平滑下降，有利於優化器跳出局部最優點。
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule
import os
import rasterio
import numpy as np
from tqdm import tqdm

# 設定資料集路徑
MADOS_ROOT = r"C:\Users\楊大正\OneDrive\文件\Runspace\MADOS"
test_txt = os.path.join(MADOS_ROOT, "splits", "test_X.txt")
oil_txt = os.path.join(MADOS_ROOT, "splits", "test_has_oil.txt")

def main():
    print("🔍 開始掃描測試集標籤，建立油污清單...")
    
    # 讀取原本的總測試清單
    with open(test_txt, 'r', encoding='utf-8') as f:
        rois = [line.strip() for line in f if line.strip()]

    oil_rois = []
    
    # 使用 tqdm 顯示掃描進度條
    for roi in tqdm(rois, desc="掃描進度"):
        parts = roi.rsplit('_', 1)
        scene_folder, patch_idx = parts[0], parts[1]
        cl_path = os.path.join(MADOS_ROOT, scene_folder, "10", f"{scene_folder}_L2R_cl_{patch_idx}.tif")
        
        # 讀取標籤並檢查是否有油污 (6)
        if os.path.exists(cl_path):
            with rasterio.open(cl_path) as src:
                data = src.read(1)
                if np.any(data == 6):
                    oil_rois.append(roi)

    # 將有油污的名單寫入獨立的 txt 檔
    with open(oil_txt, 'w', encoding='utf-8') as f:
        for roi in oil_rois:
            f.write(f"{roi}\n")
            
    print(f"✅ 掃描完成！共有 {len(rois)} 張圖磚，其中 {len(oil_rois)} 張包含油污。")
    print(f"📄 名單已儲存至: {oil_txt}")

if __name__ == "__main__":
    main()
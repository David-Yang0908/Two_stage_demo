# Marine Eye - 二階段海洋油污辨識系統 (互動展示雛形)

本儲存庫 (Repository) 包含「結合多光譜特徵與二階段深度學習架構之海洋油污辨識」框架的前端互動可視化介面與後端推論 API。

> ⚠️ **重要提示**
> 本專案專為**視覺化展示與互動式推論**而設計，旨在具體呈現演算法的商業落地可行性與系統雛形。系統直接載入預訓練權重進行推論，**本儲存庫不包含任何模型訓練的程式碼或腳本**。

## 系統架構概述 (System Overview)
本系統實作了「由粗到細」(Coarse-to-fine) 的二階段深度學習管線：
1. **第一階段 (分類模型 Classification)：** 作為輕量化的前線守門員 (EfficientNetB2 / ConvNeXt)，負責快速濾除大面積乾淨海面與船舶尾跡等干擾雜訊。
2. **第二階段 (分割模型 Segmentation)：** 高精度的像素級分割網路 (MariNeXt)，專門針對通過第一階段篩選的高風險候選影像，進行精細的像素級邊界描繪。

核心後端採用 `app.py` (FastAPI) 構建，負責處理 18 通道的高維度多光譜張量 (11 個原始 Sentinel-2 波段 + 7 個自行衍生的物理指標)；前端 `index.html` 則提供動態的互動介面，包含影像過濾動態展示、預測遮罩疊加 (Overlay)，以及即時的效能指標計算 (IoU, Recall, Precision)。

## 資料集與模型權重 (Dataset & Model Weights)
受限於 GitHub 嚴格的檔案大小限制，龐大的原始資料集與模型權重檔存放於外部雲端空間。在運行系統前，請先下載以下檔案：

1. **原始資料集 (Dataset):** 
   🔗 [MADOS Dataset on Zenodo](https://zenodo.org/records/10664073)
2. **預訓練模型權重 (Pre-trained Weights):**
   🔗 [從 Google Drive 下載權重檔](https://drive.google.com/drive/folders/1U9_BiNAJWvX17-DApESDnQPPVHzS7Hsw?usp=sharing)

**配置說明：** 請將下載好的權重檔案 (`baseline_weight.pth`, `ConvNeXt_weight`, `EB2_weight`) 直接放置於本專案的根目錄中，系統即可自動讀取。

## 安裝與執行步驟 (Installation & Setup)

1. **複製專案至本機 (Clone the repository):**
   ```bash
   git clone [https://github.com/David-Yang0908/Two_stage_demo.git](https://github.com/David-Yang0908/Two_stage_demo.git)
   cd Two_stage_demo

2.  **安裝環境依賴套件 (Install dependencies):**
  ```Bash
  pip install -r requirements.txt

3. **啟動展示伺服器 (Run the Demo Server):**
  ```Bash
  python app.py
  
  伺服器啟動後，請開啟瀏覽器並前往 http://127.0.0.1:8000 即可操作互動介面。
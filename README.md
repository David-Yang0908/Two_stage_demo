# Marine Eye - 二階段海洋油污辨識與 AI 互動展示系統

本儲存庫 (Repository) 包含「結合多光譜特徵與二階段深度學習架構之海洋油污辨識」框架的前端互動可視化介面、後端推論 API，以及支援 AI 代理協作的 MCP (Model Context Protocol) 伺服器。

> ⚠️ **重要提示**
> 本專案專為**視覺化展示與互動式推論**而設計，旨在具體呈現演算法的商業落地可行性與系統雛形。系統直接載入預訓練權重進行推論，**本儲存庫不包含任何模型訓練的程式碼或腳本**。

## 系統架構概述 (System Overview)
本系統實作了「由粗到細」(Coarse-to-fine) 的二階段深度學習管線，並結合了 AI 代理 (AI Agent) 擴充介面：
1. **第一階段 (分類模型 Classification)：** 作為輕量化的前線守門員 (EfficientNetB2 / ConvNeXt)，負責快速濾除大面積乾淨海面與船舶尾跡等干擾雜訊。
2. **第二階段 (分割模型 Segmentation)：** 高精度的像素級分割網路 (MariNeXt)，專門針對通過第一階段篩選的高風險候選影像，進行精細的像素級邊界描繪。
3. **AI 代理伺服器 (MCP Server)：** 透過獨立的微服務架構，允許外部 LLM (如 Cursor / Claude) 呼叫底層推論 API，自動執行批次評估與數據總結。

## 資料集與模型權重 (Dataset & Model Weights)
受限於 GitHub 嚴格的檔案大小限制，龐大的原始資料集與模型權重檔存放於外部雲端空間。在運行系統前，請先下載以下檔案：
1. **原始資料集 (Dataset):** 🔗 [MADOS Dataset on Zenodo](https://zenodo.org/records/10664073)
2. **預訓練模型權重 (Pre-trained Weights):** 🔗 [從 Google Drive 下載權重檔](https://drive.google.com/drive/folders/1U9_BiNAJWvX17-DApESDnQPPVHzS7Hsw?usp=sharing)

**配置說明：** 請將下載好的權重檔案 (`baseline_weight.pth`, `ConvNeXt_weight`, `EB2_weight`) 直接放置於本專案的根目錄中，系統即可自動讀取。

## 雙環境安裝與執行步驟 (Dual-Environment Setup)

為確保底層 C++ 編譯的機器學習套件 (PyTorch) 與最新版的 MCP SDK 2.x 互不干擾，本專案採用**環境解耦 (Decoupling)** 設計。請分別建立兩個虛擬環境：

### 環境一：後端推論伺服器 (Python 3.8+)
負責處理多光譜張量運算與提供 RESTful API。
1. **建立並啟動環境：**
   ```bash
   conda create -n mados_demo python=3.8 -y
   conda activate mados_demo

2. **安裝依賴並啟動伺服器：**
   ```bash
   pip install -r requirements.txt
   python app.py
   伺服器啟動後，開啟瀏覽器前往 http://127.0.0.1:8000 即可操作互動介面。

### 環境二：MCP 代理伺服器 (需 Python 3.10+)
負責與外部 AI (如 Cursor) 溝通，並透過 HTTP 請求呼叫環境一的推論 API。

1. **建立並啟動環境：**
   ```bash
   conda create -n mcp_demo python=3.11 -y
   conda activate mcp_demo
2. **安裝 MCP 依賴：**

   ```bash
   pip install mcp httpx
## Cursor MCP 整合指南 (AI Agent Integration)
要讓 AI 自動幫你跑數據，請在 Cursor 中進行以下配置：

1. 確保環境一的 app.py 正在背景運行 (Port 8000)。

2. 打開 Cursor 設定 > Features > MCP > 點擊 + Add New MCP Server。

3. 填入以下資訊：
   ```bash
   Name: MarineEye
   Type: command
   Command: 填入環境二的 Python 絕對路徑 (如 C:\anaconda\envs\mcp_demo\python.exe)
   Args: 填入 mcp_server.py 的絕對路徑 (如 C:\Users\...\Two_stage_demo\mcp_server.py)


4. 儲存並確認亮起綠燈後，即可在 Chat 視窗輸入指令（例如：「請呼叫 evaluate_random_20_images 工具，自動送 20 張圖片進行二階段評估並總結指標」），讓 AI 成為你的專屬數據分析師！
# -*- coding: utf-8 -*-
import os
import sys
import torch
import logging 
import shutil # 🚀 用於權重檔案的位移操作
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from timm.utils import ModelEma
import timm  # 🚀 使用 timm 載入高效能分類模型
from sklearn.metrics import recall_score, jaccard_score # 🚀 引入平衡指標計算工具

# -------------------------------------------------------------------------
# 1. 專案路徑與環境配置
# -------------------------------------------------------------------------
MADOS_ROOT = Path(os.getcwd()) 
# 確保環境路徑包含自定義模組資料夾，防止導入錯誤
if str(MADOS_ROOT / 'marinext') not in sys.path:
    sys.path.append(str(MADOS_ROOT / 'marinext'))

# 導入數據增強 (VSCP)、學習率調度工具
from utils.vscp import VSCP
from utils.assets import cosine_scheduler
# 🚀 載入支援 11/14/18 通道動態切換的 MADOS 資料集與指標工具
from utils.dataset import MADOS, gen_weights, class_distr_binary
from utils.metrics import Evaluation

# -------------------------------------------------------------------------
# 2. Focal Loss：解決樣本不平衡並防止模型崩潰的損失函數
# -------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss 核心機制：
    🚀 採用 alpha=0.5 的折衷配置。
    這能在重視油汙目標的同時，保留對背景（海水）特徵的學習壓力，避免模型全部預測為油汙。
    """
    def __init__(self, alpha=0.5, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha     # 🚀 設為 0.5 以平衡正負樣本關注度
        self.gamma = gamma     # 難度關注因子，強迫模型學習難分的邊界樣本
        self.weight = weight   # 來自 gen_weights 的類別補償

    def forward(self, inputs, targets):
        # 計算基礎 Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        # 計算預測正確的機率 pt
        pt = torch.exp(-ce_loss)
        # 應用 Focal 權重縮放
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

# -------------------------------------------------------------------------
# 3. EfficientNetB2Adapter 核心類別 (指標平衡與 Top-K 儲存版)
# -------------------------------------------------------------------------
class EfficientNetB2Adapter(nn.Module):
    """
    EfficientNet-B2 分類適配器：
    🚀 核心功能更新：
    1. 指標監控切換為 Recall 與 IoU 的平衡分數。
    2. 儲存機制對齊 Seg 模型，支援 Top 1-3 與 Last 權重管理。
    3. 判定閾值調整為 0.35 以優化分類邊界。
    """
    def __init__(self, config, model_path=None):
        """ 初始化適配器：建立對應通道數的模型架構 """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        exp_cfg = config.get('experiment', {})
        self.in_chans = exp_cfg.get('input_channels', 11) 
        self.num_classes = exp_cfg.get('num_classes', 2)
        
        pretrained = exp_cfg.get('pretrained', True)
        logging.info(f"🎨 初始化 EB2 分類器 (平衡模式): {self.in_chans}ch")
        
        self.model = timm.create_model(
            'efficientnet_b2', 
            pretrained=pretrained, 
            in_chans=self.in_chans, 
            num_classes=self.num_classes
        )
        self.model.to(self.device)
        
        if model_path and Path(model_path).exists():
            logging.info(f"   [Init] 載入權重檔案: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    # -------------------------------------------------------------------------
    # 3.1 訓練與日誌紀錄系統
    # -------------------------------------------------------------------------
    def _setup_logger(self, base_name, run_id):
        """ 配置實驗專屬訓練日誌紀錄檔 """
        log_dir = MADOS_ROOT / "logs" / base_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{base_name}_{run_id}_cls_train.log"
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    def train(self, data_path, train_cfg, exp_cfg):
        """ 
        主訓練流程：執行 Top-K 篩選機制，並監控 Recall 與 IoU 的折衷指標。
        """
        base_name, run_id = exp_cfg['base_name'], exp_cfg['run_id']
        project = exp_cfg.get('results_base_dir', 'results-cls') 
        self._setup_logger(base_name, run_id)
        
        # 初始化路徑
        weight_dir = Path(project) / base_name / "weight" / run_id
        weight_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = Path(project) / base_name / run_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        fold_prefix = train_cfg.get('split_prefix', None)

        logging.info("="*60)
        logging.info(f"🚀 啟動 EB2 分類訓練 (指標: Recall & IoU 平衡模式)")
        logging.info(f"🔍 模式: {self.in_chans}ch | 基準: (Recall + Oil_IoU) / 2")
        logging.info("="*60)

        # 超參數讀取
        epochs, imgsz = train_cfg.get('epochs', 100), train_cfg.get('imgsz', 240)
        batch, lr0 = train_cfg.get('batch_size', 16), train_cfg.get('lr0', 0.0001)
        patience, weight_param = train_cfg.get('patience', 35), train_cfg.get('weight_param', 1.0)

        # 資料載入
        train_loader = DataLoader(
            MADOS(data_path, 'train', task='cls', input_channels=self.in_chans, split_prefix=fold_prefix), 
            batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            MADOS(data_path, 'val', task='cls', input_channels=self.in_chans, split_prefix=fold_prefix), 
            batch_size=batch, shuffle=False, num_workers=0
        )

        n_iter_per_epoch = len(train_loader)
        lr_schedule = cosine_scheduler(lr0, 0, epochs, n_iter_per_epoch, warmup_epochs=10)
        ema_scheduler = cosine_scheduler(0.999, 0.999, epochs, n_iter_per_epoch)

        # 🚀 損失函數更換為平衡版 Focal Loss
        weights = gen_weights(class_distr_binary, c=weight_param).to(self.device) if weight_param != 1.0 else None
        criterion = FocalLoss(alpha=0.5, gamma=2.0, weight=weights) 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr0)
        model_ema = ModelEma(self.model, decay=0.999, device=self.device) if train_cfg.get('model_ema', True) else None
        
        train_loss_history, val_loss_history, val_score_history = [], [], []
        
        # 🚀 🚀 核心更新：Top-K 分數追蹤器 (監控折衷分數 current_score) 🚀 🚀
        top_k_scores = {"top1": -1.0, "top2": -1.0, "top3": -1.0}
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (EB2)")

            for it, (images, labels, _) in enumerate(pbar):
                global_it = n_iter_per_epoch * (epoch - 1) + it
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_schedule[global_it]

                if train_cfg.get('vscp', False) and (labels == 1).any():
                    labels_spatial = np.full((labels.shape[0], imgsz, imgsz), -1, dtype=np.int8)
                    for b in range(labels.shape[0]): labels_spatial[b, :, :] = labels[b].item()
                    try:
                        img_a, msk_a = VSCP(images.numpy(), labels_spatial)
                        if img_a is not None:
                            images = torch.cat([images, torch.from_numpy(img_a)])
                            labels = torch.cat([labels, torch.from_numpy(msk_a[:, 0, 0])])
                    except: pass

                images, labels = images.to(self.device), labels.to(self.device).long()
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.get('clip_grad', 1.0))
                optimizer.step()
                
                if model_ema:
                    model_ema.decay = ema_scheduler[global_it]
                    model_ema.update(self.model)
                running_loss += loss.item()

            # 🚀 驗證階段：計算 Recall 與 Oil IoU
            metrics = self.validate(self.model, val_loader, criterion)
            
            # 🚀 核心折衷指標：(Recall + IoU) / 2
            current_score = (metrics['recall'] + metrics['iou']) / 2
            
            train_loss_history.append(running_loss / n_iter_per_epoch)
            val_loss_history.append(metrics['loss'])
            val_score_history.append(current_score)

            logging.info(f"[Epoch {epoch}] Score: {current_score:.4f} | Rec: {metrics['recall']:.4f} | IoU: {metrics['iou']:.4f}")

            # 🚀 🚀 核心更新：Top-1, Top-2, Top-3 與 Last 儲存機制 🚀 🚀
            torch.save(self.model.state_dict(), weight_dir / "last_cls.pth")

            if current_score > top_k_scores["top1"]:
                logging.info(f" 🏆 New Top 1 Score! {current_score:.4f}")
                if (weight_dir / "top2_cls.pth").exists(): shutil.copy2(weight_dir / "top2_cls.pth", weight_dir / "top3_cls.pth")
                if (weight_dir / "top1_cls.pth").exists(): shutil.copy2(weight_dir / "top1_cls.pth", weight_dir / "top2_cls.pth")
                torch.save(self.model.state_dict(), weight_dir / "top1_cls.pth")
                
                top_k_scores["top3"] = top_k_scores["top2"]
                top_k_scores["top2"] = top_k_scores["top1"]
                top_k_scores["top1"] = current_score
                patience_counter = 0 
            
            elif current_score > top_k_scores["top2"]:
                logging.info(f" ⭐ New Top 2 Score! {current_score:.4f}")
                if (weight_dir / "top2_cls.pth").exists(): shutil.copy2(weight_dir / "top2_cls.pth", weight_dir / "top3_cls.pth")
                torch.save(self.model.state_dict(), weight_dir / "top2_cls.pth")
                top_k_scores["top3"] = top_k_scores["top2"]
                top_k_scores["top2"] = current_score
            
            elif current_score > top_k_scores["top3"]:
                logging.info(f" ✨ New Top 3 Score! {current_score:.4f}")
                torch.save(self.model.state_dict(), weight_dir / "top3_cls.pth")
                top_k_scores["top3"] = current_score

            if current_score <= top_k_scores["top1"]:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"🛑 早停觸發：平衡指標連續 {patience} 輪未突破。")
                    break

        self.plot_curves(train_loss_history, val_loss_history, val_score_history, plot_dir, f"{base_name}_{run_id}")
        return {'best_model_path': str(weight_dir / "top1_cls.pth")}

    # -------------------------------------------------------------------------
    # 3.2 驗證指標計算 (調高校準閾值至 0.35)
    # -------------------------------------------------------------------------
    def validate(self, model, loader, criterion):
        """ 評估模型性能：套用 0.35 閾值並產出 Oil Recall 與 Oil IoU """
        model.eval()
        total_loss, all_preds, all_labels = 0, [], []
        threshold = 0.35 # 🚀 對齊 ConvNeXt 的平衡閾值
        
        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(self.device), labels.to(self.device).long()
                outputs = model(images)
                total_loss += criterion(outputs, labels).item()
                
                # 計算機率並判定
                probs = F.softmax(outputs, dim=1)
                preds = (probs[:, 1] > threshold).long() 
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        y_true, y_pred = np.array(all_labels), np.array(all_preds)
        
        # 計算核心平衡指標
        oil_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        oil_iou = jaccard_score(y_true, y_pred, pos_label=1, zero_division=0)
        eval_info = Evaluation(y_pred, y_true)
        
        return {
            'loss': total_loss / len(loader), 
            'recall': oil_recall, 
            'iou': oil_iou,
            'f1': eval_info['macroF1'],
            'acc': (y_pred == y_true).mean()
        }

    def plot_curves(self, t_loss, v_loss, v_score, save_path, full_name):
        """ 繪製訓練損失與平衡指標趨勢圖 """
        epochs_range = range(1, len(t_loss) + 1)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, t_loss, label='Train'); plt.plot(epochs_range, v_loss, label='Val')
        plt.title('Loss History'); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, v_score, color='purple', label='Balanced Score (Val)')
        plt.title('Balanced Metric History'); plt.legend()
        plt.tight_layout()
        plt.savefig(save_path / f"{full_name}_cls_loss_curve.png"); plt.close()
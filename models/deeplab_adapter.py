# -*- coding: utf-8 -*-
import os
import sys
import torch
import logging
import shutil
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from timm.utils import ModelEma

# 🚀 建議安裝：pip install segmentation-models-pytorch
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("❌ 找不到 segmentation_models_pytorch 庫，請執行 'pip install segmentation-models-pytorch'")

# -------------------------------------------------------------------------
# 1. 路徑與環境配置
# -------------------------------------------------------------------------
MADOS_ROOT = Path(os.getcwd()) 
if str(MADOS_ROOT / 'marinext') not in sys.path:
    sys.path.append(str(MADOS_ROOT / 'marinext'))

from utils.vscp import VSCP
from utils.assets import cosine_scheduler
# 導入 MADOS 數據集與權重計算工具
from utils.dataset import MADOS, gen_weights, class_distr_binary

class DeepLabAdapter(nn.Module):
    """
    DeepLabV3+ 模型適配器：
    封裝了從模型初始化、多波段輸入處理、訓練迴圈、Top-K 權重篩選到日誌紀錄的所有邏輯。
    核心架構採用 DeepLabV3+ 搭配可變更之 Encoder (預設 ResNet50)。
    """
    def __init__(self, config, model_path=None):
        """
        初始化適配器：設定硬體運算環境、建立多波段輸入模型、並載入指定權重。
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 🚀 核心參數讀取：支援 11/14/18 通道動態切換
        exp_cfg = config.get('experiment', {})
        self.in_chans = exp_cfg.get('input_channels', 11)
        self.num_classes = 2 # 二元分類：背景 (Others) 與油汙 (Oil)
        
        # 模型架構初始化
        encoder_name = exp_cfg.get('encoder', 'resnet50')
        pretrained_tag = "imagenet" if exp_cfg.get('pretrained', True) else None
        
        logging.info(f"🎨 初始化 DeepLabV3+ 適配器 (Encoder: {encoder_name}, InChans: {self.in_chans}ch)")
        
        # 使用 smp 建立模型，in_channels 會根據 11/14/18 自動調整第一層卷積核
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name, 
            encoder_weights=pretrained_tag, 
            in_channels=self.in_chans, 
            classes=self.num_classes,
            encoder_output_stride=16 
        )
        
        self.model.to(self.device)
        
        # 若處於評估階段或需要接續訓練，則載入權重檔
        if model_path and Path(model_path).exists():
            print(f"   [Init] 載入 DeepLab 權重: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def _setup_logger(self, base_name, run_id):
        """ 配置實驗專屬訓練日誌紀錄系統 """
        log_dir = MADOS_ROOT / "logs" / base_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{base_name}_{run_id}_train.log"
        
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
        
        logging.info(f"🚀 DeepLab 訓練日誌已啟動，儲存至: {log_file}")

    def train(self, data_path, train_cfg, exp_cfg):
        """
        主訓練流程：
        實作了包含 VSCP 增強、餘弦退火排程、EMA 權重更新，
        以及最重要的「Top-1, Top-2, Top-3 權重自動競爭篩選」機制。
        """
        base_name = exp_cfg['base_name']
        run_id = exp_cfg['run_id']
        project = exp_cfg.get('results_base_dir', 'results-seg')
        
        # 1. 環境準備
        self._setup_logger(base_name, run_id)
        weight_dir = Path(project) / base_name / "weight" / run_id
        weight_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = Path(project) / base_name / run_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 讀取超參數與 3-Fold 設定
        epochs = train_cfg.get('epochs', 200)
        batch = train_cfg.get('batch_size', 16)
        patience = train_cfg.get('patience', 50)
        lr0 = train_cfg.get('lr0', 0.0001)
        weight_param = train_cfg.get('weight_param', 1.02)
        warmup_epochs = train_cfg.get('warmup_epochs', 10)
        fold_prefix = train_cfg.get('split_prefix', None)
        
        logging.info("\n" + "-"*50)
        logging.info(f"📝 DeepLab 訓練配置 (模式: {self.in_chans}ch):")
        logging.info(f"   - 核心指數模式: {'11+3 (FAI, FDI, SI)' if self.in_chans == 14 else 'Standard'}")
        logging.info(f"   - 儲存策略: top1.pth, top2.pth, top3.pth, last.pth")
        logging.info("-" * 50 + "\n")

        # 3. 初始化數據載入器 (將 3-Fold 分片前綴傳遞給 Dataset)
        train_loader = DataLoader(
            MADOS(data_path, 'train', 'seg', input_channels=self.in_chans, split_prefix=fold_prefix), 
            batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            MADOS(data_path, 'val', 'seg', input_channels=self.in_chans, split_prefix=fold_prefix), 
            batch_size=batch, shuffle=False, num_workers=0
        )

        # 4. 配置優化排程 (Step-wise 餘弦退火)
        n_iter_per_epoch = len(train_loader)
        lr_schedule = cosine_scheduler(lr0, 0, epochs, n_iter_per_epoch, warmup_epochs=warmup_epochs)
        ema_scheduler = cosine_scheduler(0.999, 0.999, epochs, n_iter_per_epoch)
        
        # 5. 配置損失函數與優化器
        weights = gen_weights(class_distr_binary, c=weight_param).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1) 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr0)
        
        # 指數移動平均 (EMA)，有助於提升模型泛化穩定性
        model_ema = ModelEma(self.model, decay=0.999, device=self.device) if train_cfg.get('model_ema', True) else None

        train_loss_history, val_loss_history, val_iou_history = [], [], []
        
        # 🚀 🚀 🚀 核心更新：Top-K 分數追蹤器與早停計數 🚀 🚀 🚀
        top_k_scores = {"top1": -1.0, "top2": -1.0, "top3": -1.0}
        patience_counter = 0
        
        # 6. 核心訓練迭代
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (DeepLab)")

            for it, (images, masks, _) in enumerate(pbar):
                global_it = n_iter_per_epoch * (epoch - 1) + it
                # 動態更新學習率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_schedule[global_it]

                # VSCP 空間拷貝貼上數據增強
                if train_cfg.get('vscp', True):
                    img_a, msk_a = VSCP(images.numpy(), masks.numpy())
                    images = torch.cat([images, torch.from_numpy(img_a)])
                    masks = torch.cat([masks, torch.from_numpy(msk_a)])

                images, masks = images.to(self.device), masks.to(self.device).long()
                optimizer.zero_grad(set_to_none=True)
                
                # 前向傳播與插值對齊 (確保 Logits 尺寸與 Masks 一致)
                outputs = self.model(images)
                if outputs.shape[-2:] != images.shape[-2:]:
                    outputs = F.interpolate(outputs, size=images.size()[2:4], mode='bilinear', align_corners=True)
                
                loss = criterion(outputs, masks)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.get('clip_grad', 1.0))
                optimizer.step()
                
                if model_ema:
                    model_ema.decay = ema_scheduler[global_it]
                    model_ema.update(self.model)
                
                running_train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})

            # 7. 驗證與指標紀錄
            metrics = self.validate(self.model, val_loader, criterion)
            avg_val_loss, current_iou = metrics['loss'], metrics['oil_iou']
            
            train_loss_history.append(running_train_loss / n_iter_per_epoch)
            val_loss_history.append(avg_val_loss)
            val_iou_history.append(current_iou)

            logging.info(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.4f} | Oil IoU: {current_iou:.4f} | Recall: {metrics['oil_recall']:.4f}")

            # 🚀 🚀 🚀 核心更新：Top-1, Top-2, Top-3 與 Last 儲存機制 🚀 🚀 🚀
            # 存儲當前最後一輪權重
            torch.save(self.model.state_dict(), weight_dir / "last.pth")

            # 權重排名更新邏輯：若分數進入前三名則觸發位移儲存
            if current_iou > top_k_scores["top1"]:
                # 晉升為 Top 1：將舊的 1 移至 2，2 移至 3
                logging.info(f" 🏆 DeepLab New Top 1! IoU: {current_iou:.4f} (舊最佳: {top_k_scores['top1']:.4f})")
                if (weight_dir / "top2.pth").exists(): shutil.copy2(weight_dir / "top2.pth", weight_dir / "top3.pth")
                if (weight_dir / "top1.pth").exists(): shutil.copy2(weight_dir / "top1.pth", weight_dir / "top2.pth")
                torch.save(self.model.state_dict(), weight_dir / "top1.pth")
                
                top_k_scores["top3"] = top_k_scores["top2"]
                top_k_scores["top2"] = top_k_scores["top1"]
                top_k_scores["top1"] = current_iou
                patience_counter = 0 # 重置早停
            
            elif current_iou > top_k_scores["top2"]:
                # 晉升為 Top 2：將舊的 2 移至 3
                logging.info(f" ⭐ DeepLab New Top 2! IoU: {current_iou:.4f}")
                if (weight_dir / "top2.pth").exists(): shutil.copy2(weight_dir / "top2.pth", weight_dir / "top3.pth")
                torch.save(self.model.state_dict(), weight_dir / "top2.pth")
                top_k_scores["top3"] = top_k_scores["top2"]
                top_k_scores["top2"] = current_iou
            
            elif current_iou > top_k_scores["top3"]:
                # 晉升為 Top 3
                logging.info(f" ✨ DeepLab New Top 3! IoU: {current_iou:.4f}")
                torch.save(self.model.state_dict(), weight_dir / "top3.pth")
                top_k_scores["top3"] = current_iou

            # 早停判定 (基於最優指標 Top 1)
            if current_iou <= top_k_scores["top1"]:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"🛑 觸發早停：連續 {patience} 輪指標未突破 Top 1。")
                    break

        # 訓練結束，產出趨勢圖
        self.plot_curves(train_loss_history, val_loss_history, val_iou_history, plot_dir, f"{base_name}_{run_id}")
        return {'best_model_path': str(weight_dir / "top1.pth")}

    def validate(self, model, loader, criterion):
        """ 評估模型在驗證集上的像素級統計性能 (IoU, Recall) """
        model.eval()
        total_loss, tp, fp, fn = 0, 0, 0, 0
        eps = 1e-10 
        with torch.no_grad():
            for images, masks, _ in loader:
                images, masks = images.to(self.device), masks.to(self.device).long()
                outputs = model(images)
                
                # 確保 Logits 尺寸與標籤對齊
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=True)
                
                total_loss += criterion(outputs, masks).item()
                preds = outputs.argmax(1)
                
                # 遮罩有效區域判定 (排除 -1 ignore_index)
                mask_valid = (masks != -1)
                tp += ((preds == 1) & (masks == 1) & mask_valid).sum().item()
                fp += ((preds == 1) & (masks == 0) & mask_valid).sum().item()
                fn += ((preds == 0) & (masks == 1) & mask_valid).sum().item()
        
        return {
            'loss': total_loss / len(loader),
            'oil_iou': tp / (tp + fp + fn + eps),
            'oil_recall': tp / (tp + fn + eps),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)
        }

    def plot_curves(self, t_loss, v_loss, v_iou, save_path, full_name):
        """ 繪製訓練損失與驗證 IoU 之歷史趨勢圖 """
        epochs_range = range(1, len(t_loss) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, t_loss, label='Train Loss')
        plt.plot(epochs_range, v_loss, label='Val Loss')
        plt.title(f'DeepLab Loss History ({full_name})'); plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, v_iou, color='green', label='Oil IoU (Val)')
        plt.title(f'DeepLab Oil IoU ({full_name})'); plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path / f"{full_name}_loss_curve.png")
        plt.close()
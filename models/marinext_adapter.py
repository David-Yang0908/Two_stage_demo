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

# -------------------------------------------------------------------------
# 1. 路徑與環境配置
# -------------------------------------------------------------------------
MADOS_ROOT = Path(os.getcwd()) 
if str(MADOS_ROOT / 'marinext') not in sys.path:
    sys.path.append(str(MADOS_ROOT / 'marinext'))

# 導入 VSCP 數據增強、餘弦退火排程器
from utils.vscp import VSCP
from utils.assets import cosine_scheduler
# 導入 MADOS 數據集、權重計算工具
from utils.dataset import MADOS, gen_weights, class_distr_binary

class MarinextAdapter(nn.Module):
    """
    MariNext 模型適配器：封裝模型初始化、訓練、驗證與 Top-K 儲存邏輯。
    🚀 核心更新：支援 11/14/18 通道切換與 Top1-3 權重自動篩選。
    """
    def __init__(self, config, model_path=None):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 🚀 支援從 YAML 動態設定通道 (11/14/18)
        exp_cfg = config.get('experiment', {})
        self.in_chans = exp_cfg.get('input_channels', 11)
        self.num_classes = 2 
        
        model_name = exp_cfg.get('model_name', 'MariNext')
        
        if model_name == 'MariNext':
            from marinext.marinext_wrapper import MariNext
            self.model = MariNext(self.in_chans, self.num_classes)
        else:
            raise ValueError(f"❌ 不支援的模型名稱: {model_name}")

        self.model.to(self.device)
        
        if model_path and Path(model_path).exists():
            print(f"   [Init] 載入模型權重: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def _setup_logger(self, base_name, run_id):
        log_dir = MADOS_ROOT / "logs" / base_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{base_name}_{run_id}_train.log"
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        logging.info(f"🚀 實驗訓練日誌已啟動，儲存於: {log_file}")

    def train(self, data_path, train_cfg, exp_cfg):
        """
        主訓練程序：包含 Top-1, Top-2, Top-3 與 Last 權重儲存策略。
        """
        base_name = exp_cfg['base_name']
        run_id = exp_cfg['run_id']
        project = exp_cfg.get('results_base_dir', 'results-seg')
        
        self._setup_logger(base_name, run_id)
        
        weight_dir = Path(project) / base_name / "weight" / run_id
        weight_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = Path(project) / base_name / run_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = train_cfg.get('epochs', 200)
        batch = train_cfg.get('batch_size', 16)
        patience = train_cfg.get('patience', 50)
        lr0 = train_cfg.get('lr0', 0.0001)
        weight_param = train_cfg.get('weight_param', 1.02)
        warmup_epochs = train_cfg.get('warmup_epochs', 10)
        fold_prefix = train_cfg.get('split_prefix', None)

        logging.info("\n" + "-"*50)
        logging.info(f"📝 訓練配置彙整 (模式: {self.in_chans}ch):")
        logging.info(f"   - 核心指數模式: {'11+3 (FAI, FDI, SI)' if self.in_chans == 14 else 'Standard'}")
        logging.info(f"   - 儲存策略: top1.pth, top2.pth, top3.pth, last.pth")
        logging.info("-" * 50 + "\n")

        train_loader = DataLoader(
            MADOS(data_path, 'train', 'seg', input_channels=self.in_chans, split_prefix=fold_prefix), 
            batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            MADOS(data_path, 'val', 'seg', input_channels=self.in_chans, split_prefix=fold_prefix), 
            batch_size=batch, shuffle=False, num_workers=0
        )

        n_iter_per_epoch = len(train_loader)
        lr_schedule = cosine_scheduler(lr0, 0, epochs, n_iter_per_epoch, warmup_epochs=warmup_epochs)
        ema_scheduler = cosine_scheduler(0.999, 0.999, epochs, n_iter_per_epoch)
        
        weights = gen_weights(class_distr_binary, c=weight_param).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1) 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr0)
        
        model_ema = ModelEma(self.model, decay=0.999, device=self.device) if train_cfg.get('model_ema', True) else None

        train_loss_history, val_loss_history, val_iou_history = [], [], []
        
        # 🚀 🚀 🚀 核心更新：Top-K 分數追蹤器 🚀 🚀 🚀
        top_k_scores = {"top1": -1.0, "top2": -1.0, "top3": -1.0}
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

            for it, (images, masks, _) in enumerate(pbar):
                global_it = n_iter_per_epoch * (epoch - 1) + it
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_schedule[global_it]

                if train_cfg.get('vscp', True):
                    img_a, msk_a = VSCP(images.numpy(), masks.numpy())
                    images = torch.cat([images, torch.from_numpy(img_a)])
                    masks = torch.cat([masks, torch.from_numpy(msk_a)])

                images, masks = images.to(self.device), masks.to(self.device).long()
                optimizer.zero_grad(set_to_none=True)
                
                outputs = self.model(images)
                logits = F.interpolate(outputs, size=images.size()[2:4], mode='bilinear', align_corners=True)
                loss = criterion(logits, masks)
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.get('clip_grad', 1.0))
                optimizer.step()
                
                if model_ema:
                    model_ema.decay = ema_scheduler[global_it]
                    model_ema.update(self.model)
                
                running_train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})

            # 驗證指標
            metrics = self.validate(self.model, val_loader, criterion)
            avg_val_loss, current_iou = metrics['loss'], metrics['oil_iou']
            
            train_loss_history.append(running_train_loss / n_iter_per_epoch)
            val_loss_history.append(avg_val_loss)
            val_iou_history.append(current_iou)

            logging.info(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.4f} | Oil IoU: {current_iou:.4f}")

            # 🚀 🚀 🚀 核心更新：Top-1, Top-2, Top-3 與 Last 儲存邏輯 🚀 🚀 🚀
            # 1. 儲存 Last 權重 (每一輪都更新)
            torch.save(self.model.state_dict(), weight_dir / "last.pth")

            # 2. 判斷是否進入前三名 (由高到低排名)
            if current_iou > top_k_scores["top1"]:
                # 晉升為 Top1：舊的 1 移到 2, 2 移到 3
                logging.info(f" 🏆 New Top 1! IoU: {current_iou:.4f} (Previous Best: {top_k_scores['top1']:.4f})")
                if (weight_dir / "top2.pth").exists(): shutil.copy2(weight_dir / "top2.pth", weight_dir / "top3.pth")
                if (weight_dir / "top1.pth").exists(): shutil.copy2(weight_dir / "top1.pth", weight_dir / "top2.pth")
                torch.save(self.model.state_dict(), weight_dir / "top1.pth")
                
                # 更新分數紀錄
                top_k_scores["top3"] = top_k_scores["top2"]
                top_k_scores["top2"] = top_k_scores["top1"]
                top_k_scores["top1"] = current_iou
                patience_counter = 0 # 重置早停
            
            elif current_iou > top_k_scores["top2"]:
                # 晉升為 Top2：舊的 2 移到 3
                logging.info(f" ⭐ New Top 2! IoU: {current_iou:.4f}")
                if (weight_dir / "top2.pth").exists(): shutil.copy2(weight_dir / "top2.pth", weight_dir / "top3.pth")
                torch.save(self.model.state_dict(), weight_dir / "top2.pth")
                top_k_scores["top3"] = top_k_scores["top2"]
                top_k_scores["top2"] = current_iou
            
            elif current_iou > top_k_scores["top3"]:
                # 晉升為 Top3
                logging.info(f" ✨ New Top 3! IoU: {current_iou:.4f}")
                torch.save(self.model.state_dict(), weight_dir / "top3.pth")
                top_k_scores["top3"] = current_iou

            # 早停判定 (基於 Top1 是否有更新)
            if current_iou <= top_k_scores["top1"]:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"🛑 早停：連續 {patience} 輪未突破 Top 1 指標。")
                    break

        self.plot_curves(train_loss_history, val_loss_history, val_iou_history, plot_dir, f"{base_name}_{run_id}")
        return {'best_model_path': str(weight_dir / "top1.pth")}

    def validate(self, model, loader, criterion):
        model.eval()
        total_loss, tp, fp, fn = 0, 0, 0, 0
        eps = 1e-10
        with torch.no_grad():
            for images, masks, _ in loader:
                images, masks = images.to(self.device), masks.to(self.device).long()
                outputs = model(images)
                logits = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=True)
                total_loss += criterion(logits, masks).item()
                preds = logits.argmax(1)
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
        epochs_range = range(1, len(t_loss) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, t_loss, label='Train')
        plt.plot(epochs_range, v_loss, label='Val')
        plt.title('Loss History'); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, v_iou, color='green', label='Oil IoU')
        plt.title('Oil IoU History'); plt.legend()
        plt.tight_layout()
        plt.savefig(save_path / f"{full_name}_loss_curve.png")
        plt.close()
# -*- coding: utf-8 -*-
import os
import sys
import torch
import logging
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from torch.utils.data import DataLoader
from timm.utils import ModelEma

MADOS_ROOT = Path(os.getcwd()) 
if str(MADOS_ROOT / 'marinext') not in sys.path:
    sys.path.append(str(MADOS_ROOT / 'marinext'))

from utils.vscp import VSCP
from utils.dataset import MADOS, gen_weights, class_distr_binary
# 🚀 引入 Evaluation 工具以進行更科學的權重儲存判定
from utils.metrics import Evaluation

class YOLOv11sAdapter(nn.Module):
    """
    YOLOv11s 分類器適配器：預設使用 Small 版本模型，
    並將 Best 權重儲存邏輯改為以 Macro F1 為指標。
    """
    def __init__(self, config, model_path=None):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        exp_cfg = config.get('experiment', {})
        self.in_chans = exp_cfg.get('input_channels', 18)
        self.num_classes = exp_cfg.get('num_classes', 2)
        
        # 🚀 修正：改為預設加載 yolo11s-cls (Small 版本)
        model_variant = exp_cfg.get('model_variant', 'yolo11s-cls.yaml')
        logging.info(f"🎨 初始化 YOLOv11s 分類器: {model_variant} ({self.in_chans}ch)")
        
        self.model = YOLO(model_variant).model
        
        # 🚀 核心修改：第一層卷積支持多通道
        old_conv = self.model.model[0].conv
        self.model.model[0].conv = nn.Conv2d(
            self.in_chans, old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
            padding=old_conv.padding, bias=False
        )
        
        inner_model = self.model.model[-1]
        if hasattr(inner_model, 'linear'):
            in_features = inner_model.linear.in_features
            inner_model.linear = nn.Linear(in_features, self.num_classes)

        self.model.to(self.device)

        if model_path and Path(model_path).exists():
            logging.info(f"   [Init] 載入權重: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def _setup_logger(self, base_name, run_id):
        log_dir = MADOS_ROOT / "logs" / base_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{base_name}_{run_id}_cls_train.log"
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

    def train(self, data_path, train_cfg, exp_cfg):
        """ 執行分類訓練迴圈 """
        # 🚀 🚀 🚀 核心修正：使用 .get() 防止 KeyError，並提供安全預設值
        base_name = exp_cfg.get('base_name', 'YOLOv11s_Exp')
        run_id = exp_cfg.get('run_id', 'RunX')
        project = exp_cfg.get('results_base_dir', 'results-cls')
        
        self._setup_logger(base_name, run_id)
        
        weight_dir = Path(project) / base_name / "weight" / run_id
        weight_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = Path(project) / base_name / run_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = train_cfg.get('epochs', 100)
        imgsz = train_cfg.get('imgsz', 240)
        batch = train_cfg.get('batch_size', 5)
        lr0 = train_cfg.get('lr0', 0.0001)
        lr_milestones = train_cfg.get('lr_steps', [30, 60, 90])
        patience = train_cfg.get('patience', 35)
        weight_param = train_cfg.get('weight_param', 1.0)

        logging.info(f"🚀 YOLOv11s 訓練啟動: {base_name}_{run_id} | Batch: {batch} | Weight: {weight_param}")

        train_loss_history, val_loss_history, val_f1_history = [], [], []
        
        if weight_param != 1.0:
            weights = gen_weights(class_distr_binary, c=weight_param).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
        model_ema = ModelEma(self.model, decay=0.999, device=self.device) if train_cfg.get('model_ema', True) else None
        
        train_loader = DataLoader(MADOS(data_path, 'train', task='cls', input_channels=self.in_chans), batch_size=batch, shuffle=True)
        val_loader = DataLoader(MADOS(data_path, 'val', task='cls', input_channels=self.in_chans), batch_size=batch, shuffle=False)

        # 🚀 核心優化：改以 Macro F1 作為判定
        best_f1 = -1.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

            for images, labels, _ in pbar:
                # 🚀 🚀 🚀 核心修正：加入 VSCP 守門員邏輯，避免小 Batch (如 Batch 8) 因無油汙樣本導致 np.stack 崩潰
                if train_cfg.get('vscp', False) and (labels == 1).any():
                    labels_np = labels.numpy()
                    labels_spatial = np.full((labels_np.shape[0], imgsz, imgsz), -1, dtype=np.int8)
                    for b in range(labels_np.shape[0]):
                        labels_spatial[b, :, :] = labels_np[b]
                    
                    try:
                        img_a, msk_a = VSCP(images.numpy(), labels_spatial)
                        # 安全檢查：確保 VSCP 成功產出增強數據
                        if img_a is not None and len(img_a) > 0:
                            images = torch.cat([images, torch.from_numpy(img_a)])
                            labels = torch.cat([labels, torch.from_numpy(msk_a[:, 0, 0])])
                    except Exception as e:
                        logging.warning(f"⚠️ VSCP 執行跳過: {e}")

                images, labels = images.to(self.device), labels.to(self.device).long()
                optimizer.zero_grad()
                
                outputs = self.model(images)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                if model_ema: model_ema.update(self.model)
                running_loss += loss.item()

            # 🚀 取得驗證指標 (包含 F1, Accuracy)
            metrics = self.validate(self.model, val_loader, criterion)
            current_f1 = metrics['f1']
            scheduler.step()
            
            train_loss_history.append(running_loss / len(train_loader))
            val_loss_history.append(metrics['loss'])
            val_f1_history.append(current_f1)

            logging.info(f"[Epoch {epoch}] Loss: {train_loss_history[-1]:.4f} | Val F1: {current_f1:.4f} | Acc: {metrics['acc']:.4f}")
            torch.save(self.model.state_dict(), weight_dir / "last_cls.pth")

            # 🚀 修正儲存邏輯：以 Macro F1 判定最佳權重，解決「背景優先」問題
            if current_f1 > best_f1:
                best_f1 = current_f1
                torch.save(self.model.state_dict(), weight_dir / "best_cls.pth")
                patience_counter = 0
                logging.info(f" ⭐ F1 指標提升: {best_f1:.4f} (權重已更新)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info("🛑 觸發早停機制。")
                    break

        self.plot_curves(train_loss_history, val_loss_history, val_f1_history, plot_dir, f"{base_name}_{run_id}")
        return {'best_model_path': str(weight_dir / "best_cls.pth")}

    def validate(self, model, loader, criterion):
        model.eval()
        total_loss, all_preds, all_labels = 0, [], []
        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(self.device), labels.to(self.device).long()
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                    
                total_loss += criterion(outputs, labels).item()
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 🚀 使用 Evaluation 工具計算學術指標
        eval_info = Evaluation(np.array(all_preds), np.array(all_labels))
        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        
        return {
            'loss': total_loss / len(loader), 
            'acc': acc, 
            'f1': eval_info['macroF1'] # 🚀 輸出 Macro F1
        }

    def plot_curves(self, t_loss, v_loss, v_f1, save_path, full_name):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.plot(t_loss, label='Train'); plt.plot(v_loss, label='Val'); plt.title('Loss'); plt.legend()
        plt.subplot(1, 2, 2); plt.plot(v_f1); plt.title('Macro F1 Score of Validation Set')
        plt.savefig(save_path / f"{full_name}_cls_curves.png"); plt.close()
# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    適用於二元分割或多類別分割的 Dice Loss
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, masks):
        # 1. 將輸出轉為機率 (Softmax)
        probs = F.softmax(logits, dim=1)
        
        # 2. 針對「油汙」類別 (index 1) 計算
        # 如果是二元分類，通常只看類別 1
        probs_oil = probs[:, 1, :, :]
        masks_oil = masks.float() # 假設 mask 1 是油汙, 0 是背景

        # 3. 拉平以便計算
        probs_oil = probs_oil.contiguous().view(-1)
        masks_oil = masks_oil.contiguous().view(-1)

        # 4. 計算 Dice 係數
        intersection = (probs_oil * masks_oil).sum()
        dice_coeff = (2. * intersection + self.smooth) / (probs_oil.sum() + masks_oil.sum() + self.smooth)
        
        return 1 - dice_coeff
    
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, masks):
        # 1. 取得機率並過濾掉 ignore_index (-1)
        probs = F.softmax(logits, dim=1)
        probs_oil = probs[:, 1, :, :]
        
        # 🚀 重要修正：只計算非 -1 的區域
        mask_valid = (masks != -1)
        if not mask_valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        p = probs_oil[mask_valid]
        target = masks[mask_valid].float()

        # 2. 計算 TP, FP, FN
        tp = (p * target).sum()
        fp = (p * (1 - target)).sum()
        fn = ((1 - p) * target).sum()

        # 3. 計算 Tversky 指數
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky_index
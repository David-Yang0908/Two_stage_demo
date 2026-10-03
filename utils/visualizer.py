# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

class SegmentationVisualizer:
    """
    分割結果視覺化工具：負責雙重權重排名、上色疊加、拼圖與數據標註。
    """
    def __init__(self, save_base_dir, base_name, run_id): # 🚀 修正：新增 base_name 以對齊 evaluation 呼叫介面
        # 建立視覺化結果儲存路徑：results-seg/[實驗名稱]/[輪次]/visuals/
        self.save_dir = Path(save_base_dir) / base_name / run_id / "visuals"
        self.colors = {
            0: (230, 216, 173), # Label 0 (Others): 淺藍色 (BGR: 173, 216, 230)
            1: (0, 0, 255),     # Label 1 (Oil): 紅色 (BGR: 0, 0, 255)
        }
        self.alpha = 0.4 # 透明度：0.4 顏色, 0.6 原圖

    def _get_rgb_image(self, image_tensor):
        """
        將 11ch/18ch Tensor 轉換為可視化的 RGB 背景 (取 Band 4, 3, 2)。
        """
        img = image_tensor.cpu().numpy()
        # 提取 RGB 通道 (假設順序為 Band 2,3,4 -> 索引 1,2,3)
        rgb = np.stack([img[3], img[2], img[1]], axis=-1)
        
        # 線性拉伸轉回 0-255 以便 OpenCV 處理
        rgb_min, rgb_max = rgb.min(), rgb.max()
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8) * 255
        return rgb.astype(np.uint8)

    def _overlay_mask(self, background_rgb, mask):
        """
        將 Mask 疊加半透明顏色到 RGB 底圖上。
        """
        overlay = background_rgb.copy()
        for label, color in self.colors.items():
            overlay[mask == label] = color
        
        # 影像混合：Dst = Src1 * alpha + Src2 * (1 - alpha)
        return cv2.addWeighted(overlay, self.alpha, background_rgb, 1 - self.alpha, 0)

    def create_comparison_image(self, image_tensor, gt_mask, pred_mask, stats, roi_name, save_path):
        """
        建立大圖：包含標題、左右對比圖以及底部的數據統計。
        """
        # 1. 準備背景與疊加圖
        bg_rgb = self._get_rgb_image(image_tensor)
        gt_overlay = self._overlay_mask(bg_rgb, gt_mask)
        pred_overlay = self._overlay_mask(bg_rgb, pred_mask)

        # 2. 拼接左右圖 (GT | Pred)
        combined_img = np.hstack([gt_overlay, pred_overlay])
        h, w, _ = combined_img.shape

        # 3. 建立 PIL 畫布增加文字資訊
        canvas_h = h + 150 # 增加上下與底部空間
        full_canvas = Image.new('RGB', (w, canvas_h), (255, 255, 255))
        
        # 貼上影像
        full_canvas.paste(Image.fromarray(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)), (0, 70))
        
        draw = ImageDraw.Draw(full_canvas)
        try:
            # 嘗試載入系統字體，若失敗則使用預設
            font_title = ImageFont.truetype("arial.ttf", 28)
            font_stats = ImageFont.truetype("arial.ttf", 22)
        except:
            font_title = font_stats = ImageFont.load_default()

        # 🚀 上方文字：ROI 名稱
        draw.text((20, 15), f"Target ROI: {roi_name}", fill=(0, 0, 0), font=font_title)
        
        # 🚀 左右圖片標籤
        draw.text((w//4 - 100, 45), "Ground Truth (Others:Blue / Oil:Red)", fill=(100, 100, 100), font=font_stats)
        draw.text((3*w//4 - 100, 45), "Prediction (Others:Blue / Oil:Red)", fill=(100, 100, 100), font=font_stats)

        # 🚀 底部統計數據：加入 TP, FP, FN, TN 以及對應的 Recall 指標
        stats_str = f"Pixel Stats -> TP: {stats['tp']} | FP: {stats['fp']} | FN: {stats['fn']} | TN: {stats['tn']} | Recall: {stats['recall']:.2%}"
        draw.text((40, h + 90), stats_str, fill=(0, 0, 0), font=font_stats)

        # 儲存
        full_canvas.save(save_path)

    def save_top_k_visuals(self, metadata_list, target_tag):
        """
        🚀 雙重排序邏輯：
        優先根據百分比(Rate)排序，若百分比相同，則以該資料夾分析的實際像素數量(Count)多者排前面。
        """
        # 定義四個分類的排序函數：(主要排序: 比例, 次要排序: 絕對像素量)
        categories = {
            "TP": lambda x: (x['recall'], x['stats']['tp']),      # Recall 高且 TP 面積大
            "FP": lambda x: (x['fp_rate'], x['stats']['fp']),     # 誤判率高且誤判面積大
            "FN": lambda x: (x.get('fn_rate', 1.0 - x['recall']), x['stats']['fn']), # 漏抓率高且漏抓面積大
            "TN": lambda x: (x['tn_ratio'], x['stats']['tn']),    # 背景準確率高且背景面積大
        }

        for cat_name, sort_func in categories.items():
            cat_dir = self.save_dir / target_tag / cat_name
            cat_dir.mkdir(parents=True, exist_ok=True)
            
            # 🚀 執行排序：reverse=True 代表由大到小 (百分比大者優先，若同則數量大者優先)
            sorted_list = sorted(metadata_list, key=sort_func, reverse=True)[:10]

            for i, item in enumerate(sorted_list):
                # 命名規則：[排名]_[ROI名稱].png
                save_name = f"{i+1:02d}_{item['roi_name']}.png"
                self.create_comparison_image(
                    item['image'], item['gt'], item['pred'], 
                    item['stats'], item['roi_name'], cat_dir / save_name
                )
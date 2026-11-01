from mcp.server.mcpserver import MCPServer
import httpx

# 在 MCP 2.x 中改用 MCPServer
mcp = MCPServer("MarineEye_MCP_Server")
BASE_URL = "http://127.0.0.1:8000" 

@mcp.tool()
def list_available_rois() -> str:
    """獲取可供分析的衛星影像 ROI (Region of Interest) 列表。"""
    try:
        response = httpx.get(f"{BASE_URL}/api/get_test_list", timeout=10.0)
        response.raise_for_status()
        files = response.json().get("files", [])
        rois = [f"{f['name']} (Has Oil: {f['has_oil']})" for f in files]
        return "可用的衛星影像列表:\n" + "\n".join(rois[:20]) + "\n...(僅顯示前20筆)"
    except Exception as e:
        return f"讀取列表失敗: {str(e)}"

@mcp.tool()
def evaluate_random_20_images() -> str:
    """隨機抽取 20 張衛星影像，透過後端自動執行雙階段推論，並列出 Stage 1 的分類清單與 Stage 2 的效能指標。"""
    try:
        res = httpx.get(f"{BASE_URL}/api/get_random_selection", timeout=10.0)
        res.raise_for_status()
        rois = res.json().get("selected", [])
        if not rois:
            return "無法取得隨機影像名單。"

        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
        candidates = []
        excluded = []
        
        for roi in rois:
            eval_res = httpx.post(f"{BASE_URL}/api/evaluate/{roi}?stage=hybrid&model=EB2", timeout=30.0)
            eval_res.raise_for_status()
            data = eval_res.json()
            
            # 記錄 Stage 1 的分類結果
            if data.get("is_candidate"):
                candidates.append(roi)
            else:
                excluded.append(roi)

            # 統計 Stage 2 的像素
            metrics = data.get("metrics", {})
            total_tp += metrics.get("tp", 0)
            total_fp += metrics.get("fp", 0)
            total_tn += metrics.get("tn", 0)
            total_fn += metrics.get("fn", 0)
            
        total_oil = total_tp + total_fn
        iou = (total_tp / (total_tp + total_fp + total_fn) * 100) if (total_tp + total_fp + total_fn) > 0 else 0
        recall = (total_tp / total_oil * 100) if total_oil > 0 else 0
        
        report = f"✅ 已成功在背景完成 {len(rois)} 張影像的雙階段評估。\n\n"
        report += f"📁 [Stage 1 分類結果]\n"
        report += f"🛑 被 Stage 1 成功攔截 (Excluded) 共 {len(excluded)} 張:\n{', '.join(excluded)}\n\n"
        report += f"🔹 進入 Stage 2 切割 (Candidates) 共 {len(candidates)} 張:\n{', '.join(candidates)}\n\n"
        
        report += f"📊 [Stage 2 統整指標]\n"
        report += f"總計像素 -> TP:{total_tp}, FP:{total_fp}, TN:{total_tn}, FN:{total_fn}\n"
        report += f"Overall Oil Pixel IoU: {iou:.2f}%, Overall Recall: {recall:.2f}%"
        
        return report
    except Exception as e:
        return f"批次執行失敗: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport='stdio')
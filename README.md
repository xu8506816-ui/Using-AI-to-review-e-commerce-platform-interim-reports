# 電商違規審核 Demo（刀具／槍械）

以 Streamlit 建立的簡易電商商品違規審核工具，整合文字風險分析（關鍵字 + BERT 分類）、圖片偵測（YOLOv8 ONNX）、以及圖片文字擷取（EasyOCR）。支援三種檢測模式：上傳檔案、單頁網址、深度網址掃描。

## 功能特色
- 文字風險分析：關鍵字 + BERT 模型混合判斷
- 圖片風險分析：YOLOv8（ONNX）偵測刀具／槍械
- 圖片 OCR：EasyOCR（預設繁中/英文）
- 網址檢測：抓取頁面文字與圖片後分析
- 深度檢測：同網域多頁面掃描（可設定深度與排除路徑）

## 專案結構
- `app.py`：主程式（Streamlit UI + 檢測邏輯）
- `requirements.txt`：套件需求
- `weights/`：YOLOv8 權重放置目錄
- `weapons_detection.json`：專案資料（如需可自行參考）

## 安裝需求
1) 建立環境並安裝套件

```bash
pip install -r requirements.txt
```

2) 下載 YOLOv8 權重（必要）
- 來源：Weapons-and-Knives-Detector-with-YOLOv8
- 檔名：`best.onnx`
- 放置位置：`weights/weapons-knives-best.onnx`

若不放在預設路徑，可在側邊欄手動輸入權重路徑。

3) 文字模型（可選）
- 預設使用 `bert-base-chinese`。
- 若環境無網路，請先下載模型並在側邊欄輸入本地路徑。
- 若模型無法載入，系統會自動退回關鍵字檢測。

## 使用方式
```bash
streamlit run app.py
```

### 檢測模式
- 上傳檔案：輸入商品標題／描述 + 圖片上傳
- 網址檢測：輸入單一商品頁網址
- 深度網址檢測：輸入起始網址，設定排除路徑、最大深度與頁數

## 主要設定（側邊欄）
- YOLO 權重路徑：預設 `weights/weapons-knives-best.onnx`
- 文字分類模型：Hugging Face 名稱或本地路徑
- OCR 語言：逗號分隔（如 `ch_tra,en`）

## 注意事項
- 網址抓取以 `requests + BeautifulSoup` 擷取文字與圖片，部分網站可能會有防爬或動態載入限制。
- EasyOCR 預設不開 GPU；如需加速可自行調整程式。
- 模型輸出標籤假設包含 `positive/negative` 或 `violation/ok`，如自訓模型請確認 label。

## 常見問題
- 無法偵測圖片：請確認 `weights/weapons-knives-best.onnx` 路徑正確。
- 文字模型載入失敗：改用本地模型路徑或檢查網路。
- OCR 無結果：可調整圖片品質或 OCR 語言設定。

## 參考
- https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8

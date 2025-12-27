from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageOps
import easyocr
from io import BytesIO
from urllib.parse import urljoin, urlparse
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from ultralytics import YOLO

# ======== 常見武器關鍵字（可自行增刪） ========
KNIFE_KEYWORDS = [
    "刀", "小刀", "尖刀", "軍刀", "蝴蝶刀", "折疊刀", "匕首",
    "獵刀", "登山刀", "菜刀", "水管刀", "工兵刀", "砍刀",
    "開山刀", "剃刀", "壓刀",
]

GUN_KEYWORDS = [
    "槍", "手槍", "步槍", "獵槍", "散彈槍", "突擊槍",
    "BB槍", "bb槍", "玩具槍", "模型槍", "仿真槍", "空氣槍",
    "衝鋒槍", "狙擊槍", "水彈槍", "水槍",
]

EN_KNIFE_KEYWORDS = [
    "knife",
    "knives",
    "dagger",
    "blade",
    "machete",
    "folding knife",
    "pocket knife",
    "hunting knife",
]

EN_GUN_KEYWORDS = [
    "gun",
    "pistol",
    "rifle",
    "sniper",
    "shotgun",
    "airsoft",
    "bb gun",
    "toy gun",
    "machine gun",
]

YOLO_DEFAULT_WEIGHTS = "weights/weapons-knives-best.onnx"
YOLO_CONF_THRESHOLD = 0.25
WEAPON_LABELS = {
    # JoaoAssalim 模型只有兩類
    "knife",
    "gun",
    # 允許大小寫／複數別名
    "knives",
    "guns",
}

DEFAULT_TEXT_MODEL = "bert-base-chinese"
DEFAULT_TEXT_MODEL_THRESHOLD = 0.5
DEFAULT_OCR_LANGS = ["ch_tra", "en"]


# ======== 文字檢查邏輯（BERT + 關鍵字混合） ========
@st.cache_resource(show_spinner=False)
def load_text_pipeline(model_path: str):
    """
    載入 transformers pipeline。
    - 預設使用 bert-base-chinese；若無網路請先下載模型並用 UI 填寫本地路徑。
    - 若載入失敗會在 analyze_text 內回退到關鍵字檢查。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)


def keyword_hits(text: str) -> Dict[str, List[str]]:
    text_lower = text.lower()
    hit_knives: List[str] = []
    hit_guns: List[str] = []

    for kw in KNIFE_KEYWORDS:
        if kw in text:
            hit_knives.append(kw)

    for kw in GUN_KEYWORDS:
        if kw in text:
            hit_guns.append(kw)

    for kw in EN_KNIFE_KEYWORDS:
        if kw in text_lower:
            hit_knives.append(kw)

    for kw in EN_GUN_KEYWORDS:
        if kw in text_lower:
            hit_guns.append(kw)

    return {"hit_knives": hit_knives, "hit_guns": hit_guns}


def analyze_text(text: str, model_path: str) -> Dict:
    """
    先嘗試 BERT 分類；若失敗則回退到關鍵字檢測。
    需自行準備二分類模型（違規/正常），並以 model_path 指向。
    模型輸出 label 包含 'positive'/'negative' 或 'violation'/'ok' 皆可。
    """
    hits = keyword_hits(text)
    keyword_score = 0.0
    if hits["hit_knives"] or hits["hit_guns"]:
        keyword_score = min(1.0, 0.6 + 0.1 * (len(hits["hit_knives"]) + len(hits["hit_guns"])))

    try:
        clf = load_text_pipeline(model_path)
        preds = clf(text)
        # pipeline(top_k=None) 會回傳 list of list
        first = preds[0][0] if preds and isinstance(preds[0], list) else preds[0]
        label = first["label"].lower()
        score = first["score"]
        # 假設 label 包含 "viol" 或 "pos" 表示高風險；可依實際模型標籤調整
        risk_from_model = score if ("viol" in label or "pos" in label) else (1 - score)
        final_score = max(keyword_score, risk_from_model)
        return {
            "score": final_score,
            "hit_knives": hits["hit_knives"],
            "hit_guns": hits["hit_guns"],
            "model_label": label,
            "model_score": score,
            "source": "bert",
        }
    except Exception as exc:
        return {
            "score": keyword_score,
            "hit_knives": hits["hit_knives"],
            "hit_guns": hits["hit_guns"],
            "model_label": None,
            "model_score": None,
            "source": f"keyword_fallback ({exc})",
        }


# ======== 網頁文字擷取 ========
def fetch_url_text(url: str, timeout: int = 8) -> Dict[str, Optional[str]]:
    """
    抓取網頁文字做檢測。僅擷取 <body> 文字，移除 script/style。
    會回傳 text、images（網址列表）與 debug 訊息。
    """
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        # 限制長度，避免過長
        max_len = 8000
        if len(text) > max_len:
            text = text[:max_len]
        # 收集圖片連結
        imgs: List[str] = []
        for img_tag in soup.find_all("img"):
            src = img_tag.get("src") or img_tag.get("data-src") or ""
            if not src and img_tag.get("srcset"):
                src = img_tag.get("srcset").split(",")[0].split()[0]
            if not src:
                continue
            full = urljoin(resp.url, src)
            imgs.append(full)
        return {"text": text, "images": imgs, "debug": f"成功抓取文字，長度 {len(text)}，圖片 {len(imgs)} 張"}
    except Exception as exc:
        return {"text": "", "images": [], "debug": f"無法抓取網址內容: {exc}"}


def fetch_url_page(url: str, timeout: int = 30) -> Dict[str, Optional[str]]:
    """
    抓取頁面文字、圖片與同網域連結。
    """
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if len(text) > 8000:
            text = text[:8000]

        imgs: List[str] = []
        for img_tag in soup.find_all("img"):
            src = img_tag.get("src") or img_tag.get("data-src") or ""
            if not src and img_tag.get("srcset"):
                src = img_tag.get("srcset").split(",")[0].split()[0]
            if not src:
                continue
            imgs.append(urljoin(resp.url, src))

        links: List[str] = []
        for a in soup.find_all("a"):
            href = a.get("href")
            if not href:
                continue
            links.append(urljoin(resp.url, href))

        return {
            "text": text,
            "images": imgs,
            "links": links,
            "debug": f"成功抓取頁面：文字 {len(text)}，圖片 {len(imgs)}，連結 {len(links)}",
        }
    except Exception as exc:
        return {"text": "", "images": [], "links": [], "debug": f"抓取失敗: {exc}"}


def is_same_domain(base: str, target: str) -> bool:
    return urlparse(base).netloc == urlparse(target).netloc


def is_excluded_path(url: str, exclude_paths: List[str]) -> bool:
    path = urlparse(url).path.lower()
    return any(path.startswith(p) for p in exclude_paths)


def download_images(urls: List[str], limit: int = 5) -> Tuple[List[Image.Image], List[str]]:
    images: List[Image.Image] = []
    debug_msgs: List[str] = []
    max_bytes = 5 * 1024 * 1024
    for url in urls[:limit]:
        try:
            r = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if "image" not in ctype:
                debug_msgs.append(f"SKIP非圖片: {url}")
                continue
            raw = r.content
            if len(raw) > max_bytes:
                debug_msgs.append(f"SKIP過大: {url}")
                continue
            img = Image.open(BytesIO(raw)).convert("RGB")
            images.append(img)
            debug_msgs.append(f"OK: {url}")
        except Exception as exc:
            debug_msgs.append(f"FAIL: {url} ({exc})")
    return images, debug_msgs


# ======== 影像檢查邏輯（YOLOv8）========
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights: str) -> YOLO:
    return YOLO(weights)


def analyze_image(img: Image.Image, weights_path: str) -> Dict:
    """
    - 權重路徑由 UI 設定（預設 weights/weapons-knives-best.onnx）
    - 若未提供模型檔，請至 https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8
      下載 best.onnx，並放置於 weights/weapons-knives-best.onnx 或在側邊設定中指定自訂路徑
    """
    if img is None:
        return {"score": 0.0, "labels": [], "debug": "尚未上傳圖片"}

    try:
        model = load_yolo_model(weights_path)
    except Exception as exc:
        return {
            "score": 0.0,
            "labels": [],
            "debug": f"YOLO 載入或設定錯誤: {exc}",
        }

    results = model.predict(
        img,
        imgsz=640,
        conf=YOLO_CONF_THRESHOLD,
        verbose=False,
    )

    if not results:
        return {"score": 0.0, "labels": [], "debug": "模型無輸出結果"}

    res = results[0]
    names = res.names
    labels: List[str] = []
    weapon_hits: List[str] = []

    boxes = res.boxes
    if boxes is None or boxes.cls is None or boxes.conf is None:
        return {"score": 0.0, "labels": [], "debug": "模型無輸出結果"}

    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        name = names[int(cls_id)]
        label_text = f"{name} ({conf:.2f})"
        labels.append(label_text)
        if name.lower() in WEAPON_LABELS:
            weapon_hits.append(label_text)

    # 模型只有 knife/gun 兩類，命中時給較高權重
    base_score = 0.05 * len(labels)
    weapon_bonus = 0.5 * len(weapon_hits)
    score = min(1.0, base_score + weapon_bonus)

    debug = (
        "預設指向 weapons/knives 模型（best.onnx）；"
        "若未下載請從專案取得，並在側邊設定中指定路徑"
    )

    return {
        "score": score,
        "labels": labels,
        "debug": debug,
        "weapon_hits": weapon_hits,
    }


# ======== 圖片 OCR（EasyOCR） ========
@st.cache_resource(show_spinner=False)
def load_ocr_reader(langs: List[str]):
    # EasyOCR reader 預設不開啟 GPU（如需可手動調整）
    return easyocr.Reader(langs, gpu=False)


def prepare_ocr_image(img: Image.Image) -> Image.Image:
    """簡單增強：灰階、自動對比、若過小則放大到寬度>=800。"""
    gray = img.convert("L")
    enhanced = ImageOps.autocontrast(gray)
    if enhanced.width < 800:
        ratio = 800 / enhanced.width
        new_size = (int(enhanced.width * ratio), int(enhanced.height * ratio))
        enhanced = enhanced.resize(new_size)
    return enhanced


def extract_text_from_image(
    img: Image.Image,
    langs: List[str],
) -> Dict[str, Optional[str]]:
    """
    使用 EasyOCR 將圖片轉文字。
    - 需先安裝 easyocr（requirements 已列）
    - 回傳 text 及 debug 訊息；失敗時 text 為空、debug 為錯誤訊息
    """
    try:
        prepped = prepare_ocr_image(img)
        reader = load_ocr_reader(langs)
        # EasyOCR 需要 numpy 陣列
        result = reader.readtext(np.array(prepped), detail=0)
        text = "\n".join(result).strip()
        debug_msg = (
            f"EasyOCR (lang={','.join(langs)})；"
            f"行數={len(result)}，長度={len(text)}"
        )
        return {"text": text.strip(), "debug": debug_msg}
    except Exception as exc:
        return {"text": "", "debug": f"OCR 失敗或未安裝 easyocr: {exc}"}


# ======== 總體風險合成 ========
def combine_risk(text_score: float, image_score: float) -> float:
    """
    簡單合成：
    - text / image 分數 0~1
    - 1 - (1 - a) * (1 - b)（任一高即拉高）
    """
    return 1 - (1 - text_score) * (1 - image_score)


def risk_level(score: float) -> str:
    if score >= 0.8:
        return "高風險（建議直接拒絕上架）"
    elif score >= 0.5:
        return "中度風險（建議人工進一步審查）"
    else:
        return "低風險（可上架）"

# ======== Streamlit UI ========
def main():
    st.set_page_config(page_title="電商違規審核系統", page_icon="shield", layout="centered")

    st.title("電商違規審核 Demo（刀具／槍械）")
    st.write("上傳商品圖片與文字，系統會進行 **刀具 / 槍械** 的風險檢查。")

    st.sidebar.header("檢測設定")
    yolo_weights_path = st.sidebar.text_input("YOLO 權重路徑", value=YOLO_DEFAULT_WEIGHTS)
    text_model_path = st.sidebar.text_input(
        "文字分類模型（本地路徑或 Hugging Face 名稱）", value=DEFAULT_TEXT_MODEL
    )
    ocr_langs_input = st.sidebar.text_input(
        "OCR 語言（逗號分隔，建議 ch_tra,en 或 ch_sim,en）",
        value=",".join(DEFAULT_OCR_LANGS),
    )
    raw_langs = [lang.strip() for lang in ocr_langs_input.split(",") if lang.strip()]
    # Chinese_tra / Chinese_sim 只能搭配英文，若用戶填多個非英文僅取第一個
    ocr_langs = []
    non_en = [l for l in raw_langs if l != "en"]
    if non_en:
        ocr_langs.append(non_en[0])
    ocr_langs.append("en")

    mode = st.radio("選擇檢測模式", ["上傳檔案", "網址檢測", "深度網址檢測"], horizontal=True)

    img = None
    title = ""
    description = ""
    url_input = ""

    if mode == "上傳檔案":
        st.header("步驟1：上傳商品內容")
        col1, col2 = st.columns(2)

        with col1:
            uploaded_image = st.file_uploader(
                "上傳商品圖片（jpg / png）",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_image is not None:
                img = Image.open(uploaded_image).convert("RGB")
                st.image(img, caption="商品圖片預覽", use_column_width=True)

        with col2:
            title = st.text_input("商品標題", value="")
            description = st.text_area("商品描述 / 補充說明", height=150)

        run_upload_check = st.button("開始違規審查", type="primary")
        run_url_check = False

    elif mode == "網址檢測":
        st.header("步驟1：輸入商品頁網址")
        url_input = st.text_input("商品頁網址", value="")
        run_url_check = st.button("檢查網址內容", type="primary")
        run_upload_check = False
        run_deep_check = False

    else:
        st.header("步驟1：深度網址檢測")
        url_input = st.text_input("網站起始網址", value="")
        exclude_input = st.text_input("排除路徑（逗號分隔）", value="/login,/cart,/checkout")
        max_pages = st.number_input("最大頁數", min_value=1, max_value=100, value=20, step=1)
        max_depth = st.number_input("最大深度", min_value=0, max_value=5, value=2, step=1)
        run_deep_check = st.button("開始深度網址檢測", type="primary")
        run_upload_check = False
        run_url_check = False

    if run_upload_check:
        if not title and not description and uploaded_image is None:
            st.warning("請至少提供文字或圖片才能檢查。")
            return

        st.header("步驟2：檢查結果")

        ocr_text = ""
        ocr_debug = ""
        if img is not None:
            ocr_result = extract_text_from_image(
                img,
                langs=ocr_langs or DEFAULT_OCR_LANGS,
            )
            ocr_text = ocr_result.get("text", "")
            ocr_debug = ocr_result.get("debug", "無 OCR 訊息")

        full_text = (title or "") + "\n" + (description or "")
        if ocr_text:
            full_text += "\n" + ocr_text
        text_result = (
            analyze_text(full_text, model_path=text_model_path)
            if full_text.strip()
            else {"score": 0.0, "hit_knives": [], "hit_guns": []}
        )

        image_result = (
            analyze_image(img, weights_path=yolo_weights_path)
            if img is not None
            else {"score": 0.0, "labels": [], "debug": "尚未上傳圖片"}
        )

        final_score = combine_risk(text_result["score"], image_result["score"])

        st.subheader("總體風險評估")
        st.metric(label="風險分數（0~1）", value=f"{final_score:.2f}")
        st.write("目前判定：", risk_level(final_score))

        with st.expander("詳細檢查說明", expanded=True):
            st.markdown("### 文字檢查結果")
            st.write(f"文字風險分數：**{text_result['score']:.2f}**")
            if text_result.get("model_label") is not None:
                st.write(f"Transformers 來源：{text_result['source']}，label={text_result['model_label']}，score={text_result['model_score']}")

            if text_result["hit_knives"]:
                st.write("🔪 命中 **刀具** 關鍵字：", ", ".join(set(text_result["hit_knives"])))

            if text_result["hit_guns"]:
                st.write("🔫 命中 **槍械** 關鍵字：", ", ".join(set(text_result["hit_guns"])))

            if not text_result["hit_knives"] and not text_result["hit_guns"]:
                st.write("✅ 文字內容未檢出明顯刀具／槍械關鍵字。")

            if img is not None:
                st.markdown("---")
                st.markdown("### 圖片文字檢測（OCR）")
                st.write(ocr_text if ocr_text.strip() else "（無有效文字）")
                st.caption(ocr_debug or "OCR 完成")

            st.markdown("---")
            st.markdown("### 影像檢查結果（YOLOv8）")
            st.write(f"影像風險分數：**{image_result['score']:.2f}**")
            if image_result.get("weapon_hits"):
                st.write("⚠️ YOLO 命中 **刀具/槍械** 類別：", ", ".join(image_result["weapon_hits"]))
            if image_result.get("labels"):
                st.write("📌 模型偵測清單：", ", ".join(image_result["labels"]))
            st.caption(image_result.get("debug", ""))

        if img is not None:
            st.markdown("### 圖片文字檢測結果")
            st.write(ocr_text if ocr_text.strip() else "（無有效文字）")
            st.caption(ocr_debug or "OCR 完成")

        st.info(
            "YOLOv8 已啟用，預設指向 Weapons-and-Knives-Detector-with-YOLOv8 的 ONNX 權重。"
            "請從該專案下載 best.onnx，放到 weights/weapons-knives-best.onnx，"
            "或在左側設定中指定你的模型路徑。"
        )
        st.caption(
            "文字辨識改用 EasyOCR（預設語言 ch_sim,ch_tra,en），可在左側調整；"
            "文字分類仍採 transformers BERT（預設 bert-base-chinese），可在左側覆蓋；"
            "若模型載入失敗，會回退到關鍵字檢查。"
        )

    if run_url_check:
        if not url_input.strip():
            st.warning("請輸入網址再檢查。")
            return

        st.header("步驟2：網址檢查結果")
        fetch_res = fetch_url_text(url_input.strip())
        page_text = fetch_res.get("text", "")
        page_images = fetch_res.get("images", [])
        debug_info = fetch_res.get("debug", "")
        if not page_text:
            st.error(f"抓取失敗：{debug_info}")
            return

        text_result = analyze_text(page_text, model_path=text_model_path)
        # 下載部分圖片並跑 YOLO
        dl_images, img_debugs = download_images(page_images, limit=5)
        image_scores: List[float] = []
        image_hits: List[str] = []
        image_labels: List[str] = []
        for img in dl_images:
            res = analyze_image(img, weights_path=yolo_weights_path)
            image_scores.append(res["score"])
            image_hits.extend(res.get("weapon_hits", []))
            image_labels.extend(res.get("labels", []))
        image_score = max(image_scores) if image_scores else 0.0
        final_score = combine_risk(text_result["score"], image_score)

        st.subheader("文字檢查")
        st.write(f"文字風險分數：**{text_result['score']:.2f}**")
        if text_result.get("model_label") is not None:
            st.write(f"Transformers 來源：{text_result['source']}，label={text_result['model_label']}，score={text_result['model_score']}")
        if text_result["hit_knives"]:
            st.write("🔪 命中 **刀具** 關鍵字：", ", ".join(set(text_result["hit_knives"])))
        if text_result["hit_guns"]:
            st.write("🔫 命中 **槍械** 關鍵字：", ", ".join(set(text_result["hit_guns"])))
        if not text_result["hit_knives"] and not text_result["hit_guns"]:
            st.write("✅ 文字內容未檢出明顯刀具／槍械關鍵字。")

        st.markdown("### 圖片檢查（從網址抓取）")
        if dl_images:
            st.write(f"下載圖片 {len(dl_images)} 張，YOLO 最高分：**{image_score:.2f}**")
            if image_hits:
                st.write("⚠️ 命中 **刀具/槍械** 類別：", ", ".join(image_hits))
            if image_labels:
                st.write("📌 偵測清單：", ", ".join(image_labels))
            with st.expander("下載/偵測紀錄", expanded=False):
                st.write("\n".join(img_debugs))
        else:
            st.write("（此頁未成功抓取或下載圖片）")

        st.markdown("### 綜合風險（文字+圖片）")
        st.metric("風險分數（0~1）", f"{final_score:.2f}")
        st.write("目前判定：", risk_level(final_score))

        st.markdown("---")
        st.markdown("### 抓取的頁面文字（節錄）")
        st.caption(debug_info)
        st.write(page_text[:2000] + ("..." if len(page_text) > 2000 else ""))

    if run_deep_check:
        if not url_input.strip():
            st.warning("請輸入網站起始網址。")
            return

        base_url = url_input.strip()
        exclude_paths = [p.strip().lower() for p in exclude_input.split(",") if p.strip()]
        max_pages_int = int(max_pages)
        max_depth_int = int(max_depth)

        st.header("步驟2：深度網址檢測結果")
        queue: List[Tuple[str, int]] = [(base_url, 0)]
        visited: set[str] = set()
        results: List[Dict[str, Optional[str]]] = []
        progress = st.progress(0)

        while queue and len(visited) < max_pages_int:
            current_url, depth = queue.pop(0)
            if current_url in visited:
                continue
            if not is_same_domain(base_url, current_url):
                continue
            if is_excluded_path(current_url, exclude_paths):
                continue
            visited.add(current_url)

            page_res = fetch_url_page(current_url, timeout=30)
            page_text = page_res.get("text", "")
            page_images = page_res.get("images", [])
            page_links = page_res.get("links", [])

            text_result = analyze_text(page_text, model_path=text_model_path) if page_text else {"score": 0.0, "hit_knives": [], "hit_guns": []}
            dl_images, _ = download_images(page_images, limit=3)
            image_scores: List[float] = []
            image_hits: List[str] = []
            for img in dl_images:
                res = analyze_image(img, weights_path=yolo_weights_path)
                image_scores.append(res["score"])
                image_hits.extend(res.get("weapon_hits", []))
            image_score = max(image_scores) if image_scores else 0.0
            final_score = combine_risk(text_result["score"], image_score)

            results.append({
                "url": current_url,
                "text_score": f"{text_result['score']:.2f}",
                "image_score": f"{image_score:.2f}",
                "final_score": f"{final_score:.2f}",
                "hits": ", ".join(set(text_result.get("hit_knives", []) + text_result.get("hit_guns", []))),
                "image_hits": ", ".join(set(image_hits)),
            })

            if depth < max_depth_int:
                for link in page_links:
                    if link in visited:
                        continue
                    if not is_same_domain(base_url, link):
                        continue
                    if is_excluded_path(link, exclude_paths):
                        continue
                    queue.append((link, depth + 1))

            progress.progress(min(1.0, len(visited) / max_pages_int))

        st.subheader("掃描摘要")
        st.write(f"已掃描 {len(visited)} 頁，深度上限 {max_depth_int}。")

        if results:
            st.dataframe(results)
        else:
            st.write("（沒有可用的掃描結果）")


if __name__ == "__main__":
    main()





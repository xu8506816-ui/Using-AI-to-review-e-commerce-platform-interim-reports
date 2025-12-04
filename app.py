import os
from typing import Dict, List

import streamlit as st
from PIL import Image
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

YOLO_DEFAULT_WEIGHTS = os.environ.get(
    "YOLO_MODEL_PATH",
    # 建議下載 https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8 的 best.onnx
    # 並放在專案下 weights/weapons-knives-best.onnx
    "weights/weapons-knives-best.onnx",
)
YOLO_CONF_THRESHOLD = 0.25
WEAPON_LABELS = {
    # JoaoAssalim 模型只有兩類
    "knife",
    "gun",
    # 允許大小寫／複數別名
    "knives",
    "guns",
}


# ======== 文字檢查邏輯 ========
def analyze_text(text: str) -> Dict:
    text_lower = text.lower()

    hit_knives: List[str] = []
    hit_guns: List[str] = []

    # 中文關鍵字
    for kw in KNIFE_KEYWORDS:
        if kw in text:
            hit_knives.append(kw)

    for kw in GUN_KEYWORDS:
        if kw in text:
            hit_guns.append(kw)

    # 英文關鍵字
    for kw in EN_KNIFE_KEYWORDS:
        if kw in text_lower:
            hit_knives.append(kw)

    for kw in EN_GUN_KEYWORDS:
        if kw in text_lower:
            hit_guns.append(kw)

    # 簡單風險分數：命中關鍵字就給較高基線
    score = 0.0
    if hit_knives or hit_guns:
        score = min(1.0, 0.6 + 0.1 * (len(hit_knives) + len(hit_guns)))

    return {
        "score": score,
        "hit_knives": hit_knives,
        "hit_guns": hit_guns,
    }


# ======== 影像檢查邏輯（YOLOv8）========
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights: str = YOLO_DEFAULT_WEIGHTS) -> YOLO:
    return YOLO(weights)


def analyze_image(img: Image.Image) -> Dict:
    """
    - 透過環境變數 `YOLO_MODEL_PATH` 指定權重，預設使用 weapons/knives 專案的 best.onnx
    - 若未提供模型檔，請至 https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8
      下載 best.onnx，並放置於 weights/weapons-knives-best.onnx 或自行設置 `YOLO_MODEL_PATH`
    """
    if img is None:
        return {"score": 0.0, "labels": [], "debug": "尚未上傳圖片"}

    try:
        model = load_yolo_model()
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
        "若未下載請從專案取得，並以 `YOLO_MODEL_PATH` 或 weights/weapons-knives-best.onnx 指定路徑"
    )

    return {
        "score": score,
        "labels": labels,
        "debug": debug,
        "weapon_hits": weapon_hits,
    }


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
        return "🚫 高風險（建議直接拒絕上架）"
    elif score >= 0.5:
        return "⚠️ 中度風險（建議人工進一步審查）"
    else:
        return "✅ 低風險（可上架）"


# ======== Streamlit UI ========
def main():
    st.set_page_config(page_title="電商違規審核系統", page_icon="🛡️", layout="centered")

    st.title("🛡️ 電商違規審核 Demo（刀具／槍械）")
    st.write("上傳商品圖片與文字，系統會進行 **刀具 / 槍械** 的風險檢查。")

    # 上傳區塊
    st.header("1️⃣ 上傳商品內容")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_image = st.file_uploader(
            "上傳商品圖片（jpg / png）",
            type=["jpg", "jpeg", "png"]
        )
        img = None
        if uploaded_image is not None:
            img = Image.open(uploaded_image).convert("RGB")
            st.image(img, caption="商品圖片預覽", use_column_width=True)

    with col2:
        title = st.text_input("商品標題", value="")
        description = st.text_area("商品描述 / 補充說明", height=150)

    if st.button("🚀 開始違規審查", type="primary"):
        if not title and not description and uploaded_image is None:
            st.warning("請至少提供文字或圖片才能檢查。")
            return

        st.header("2️⃣ 檢查結果")

        # 文字檢查
        full_text = (title or "") + "\n" + (description or "")
        text_result = analyze_text(full_text) if full_text.strip() else {"score": 0.0, "hit_knives": [], "hit_guns": []}

        # 影像檢查
        image_result = analyze_image(img) if img is not None else {"score": 0.0, "labels": [], "debug": "尚未上傳圖片"}

        # 合併風險
        final_score = combine_risk(text_result["score"], image_result["score"])

        # 總覽
        st.subheader("總體風險評估")
        st.metric(
            label="風險分數（0~1）",
            value=f"{final_score:.2f}"
        )
        st.write("目前判定：", risk_level(final_score))

        # 詳細說明
        with st.expander("📄 詳細檢查說明", expanded=True):
            st.markdown("### 文字檢查結果")
            st.write(f"文字風險分數：**{text_result['score']:.2f}**")

            if text_result["hit_knives"]:
                st.write("🔪 命中 **刀具** 關鍵字：", ", ".join(set(text_result["hit_knives"])))

            if text_result["hit_guns"]:
                st.write("🔫 命中 **槍械** 關鍵字：", ", ".join(set(text_result["hit_guns"])))

            if not text_result["hit_knives"] and not text_result["hit_guns"]:
                st.write("✅ 文字內容未檢出明顯刀具／槍械關鍵字。")

            st.markdown("---")
            st.markdown("### 影像檢查結果（YOLOv8）")
            st.write(f"影像風險分數：**{image_result['score']:.2f}**")
            if image_result.get("weapon_hits"):
                st.write("⚠️ YOLO 命中 **刀具/槍械** 類別：", ", ".join(image_result["weapon_hits"]))
            if image_result.get("labels"):
                st.write("📌 模型偵測清單：", ", ".join(image_result["labels"]))
            st.caption(image_result.get("debug", ""))

        st.info(
            "YOLOv8 已啟用，預設指向 Weapons-and-Knives-Detector-with-YOLOv8 的 ONNX 權重。"
            "請從該專案下載 best.onnx，放到 weights/weapons-knives-best.onnx，"
            "或以環境變數 `YOLO_MODEL_PATH` 指向你的模型路徑。"
        )


if __name__ == "__main__":
    main()

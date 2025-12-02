import streamlit as st
from PIL import Image
from typing import Dict, List

# ======== 規則：關鍵字清單（可以自己再慢慢擴充）========
KNIFE_KEYWORDS = [
    "刀", "小刀", "匕首", "軍刀", "折疊刀", "摺疊刀", "折刀",
    "瑞士刀", "開山刀", "砍刀", "水果刀", "工作刀", "獵刀",
    "開刃", "刀具", "壓刃"
]

GUN_KEYWORDS = [
    "槍", "手槍", "長槍", "步槍", "狙擊槍", "散彈槍", "霰彈槍",
    "BB槍", "bb槍", "氣槍", "模型槍", "仿真槍", "空氣槍",
    "衝鋒槍", "手拉槍", "水彈槍", "水彈"
]

EN_KNIFE_KEYWORDS = [
    "knife", "knives", "dagger", "blade", "machete",
    "folding knife", "pocket knife", "hunting knife"
]

EN_GUN_KEYWORDS = [
    "gun", "pistol", "rifle", "sniper", "shotgun",
    "airsoft", "bb gun", "toy gun", "machine gun"
]


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

    # 風險分數簡單設計：有關鍵字就給比較高分
    score = 0.0
    if hit_knives or hit_guns:
        # 有命中就 0.8 起跳，命中多一點可以再微調
        score = min(1.0, 0.6 + 0.1 * (len(hit_knives) + len(hit_guns)))

    result = {
        "score": score,
        "hit_knives": hit_knives,
        "hit_guns": hit_guns,
    }
    return result


# ======== 圖像檢查邏輯（目前先做 placeholder）========
def analyze_image(img: Image.Image) -> Dict:
    """
    這裡目前是示意用的規則：
    - 真正實作時你可以把 YOLO / Detectron2 / 其他模型接進來
    - 例如：讀取模型 → 推論 → 看有沒有 'knife', 'gun' 類別
    """
    # 先回傳一個中立的結果，讓整個 app 可以跑
    result = {
        "score": 0.0,          # 0~1，之後你可以接模型結果
        "labels": [],          # ['knife', 'gun'] ...
        "debug": "尚未接上圖像模型，目前為示意結果"
    }
    return result


# ======== 最終風險整合 ========
def combine_risk(text_score: float, image_score: float) -> float:
    """
    簡單做一個融合方法：
    - 假設 text / image 都是 0~1
    - 用 1 - (1 - a) * (1 - b) 的方式把兩個風險合併
      （任何一邊高，都會把整體風險拉高）
    """
    return 1 - (1 - text_score) * (1 - image_score)


def risk_level(score: float) -> str:
    if score >= 0.8:
        return "🚫 高風險（建議直接拒絕上架）"
    elif score >= 0.5:
        return "⚠️ 中等風險（建議人工複審）"
    else:
        return "✅ 低風險（可以上架）"


# ======== Streamlit UI ========
def main():
    st.set_page_config(page_title="違禁品審查系統", page_icon="🛡️", layout="centered")

    st.title("🛡️ 電商違禁品審查 Demo（刀／槍枝）")
    st.write("上傳商品圖片與文字，系統會進行 **刀具 / 槍枝** 相關風險檢查。")

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

    if st.button("🔍 開始違禁品檢查", type="primary"):
        if not title and not description and uploaded_image is None:
            st.warning("請至少提供文字或圖片再進行檢查。")
            return

        st.header("2️⃣ 檢查結果")

        # 文字檢查
        full_text = (title or "") + "\n" + (description or "")
        text_result = analyze_text(full_text) if full_text.strip() else {"score": 0.0, "hit_knives": [], "hit_guns": []}

        # 圖像檢查
        image_result = analyze_image(img) if img is not None else {"score": 0.0, "labels": [], "debug": "未上傳圖片"}

        # 整體風險
        final_score = combine_risk(text_result["score"], image_result["score"])

        # 顯示數值
        st.subheader("整體風險評估")
        st.metric(
            label="風險分數（0～1）",
            value=f"{final_score:.2f}"
        )
        st.write("目前判定：", risk_level(final_score))

        # 詳細說明區塊
        with st.expander("📄 詳細檢查說明", expanded=True):
            st.markdown("### 文字檢查結果")
            st.write(f"文字風險分數：**{text_result['score']:.2f}**")

            if text_result["hit_knives"]:
                st.write("🔪 偵測到與 **刀具** 相關的關鍵字：")
                st.write(", ".join(set(text_result["hit_knives"])))

            if text_result["hit_guns"]:
                st.write("🔫 偵測到與 **槍枝** 相關的關鍵字：")
                st.write(", ".join(set(text_result["hit_guns"])))

            if not text_result["hit_knives"] and not text_result["hit_guns"]:
                st.write("✅ 文字內容中未偵測到明顯刀具／槍枝關鍵字。")

            st.markdown("---")
            st.markdown("### 圖像檢查結果（目前為示意）")
            st.write(f"圖像風險分數：**{image_result['score']:.2f}**")
            if image_result.get("labels"):
                st.write("偵測到疑似物件：", ", ".join(image_result["labels"]))
            st.caption(image_result.get("debug", ""))

        st.info("⚙️ 提示：目前圖像部分只是範例，你可以把這裡接成 YOLO / Detectron2 / 其他模型的輸出。")


if __name__ == "__main__":
    main()

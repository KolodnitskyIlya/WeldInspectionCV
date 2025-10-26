import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd
from io import StringIO

st.set_page_config(page_title="WeldInspectionCV – Анализ рентгеновских снимков", layout="wide")
st.title("Пакетный анализ рентгеновских снимков сварных швов")

# ===================== Предобработка =====================
def adaptive_contrast_strong(img_gray: np.ndarray) -> np.ndarray:
    """Усиление контраста с адаптивной коррекцией"""
    std = np.std(img_gray)
    if std < 40:
        gamma = 0.6
    elif std < 70:
        gamma = 1.0
    else:
        gamma = 1.4
    img_gamma = np.array(255 * (img_gray / 255) ** gamma, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gamma)
    img_norm = cv2.normalize(img_clahe, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm


def preprocess_image(uploaded_file, target_size=(640, 640)):
    """Полная предобработка изображения"""
    img = Image.open(uploaded_file).convert("L")
    img_np = np.array(img)
    img_contrasted = adaptive_contrast_strong(img_np)
    img_resized = cv2.resize(img_contrasted, target_size)
    img_for_model = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    return img_for_model

# ===================== Загрузка модели =====================
@st.cache_resource
def load_model(path="best.pt"):
    return YOLO(path)

model = load_model("best.pt")

# ===================== Классы дефектов =====================
CLASS_DICT = {
    0: 'пора',
    1: 'включение',
    2: 'подрез',
    3: 'эталон1',
    4: 'эталон2',
    5: 'утяжина',
    6: 'несплавление',
    7: 'непровар корня'
}

# ===================== Стандартные пределы (мм) =====================
DEFAULT_LIMITS = {
    'пора': {'diameter': {'ok': 1.5, 'reject': 3.0}},
    'включение': {'length': {'ok': 3.0, 'reject': 5.0}, 'width': {'ok': 0.5, 'reject': 1.0}},
    'подрез': {'depth': {'ok': 0.5, 'reject': 1.0}},
    'несплавление': {'length': {'ok': 10.0, 'reject': 25.0}},
    'непровар корня': {'length': {'ok': 10.0, 'reject': 25.0}},
    'эталон1': {},
    'эталон2': {},
    'утяжина': {}
}

# ===================== Логика оценки дефектов =====================
def classify_defect(defect_type, area, length, width, limits):
    if defect_type not in limits:
        return "Нет нормы", 0.5
    rule = limits[defect_type]
    if defect_type == 'пора':
        diameter = (4 * area / np.pi) ** 0.5
        ok, reject = rule['diameter']['ok'], rule['diameter']['reject']
        if diameter <= ok:
            return "Допустимо", 0.0
        elif diameter > reject:
            return "Недопустимо", 1.0
        else:
            closeness = (diameter - ok) / (reject - ok)
            return "Требует проверки", closeness
    elif defect_type == 'включение':
        ok_L, reject_L = rule['length']['ok'], rule['length']['reject']
        ok_W, reject_W = rule['width']['ok'], rule['width']['reject']
        if length <= ok_L and width <= ok_W:
            return "Допустимо", 0.0
        elif length > reject_L or width > reject_W:
            return "Недопустимо", 1.0
        else:
            closeness_L = (length - ok_L) / (reject_L - ok_L)
            closeness_W = (width - ok_W) / (reject_W - ok_W)
            return "Требует проверки", max(closeness_L, closeness_W)
    elif defect_type == 'подрез':
        ok, reject = rule['depth']['ok'], rule['depth']['reject']
        if width <= ok:
            return "Допустимо", 0.0
        elif width > reject:
            return "Недопустимо", 1.0
        else:
            return "Требует проверки", (width - ok) / (reject - ok)
    elif defect_type in ['несплавление', 'непровар корня']:
        ok, reject = rule['length']['ok'], rule['length']['reject']
        if length <= ok:
            return "Допустимо", 0.0
        elif length > reject:
            return "Недопустимо", 1.0
        else:
            return "Требует проверки", (length - ok) / (reject - ok)
    elif defect_type in ['эталон1', 'эталон2', 'эталон3']:
        return "Допустимо", 0.0
    elif defect_type == 'трещина':
        return "Недопустимо", 1.0
    return "Нет нормы", 0.5

# ===================== Оценка результатов YOLO =====================
def evaluate_defects(yolo_output, limits=DEFAULT_LIMITS):
    results_eval = []
    for box in yolo_output:
        xyxy = box.xyxy.cpu().numpy()[0]
        length = xyxy[2] - xyxy[0]
        width = xyxy[3] - xyxy[1]
        area = length * width
        defect_type = CLASS_DICT.get(int(box.cls.cpu().numpy()[0]), "неизвестно")
        severity, closeness = classify_defect(defect_type, area, length, width, limits)
        results_eval.append({
            "Тип дефекта": defect_type,
            "Уверенность": float(box.conf.cpu().numpy()[0]),
            "Статус": severity,
            "Серьёзность": closeness
        })
    return pd.DataFrame(results_eval)

# ===================== Настройки боковой панели =====================
st.sidebar.header("Настройки анализа")
conf_thresh = st.sidebar.slider("Порог уверенности", 0.0, 1.0, 0.25)
iou_thresh = st.sidebar.slider("Порог IoU", 0.0, 1.0, 0.45)

# ===================== Пакетная загрузка изображений =====================
uploaded_files = st.file_uploader(
    "Загрузите несколько рентгеновских снимков",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} файлов выбрано.")
    if st.button("Запустить пакетный анализ"):
        batch_results = []
        with st.spinner("Анализ изображений... ⏳"):
            for file in uploaded_files:
                # === Предобработка ===
                img_for_model = preprocess_image(file)

                # === Анализ YOLO ===
                results = model.predict(img_for_model, imgsz=640, conf=conf_thresh, iou=iou_thresh)
                detections = results[0].boxes

                if len(detections) == 0:
                    verdict = "✅ Допустимо (дефекты не обнаружены)"
                    defect_count = 0
                else:
                    df_eval = evaluate_defects(detections)
                    defect_count = len(df_eval)
                    if any(df_eval["Статус"] == "Недопустимо"):
                        verdict = "❌ Недопустимо"
                    else:
                        verdict = "⚠️ Требует проверки / Допустимо"

                batch_results.append({
                    "Имя файла": file.name,
                    "Обнаруженные дефекты": defect_count,
                    "Вердикт": verdict
                })

        df_batch = pd.DataFrame(batch_results)
        st.success("Пакетный анализ завершён!")

        st.dataframe(df_batch, use_container_width=True)

        # ===================== Сводка + Загрузка CSV =====================
        total_files = len(df_batch)
        unacceptable = sum(df_batch["Вердикт"].str.contains("Недопустимо"))
        avg_defects = df_batch["Обнаруженные дефекты"].mean()

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Сводка")
            st.markdown(f"- **Всего файлов:** {total_files}")
            st.markdown(f"- **Среднее количество дефектов на файл:** {avg_defects:.2f}")
            st.markdown(f"- **Недопустимые изображения:** {unacceptable} ({unacceptable / total_files * 100:.1f}%)")

        with col2:
            csv_buffer = StringIO()
            df_batch.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Скачать CSV",
                data=csv_buffer.getvalue(),
                file_name="batch_analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )

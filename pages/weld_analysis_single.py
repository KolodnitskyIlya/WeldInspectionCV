import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd
from io import StringIO

# ===================== Предобработка =====================
def adaptive_contrast_strong(img_gray: np.ndarray) -> np.ndarray:
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


def preprocess_uploaded_image(uploaded_file, target_size=(640, 640)):
    img = Image.open(uploaded_file).convert("L")
    img_np = np.array(img)
    img_contrasted = adaptive_contrast_strong(img_np)
    img_resized = cv2.resize(img_contrasted, target_size)
    img_for_model = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    return img, img_for_model


# ===================== Интерфейс =====================
st.set_page_config(page_title="WeldInspectionCV – Анализ рентгеновского снимка", layout="wide")
st.title("Анализ рентгеновского снимка сварного шва")

# ===================== Модель =====================
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

# ===================== Боковая панель =====================
st.sidebar.header("Настройки анализа")
conf_thresh = st.sidebar.slider("Порог уверенности", 0.0, 1.0, 0.25)
iou_thresh = st.sidebar.slider("Порог IoU", 0.0, 1.0, 0.45)
show_boxes = st.sidebar.checkbox("Показывать рамки", value=True)
phys_size_mm = st.sidebar.slider("Физический размер шва на фото (мм)", 1, 50, 15)

st.sidebar.markdown("---")
st.sidebar.subheader("Пределы дефектов (Пользовательские, мм)")
user_limits = {}
for defect_name, params in DEFAULT_LIMITS.items():
    if not params:
        continue
    st.sidebar.markdown(f"**{defect_name.capitalize()}**")
    user_limits[defect_name] = {}
    for param_name, values in params.items():
        ok_val = st.sidebar.number_input(
            f"{param_name} допустимо ({defect_name}, мм)",
            value=values['ok'], step=0.1, key=f"{defect_name}_{param_name}_ok"
        )
        reject_val = st.sidebar.number_input(
            f"{param_name} недопустимо ({defect_name}, мм)",
            value=values['reject'], step=0.1, key=f"{defect_name}_{param_name}_reject"
        )
        user_limits[defect_name][param_name] = {'ok': ok_val, 'reject': reject_val}


# ===================== Классификация =====================
def classify_defect(defect_type, area, length, width, limits):
    if defect_type in ['эталон1', 'эталон2']:
        return "Допустимо", 0.0
    elif defect_type == 'пора':
        rule = limits['пора']
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
        rule = limits['включение']
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
    elif defect_type == 'трещина':
        return "Недопустимо", 1.0
    elif defect_type in ['несплавление', 'непровар корня']:
        rule = limits[defect_type]
        ok, reject = rule['length']['ok'], rule['length']['reject']
        if length <= ok:
            return "Допустимо", 0.0
        elif length > reject:
            return "Недопустимо", 1.0
        else:
            return "Требует проверки", (length - ok) / (reject - ok)
    elif defect_type == 'подрез':
        rule = limits['подрез']
        ok, reject = rule['depth']['ok'], rule['depth']['reject']
        if width <= ok:
            return "Допустимо", 0.0
        elif width > reject:
            return "Недопустимо", 1.0
        else:
            return "Требует проверки", (width - ok) / (reject - ok)
    elif defect_type not in limits or defect_type not in ['пора', 'включение', 'подрез', 'несплавление', 'непровар корня', 'трещина']:
        return "Нет нормы", 0.5
    else:
        return "Допустимо", 0.0


# ===================== Анализ дефектов =====================
def evaluate_defects(yolo_output, model_results, img_w_px, img_h_px, img_size_mm=15, limits=DEFAULT_LIMITS):
    results_eval = []
    px_to_mm_w = img_size_mm / img_w_px
    px_to_mm_h = img_size_mm / img_h_px

    for i, box in enumerate(yolo_output):
        xyxy = box.xyxy.cpu().numpy()[0]
        length_mm = (xyxy[2] - xyxy[0]) * px_to_mm_w
        width_mm = (xyxy[3] - xyxy[1]) * px_to_mm_h
        area_mm2 = length_mm * width_mm

        defect_type = CLASS_DICT.get(int(box.cls.cpu().numpy()[0]), "неизвестно")
        severity, closeness = classify_defect(defect_type, area_mm2, length_mm, width_mm, limits)

        seg_coords_norm = []
        if hasattr(model_results[0], "masks") and model_results[0].masks is not None:
            if i < model_results[0].masks.data.shape[0]:
                mask_np = model_results[0].masks.data[i].cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    cnt = contours[0]
                    for pt in cnt:
                        x_norm = round(pt[0][0] / img_w_px, 4)
                        y_norm = round(pt[0][1] / img_h_px, 4)
                        seg_coords_norm.append([x_norm, y_norm])

        results_eval.append({
            "№": len(results_eval) + 1,
            "Тип дефекта": defect_type,
            "Уверенность": round(float(box.conf.cpu().numpy()[0]), 2),
            "Статус": severity,
            "Серьёзность (см. примечание)": round(closeness, 2),
            "Координаты сегмента (нормированные)": seg_coords_norm
        })
    return results_eval


# ===================== Работа с изображением =====================
uploaded_file = st.file_uploader("Загрузите рентгеновский снимок", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.model_results = None
        st.session_state.last_uploaded_file = uploaded_file.name
    img_original, img_for_model = preprocess_uploaded_image(uploaded_file, target_size=(640, 640))
    orig_w, orig_h = img_original.size
    proc_h, proc_w = img_for_model.shape[:2]
    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Исходное изображение")
        st.image(np.array(img_original), channels="L", width=400)

    if st.button("Запустить анализ"):
        with st.spinner("Модель анализирует изображение... ⏳"):
            st.session_state.model_results = model.predict(
                img_for_model, imgsz=640, conf=conf_thresh, iou=iou_thresh
            )

    if "model_results" in st.session_state and st.session_state.model_results:
        detections = st.session_state.model_results[0].boxes
        img_display = np.array(img_original.convert("RGB"))

        if show_boxes and len(detections) > 0:
            for i, box in enumerate(detections):
                xyxy = box.xyxy.cpu().numpy()[0]
                x1, y1 = int(xyxy[0] * scale_x), int(xyxy[1] * scale_y)
                x2, y2 = int(xyxy[2] * scale_x), int(xyxy[3] * scale_y)
                cls_id = int(box.cls.cpu().numpy()[0])
                label = str(cls_id)
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_display, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        with col2:
            st.markdown("#### Сегментированное изображение")
            st.image(img_display, width=400)

        st.markdown("---")
        st.subheader("Результаты анализа")

        if len(detections) == 0:
            st.warning("Дефекты не обнаружены.")
        else:
            df_results = pd.DataFrame(
                evaluate_defects(
                    detections,
                    st.session_state.model_results,
                    img_w_px=img_for_model.shape[1],
                    img_h_px=img_for_model.shape[0],
                    img_size_mm=phys_size_mm,
                    limits=user_limits
                )
            )
            st.dataframe(df_results, use_container_width=True)

            csv_buffer = StringIO()
            df_results.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Скачать результаты (CSV)",
                data=csv_buffer.getvalue(),
                file_name="weld_defect_analysis.csv",
                mime="text/csv"
            )

        st.success("Анализ завершён!")
        if len(detections) > 0:
            st.markdown("***Серьёзность** — вероятность недопустимости дефекта (0 — допустимо, 1 — недопустимо)*")

import streamlit as st

st.set_page_config(
    page_title="WeldInspectionCV – Анализ сварных швов",
    layout="wide",
    initial_sidebar_state="expanded"
)

is_dark = st.get_option("theme.base") == "dark"
text_color = "white" if is_dark else "gray"

st.markdown(
    """
    <div style="padding:50px;text-align:center;">
        <h1 style="font-size:55px;font-weight:bold;">
        <span style="color:#4f46e5;">WeldInspectionCV</span> — интеллектуальный контроль сварных соединений
        </h1>
        <p style="font-size:18px;color:gray;max-width:850px;margin:auto;">
        Система на базе искусственного интеллекта, которая автоматически анализирует рентгеновские снимки сварных швов, 
        обнаруживает и классифицирует дефекты, оценивает их серьёзность и формирует отчёты.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.header("Основные возможности")
features = [
    {"title": "Детекция дефектов", "desc": "Автоматическое нахождение дефектов с высокой точностью."},
    {"title": "Классификация", "desc": "Определение типа каждого дефекта на основе обученной модели."},
    {"title": "Генерация отчётов", "desc": "Структурированный отчёт с классами дефектов."},
    {"title": "Сегментация", "desc": "Пиксельная маска каждого дефекта (Instance Segmentation)."},
    {"title": "Улучшение контраста", "desc": "Автоматическое улучшение низкоконтрастных изображений для более точного выявления мелких дефектов."},
    {"title": "Оценка серьёзности дефекта", "desc": "Система оценивает потенциальную недопустимость дефекта на основе его размера и расположения."},
    {"title": "Пользовательские критерии", "desc": "Возможность задавать собственные значения для «допустимо», «недопустимо» и «требующий проверки»."},
    {"title": "Высокая скорость предсказания", "desc": "Модель выполняет анализ изображения менее чем за 50 мс, обеспечивая мгновенный отклик."}
]
cols = st.columns(3)
for i, feat in enumerate(features):
    with cols[i % 3]:
        st.subheader(feat["title"])
        st.write(feat["desc"])

st.markdown("---")

# ===================== Блок описания модели и метрик =====================
st.header("Описание модели и метрики")

st.markdown(f"""
<div style="background:transparent;padding:30px;border-radius:20px;width:100%;box-sizing:border-box;">
    <h3 style="color:#4f46e5;">Модель WeldInspectionCV</h3>
    <p style="color:{text_color} !important; max-width:100% !important;">
        Используется нейросеть на основе YOLOv9c-seg (для детекции и сегментации), дообученная на рентгеновских снимках сварных швов.
        Модель автоматически распознаёт 13 типов дефектов и оценивает их серьёзность.
    </p>
    <h4 style="color:#4f46e5;">Основные метрики:</h4>
    <ul style="color:{text_color} !important; max-width:100% !important;">
        <li><b>mAP50-95 (seg) = 30.2%</b> — основной показатель точности сегментации по диапазону IoU от 0.5 до 0.95.</li>
        <li><b>mAP50-95 (box) = 29.4%</b> — дополнительный показатель точности обнаружения объектов в виде боксов.</li>
        <li><b>Precision = 54.7%</b> — качество предсказаний: доля верно предсказанных дефектов среди всех обнаруженных.</li>
        <li><b>Recall = 29.0%</b> — полнота: доля найденных дефектов от всех существующих на изображении.</li>
    </ul>
</div>
""", unsafe_allow_html=True)


st.markdown("---")

st.markdown(f"""
<div style="padding:40px;text-align:center;background:#f3f4f6;border-radius:20px;">
  <h2 style="font-size:32px;font-weight:bold;color:black;">Готовы протестировать?</h2>
  <p style="font-size:16px;color:black;max-width:700px;margin:auto;">
    Перейдите во вкладку <b>“X-Ray Анализ”</b> или <b>“Batch Analysis”</b> в верхнем меню, чтобы загрузить изображение и запустить анализ дефектов.
  </p>
</div>
""", unsafe_allow_html=True)

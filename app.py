import streamlit as st

page_1 = st.Page("pages/home.py", title="Главная страница")
page_2 = st.Page("pages/weld_analysis_single.py", title="Анализ рентгеновского снимка сварного шва")
page_3 = st.Page("pages/weld_analysis_batch.py", title="Пакетный анализ рентгеновских снимков сварных швов")

pg = st.navigation([page_1, page_2, page_3])

pg.run()
import streamlit as st

st.set_page_config(page_title="Мои дашборды", page_icon="📊")

st.title("📊 Marketing & E-Commerce Analytics")
st.markdown("### Отчет")

dashboards = [
    {"name": "A/B Тестирование", "url": "https://marketing---e-commerce-analytics-3fnpag8vvrqyb46evvadud.streamlit.app/"},
    {"name": "Воронка / Трафик / Устройства", "url": "https://marketing---e-commerce-analytics-rgucraef3nahjstsklhbkr.streamlit.app/"},
    {"name": "Продажи", "url": "https://marketing---e-commerce-analytics-wxjcqwhbsrmtaxpjjms6mq.streamlit.app/"},
    {"name": "Профиль клиентов", "url": "https://marketing---e-commerce-analytics-skigjbgbgkps8ebcprm6ve.streamlit.app/"},
    {"name": "Профиль кампаний", "url": "https://marketing---e-commerce-analytics-impqdc66jmx2hwpjah9zn4.streamlit.app/"},
]

for db in dashboards:
    st.markdown(f"### {db['name']}")
    st.markdown(f"[Открыть]({db['url']})")
    st.markdown("---")
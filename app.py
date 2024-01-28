import streamlit as st
import datetime
from st_pages import Page, show_pages

show_pages(
    [
        Page('app.py', 'Главная'),
        Page('pages/user_input_page.py', 'Ввод пользовательской информации'),
        Page('pages/analitycs_page.py', 'Аналитика'),
        Page('pages/results.py', 'Результаты'),
        Page('pages/OCR.py', 'OCR')
    ]
)

fa = st.button('get')

if fa:
    time = datetime.datetime.now()
    st.write(time)
    
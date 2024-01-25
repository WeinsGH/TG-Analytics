from datetime import datetime
import streamlit as st
import plotly.express as px
import pandas as pd
import nltk
import os

from funcs.charts import common_charts, comparative_analysis, keyword_analysis

# Скачивание необходимых компонентов NLTK
nltk.download('stopwords')
nltk.download('punkt')

def func(path):
    df = pd.read_csv(path)
    return df


st.header("""
 Аналитика полученных данных
""", divider='blue')

files = [f for f in os.listdir('ldb/tasks/completed') if os.path.isfile(os.path.join('ldb/tasks/completed', f))]
files.append('data.csv')

selected_database = st.selectbox('Что будем смотреть?', files)
if selected_database != 'data.csv':
    path = f'ldb/SAcompleted/{selected_database}'
else:
    path = 'ldb/data/data.csv'

df = func(path)

# Преобразование строчных данных в datetime формат
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['date'] = df['date'].dt.date

# Преобразование числовых значений в текстовые
df['predicted_class'] = df['predicted_class'].map({0: 'нейтральный', 1: 'позитивный', 2: 'негативный'})

# Ввод пользователем периода для анализа
def_start_date = datetime.strptime('01/01/2023', '%d/%m/%Y').date()
def_end_date = datetime.strptime('01/08/2023', '%d/%m/%Y').date()
period_selector = st.date_input('Выберите нужный период', (def_start_date, def_end_date), key=1, format="DD/MM/YYYY")

# Блок выбора цветовой схемы для отображения графиков
color_schemes = {
        "Plotly": px.colors.qualitative.Plotly,
        "G10": px.colors.qualitative.G10,
        "T10": px.colors.qualitative.T10,
        "D3": px.colors.qualitative.D3,
        "Pastel": px.colors.qualitative.Pastel,
        "Dark24": px.colors.qualitative.Dark24,
        "Optimal": {'нейтральный': 'lightgrey', 'позитивный': 'lightgreen', 'негативный': 'lightcoral'}
    }

color_scheme = st.selectbox(
    "Выберите цветовую схему для графиков:",
    list(color_schemes.keys()),
    index=6
    )

color = color_schemes[color_scheme]

# Фильтрация выгрузки по пользовательскому периоду
df_filtered = df[(df['date'] >= period_selector[0]) & (df['date'] <= period_selector[1])]

# Блок для возможности выгрузки отфильтрованных данных
@st.cache_data
def convert_df(df):
    return df.iloc[:, :-1].to_csv(index=False).encode('utf-8')

csv = convert_df(df_filtered)

st.download_button(
    label="Выгрузка данных парсинга в формате CSV",
    data=csv,
    file_name='chanel_posts_proceed_filtered.csv',
    mime='text/csv',
)

# Блок по построению графиков
option = st.selectbox(
    'Какой анализ по выгрузке Вы хотите произвести?',
    ('Сводный анализ', 
     'Анализ по ключевому слову', 
     'Сравнительный анализ ключевых слов')
    )

if option == 'Сводный анализ':
    common_charts(df_filtered, color_scheme, color)
elif option == 'Сравнительный анализ ключевых слов':
    comparative_analysis(df_filtered, color_scheme, color)
elif option == 'Анализ по ключевому слову':
    keyword_analysis(df_filtered, color_scheme, color)
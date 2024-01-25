import streamlit as st
import pandas as pd
import os

# Загрузите ваш датафрейм
def func(path):
    df = pd.read_csv(path)

# Преобразование 'post_date' в datetime
    if df['date'].dtypes != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Замена nan в 'summary' текстом из 'text'
    df['summary'] = df.apply(lambda row: row['text'] if pd.isna(row['summary']) else row['summary'], axis=1)

    return df


files = [f for f in os.listdir('ldb/tasks/completed') if os.path.isfile(os.path.join('ldb/tasks/completed', f))]
files.append('data.csv')

selected_database = st.selectbox('Что будем смотреть?', files)
if selected_database != 'data.csv':
    path = f'ldb/SAcompleted/{selected_database}'
else:
    path = 'ldb/data/data.csv'

df = func(path)

# Получение уникальных каналов
unique_channels = df['chanel'].unique()

# Виджет для выбора нескольких каналов
selected_channels = st.multiselect("Выберите каналы", unique_channels)

# Виджет для выбора даты
selected_date = st.date_input("Выберите дату", value=pd.to_datetime('today').date())

# Фильтрация данных по выбранной дате и каналам
filtered_data = df[(df['date'].dt.date == selected_date) & (df['chanel'].isin(selected_channels))]

col1, col2 = st.columns([4, 1])
with col1:
    # Вывод информации о количестве постов
    st.write(f'Всего постов: {len(filtered_data)}')
with col2:
    # Есть декоратор. Я был бы рад сказать, что это он, но это декорация
    joke = st.button('summary')
    if joke:
            st.toast('🔴 Сейчас недоступно, потому что сервер превратится в кирпич.. Или мы были заняты другим и забыли')

# Отображение отфильтрованных данных
if not filtered_data.empty:
    for index, row in filtered_data.iterrows():
        st.markdown(f"##### Канал: {row['chanel']}")
        st.markdown(f"##### [Ссылка на пост]({row['link']})")
        st.markdown(f"##### Дата поста: {row['date'].strftime('%Y-%m-%d')}")
        st.markdown(f"##### Сводка: \n{row['summary']}")
        st.markdown(f"###### BS precision: {row['bert_score_precision']}        BS recall: {row['bert_score_recall']}          f1 BS: {row['bert_score_f1']}")
        st.markdown("---")  # Для отображения горизонтальной линии
else:
    st.write("За выбранные дату и каналы данных нет.")

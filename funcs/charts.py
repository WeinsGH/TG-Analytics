import string
import streamlit as st
import plotly.express as px
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def common_charts(df: pd.DataFrame, color_scheme_name: str, color_scheme_value):
    st.subheader("""
    Сводный анализ по выгрузке
    """, divider='blue')
    # Подсчет количества значений для каждого класса в 'predicted_class'
    class_counts = df['predicted_class'].value_counts().reset_index()
    class_counts.columns = ['Класс', 'Количество']

    # Выбор типа графика
    chart_type = st.selectbox(
        "Выберите тип графика",
        ["Столбчатая диаграмма", "Круговая диаграмма"]
    )

    # Построение выбранного типа графика
    if chart_type == "Столбчатая диаграмма":
        if color_scheme_name == "Optimal":
            fig = px.bar(class_counts, x='Класс', y='Количество', title='Распределение тональности постов',
                         color='Класс', color_discrete_map=color_scheme_value)
        else:
            fig = px.bar(class_counts, x='Класс', y='Количество', title='Распределение тональности постов',
                         color='Класс', color_discrete_sequence=color_scheme_value)
        fig.update_layout(showlegend=False)
    elif chart_type == "Круговая диаграмма":
        if color_scheme_name == "Optimal":
            fig = px.pie(class_counts, names='Класс', values='Количество', title='Распределение тональности постов',
                         color='Класс', color_discrete_map=color_scheme_value)
        else:
            fig = px.pie(class_counts, names='Класс', values='Количество', title='Распределение тональности постов',
                         color='Класс', color_discrete_sequence=color_scheme_value)
    # Вывод графика в Streamlit
    st.plotly_chart(fig)

    # Выбор периода построения графика

    periods_dict = {'День': 'D', 'Неделя': 'W', 'Месяц': 'M', 'Год': 'Y'}

    period = st.selectbox(
        "Выберите период для построения графика:",
        periods_dict.keys(),
        index=2
    )
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Создание графика
    periodic_sentiment_count_all = df.groupby(
        [pd.Grouper(key='date', freq=periods_dict[period]), 'predicted_class']).size().reset_index(name='counts')

    periods_shorts = {'День': 'дням', 'Неделя': 'неделям', 'Месяц': 'месяцам', 'Год': 'годам'}

    # Создание графика с использованием Plotly
    if color_scheme_name == "Optimal":
        fig = px.bar(periodic_sentiment_count_all, x='date', y='counts', color='predicted_class',
                     title=f'Соотношение всех типов тональности постов по {periods_shorts[period]}',
                     color_discrete_map=color_scheme_value)
    else:
        fig = px.bar(periodic_sentiment_count_all, x='date', y='counts', color='predicted_class',
                     title=f'Соотношение всех типов тональности постов по {periods_shorts[period]}',
                     color_discrete_sequence=color_scheme_value)
    fig.update_xaxes(title_text='Дата поста')
    fig.update_yaxes(title_text='Количество')
    fig.update_layout(legend_title_text='Тональность')
    st.plotly_chart(fig)

    if color_scheme_name == "Optimal":
        fig = px.line(periodic_sentiment_count_all, x='date', y='counts', color='predicted_class',
                      title=f'Динамика количества постов по тональностям по {periods_shorts[period]}',
                      color_discrete_map=color_scheme_value)
    else:
        fig = px.line(periodic_sentiment_count_all, x='date', y='counts', color='predicted_class',
                      title=f'Динамика количества постов по тональностям по {periods_shorts[period]}',
                      color_discrete_sequence=color_scheme_value)
    fig.update_xaxes(title_text='Дата поста')
    fig.update_yaxes(title_text='Количество')
    fig.update_layout(legend_title_text='Тональность')
    st.plotly_chart(fig)

    st.write('Облака слов по тональности постов')
    # Ваш кастомный список слов для удаления
    custom_words_to_remove = ['дек', 'янв', 'году', "это", "года", "изза", "за", "аэрофлот", "аэрофлота", "млрд",
                              "также", "россии", "bilety_bilety", "из", "biletybilety"]

    # Функция для удаления слов из текста
    def remove_custom_words(text):
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление знаков препинания
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Разбиение текста на слова
        words = text.split()
        # Удаление кастомных слов
        filtered_words = [word for word in words if word not in custom_words_to_remove]
        return ' '.join(filtered_words)

    # Применение функции к столбцу текста в DataFrame
    df['text'] = df['text'].apply(remove_custom_words)

    def remove_stopwords(text):
        stop_words = set(stopwords.words('russian'))  # Используйте 'english' для английского языка
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        return ' '.join(filtered_text)

    df['text'] = df['text'].apply(remove_stopwords)

    # Функция для создания облака слов
    def create_wordcloud(text, title):
        fig, ax = plt.subplots(figsize=(5, 5))
        # Генерация облака слов
        wordcloud = WordCloud(width=400, height=400, background_color='white').generate(text)
        # Отображение облака слов
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title)
        ax.axis('off')

        # Отображение фигуры в Streamlit
        st.pyplot(fig)

    # Создание текстов для каждой категории
    positive_text = ' '.join(df[df['predicted_class'] == 'позитивный']['text'])
    neutral_text = ' '.join(df[df['predicted_class'] == 'нейтральный']['text'])
    negative_text = ' '.join(df[df['predicted_class'] == 'негативный']['text'])

    col1, col2, col3 = st.columns(3)
    with col1:
        create_wordcloud(positive_text, 'Положительные отзывы')
    with col2:
        create_wordcloud(neutral_text, 'Нейтральные отзывы')
    with col3:
        create_wordcloud(negative_text, 'Отрицательные отзывы')


def comparative_analysis(df: pd.DataFrame, color_scheme_name: str, color_scheme_value):
    st.subheader("""
    Сравненительный анализ ключевых слов
    """, divider='blue')

    user_input = st.text_input("**Введите два ключевых слова, разделенных пробелом**", "Аэрофлот s7")

    # Разделение ввода на два ключа
    brands = user_input.split()
    if len(brands) >= 2:
        brand1, brand2 = brands[0], brands[1]

        # Фильтрация DataFrame для строк с упоминанием первого ключа
        df_1 = df[df['text'].str.contains(brand1, case=False, na=False)]

        # Фильтрация DataFrame для строк с упоминанием второго ключа
        df_2 = df[df['text'].str.contains(brand2, case=False, na=False)]

    merged_df = pd.concat([df_1, df_2], ignore_index=True)

    @st.cache_data
    def convert_df(df):
        return df.iloc[:, :-1].to_csv(index=False).encode('utf-8')

    csv = convert_df(merged_df)

    st.download_button(
        label="Выгрузка данных по ключам в формате CSV",
        data=csv,
        file_name='chanel_posts_proceed_filtered_keywords.csv',
        mime='text/csv',
    )

    chanels_list = ['Все каналы']
    chanels_list.extend(df['chanel'].unique().tolist())
    chanel_selector = st.selectbox('Выберите канал для анализа по упоминаемости и тональности', chanels_list, index=0)

    # Подсчет количества упоминаний ключей по всем каналам
    channel_counts_1 = df_1['chanel'].value_counts().reset_index()
    channel_counts_1.columns = ['Канал', 'Количество']
    channel_counts_2 = df_2['chanel'].value_counts().reset_index()
    channel_counts_2.columns = ['Канал', 'Количество']

    if chanel_selector == 'Все каналы':
        # Создание графиков с использованием Plotly
        if color_scheme_name == "Optimal":
            color_discrete_map = color_scheme_value
        else:
            color_discrete_sequence = color_scheme_value

        fig_1 = px.bar(channel_counts_1, x='Канал', y='Количество',
                       title=f'Распределение упоминаний<br>{brand1} по каналам',
                       color_discrete_sequence=color_discrete_sequence if color_scheme_name != "Optimal" else None,
                       color='Канал' if color_scheme_name == "Optimal" else None,
                       color_discrete_map=color_discrete_map if color_scheme_name == "Optimal" else None)
        fig_1.update_layout(xaxis_title='Канал', yaxis_title='Количество')
        fig_1.update_xaxes(showticklabels=False)
        fig_2 = px.bar(channel_counts_2, x='Канал', y='Количество',
                       title=f'Распределение упоминаний<br>{brand2} по каналам',
                       color_discrete_sequence=color_discrete_sequence if color_scheme_name != "Optimal" else None,
                       color='Канал' if color_scheme_name == "Optimal" else None,
                       color_discrete_map=color_discrete_map if color_scheme_name == "Optimal" else None)
        fig_2.update_layout(xaxis_title='Канал', yaxis_title='Количество', xaxis_tickangle=-45)
        fig_2.update_xaxes(showticklabels=False)

    else:
        # Подсчет количества упоминаний ключей по каналам
        chanel_post_counts_1 = df_1[df_1['chanel'] == chanel_selector]['predicted_class'].value_counts().reset_index()
        chanel_post_counts_1.columns = ['Канал', 'Количество']
        chanel_post_counts_2 = df_2[df_2['chanel'] == chanel_selector]['predicted_class'].value_counts().reset_index()
        chanel_post_counts_2.columns = ['Канал', 'Количество']

        # Создание графиков с использованием Plotly
        if color_scheme_name == "Optimal":
            color_discrete_map = color_scheme_value
        else:
            color_discrete_sequence = color_scheme_value

        fig_1 = px.bar(chanel_post_counts_1, x='Канал', y='Количество',
                       title=f'Распределение упоминаний<br>{brand1} по тональности в {chanel_selector}',
                       color_discrete_sequence=color_discrete_sequence if color_scheme_name != "Optimal" else None,
                       color='Канал' if color_scheme_name == "Optimal" else None,
                       color_discrete_map=color_discrete_map if color_scheme_name == "Optimal" else None)
        fig_1.update_layout(xaxis_title='Канал', yaxis_title='Количество')
        fig_1.update_xaxes(showticklabels=False)

        fig_2 = px.bar(chanel_post_counts_2, x='Канал', y='Количество',
                       title=f'Распределение упоминаний<br>{brand2} по тональности в {chanel_selector}',
                       color_discrete_sequence=color_discrete_sequence if color_scheme_name != "Optimal" else None,
                       color='Канал' if color_scheme_name == "Optimal" else None,
                       color_discrete_map=color_discrete_map if color_scheme_name == "Optimal" else None)
        fig_2.update_layout(xaxis_title='Канал', yaxis_title='Количество')
        fig_2.update_xaxes(showticklabels=False)

    # Вывод графиков в одной строке
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_1, use_container_width=True)
    with col2:
        st.plotly_chart(fig_2, use_container_width=True)

    # Подсчет количества упоминаний брендов по каналам
    channel_counts_1 = df_1['chanel'].value_counts().reset_index()
    channel_counts_1.columns = ['Канал', f'Упоминания {brand1}']

    channel_counts_2 = df_2['chanel'].value_counts().reset_index()
    channel_counts_2.columns = ['Канал', f'Упоминания {brand2}']

    # Объединение данных об упоминаниях брендов
    combined_channel_counts = channel_counts_1.merge(channel_counts_2, on='Канал', how='outer')

    # Создание графика с использованием Plotly
    if color_scheme_name == "Optimal":
        fig = px.bar(combined_channel_counts, x='Канал',
                     y=[f'Упоминания {brand1}', f'Упоминания {brand2}'],
                     title=f'Соотношение упоминаний ключей по каналам',
                     color_discrete_map=color_scheme_value)
    else:
        fig = px.bar(combined_channel_counts, x='Канал',
                     y=[f'Упоминания {brand1}', f'Упоминания {brand2}'],
                     title=f'Соотношение упоминаний ключей по каналам',
                     color_discrete_sequence=color_scheme_value)

    # Вывод графика в Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Подсчет тональности отзывов для каждого бренда
    sentiment_counts_1 = df_1['predicted_class'].value_counts().reset_index()
    sentiment_counts_1.columns = ['Тональность', f'Количество {brand1}']

    sentiment_counts_2 = df_2['predicted_class'].value_counts().reset_index()
    sentiment_counts_2.columns = ['Тональность', f'Количество {brand2}']

    # Создание круговых графиков с использованием Plotly
    if color_scheme_name == "Optimal":
        fig1 = px.pie(sentiment_counts_1, values=f'Количество {brand1}', names='Тональность',
                      title=f'Тональность всех постов для {brand1}',
                      color='Тональность',
                      color_discrete_map=color_scheme_value)

        fig2 = px.pie(sentiment_counts_2, values=f'Количество {brand2}', names='Тональность',
                      title=f'Тональность всех постов для {brand2}',
                      color='Тональность',
                      color_discrete_map=color_scheme_value)
    else:
        fig1 = px.pie(sentiment_counts_1, values=f'Количество {brand1}', names='Тональность',
                      title=f'Тональность всех постов для {brand1}',
                      color_discrete_sequence=color_scheme_value)

        fig2 = px.pie(sentiment_counts_2, values=f'Количество {brand2}', names='Тональность',
                      title=f'Тональность всех постов для {brand2}',
                      color_discrete_sequence=color_scheme_value)

    # Вывод графиков в одной строке
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)


def keyword_analysis(df: pd.DataFrame, color_scheme_name: str, color_scheme_value):
    st.subheader("""
    Анализ по ключевому слову
    """, divider='blue')

    user_input = st.text_input("**Введите ключевое слово**", "Utair")

    # Фильтрация DataFrame по упоминаниям ключа
    df = df[df['text'].str.contains(user_input, case=False, na=False)]

    @st.cache_data
    def convert_df(df):
        return df.iloc[:, :-1].to_csv(index=False).encode('utf-8')

    csv = convert_df(df)

    st.download_button(
        label="Выгрузка данных по ключу в формате CSV",
        data=csv,
        file_name='chanel_posts_proceed_filtered_keyword.csv',
        mime='text/csv',
    )

    chart_type = st.selectbox(
        "Выберите тип графика распределения тональности упоминаний",
        ["Столбчатая диаграмма", "Круговая диаграмма"]
    )

    class_counts = df['predicted_class'].value_counts().reset_index()
    class_counts.columns = ['Класс', 'Количество']

    # Построение выбранного типа графика
    if chart_type == "Столбчатая диаграмма":
        if color_scheme_name == "Optimal":
            fig = px.bar(class_counts, x='Класс', y='Количество',
                         title=f'Распределение тональности упоминаний для {user_input} за весь период',
                         color='Класс', color_discrete_map=color_scheme_value)
        else:
            fig = px.bar(class_counts, x='Класс', y='Количество',
                         title=f'Распределение тональности упоминаний для {user_input} за весь период',
                         color='Класс', color_discrete_sequence=color_scheme_value)
    elif chart_type == "Круговая диаграмма":
        if color_scheme_name == "Optimal":
            fig = px.pie(class_counts, names='Класс', values='Количество',
                         title=f'Распределение тональности упоминаний для {user_input} за весь период',
                         color='Класс', color_discrete_map=color_scheme_value)
        else:
            fig = px.pie(class_counts, names='Класс', values='Количество',
                         title=f'Распределение тональности упоминаний для {user_input} за весь период',
                         color='Класс', color_discrete_sequence=color_scheme_value)
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig)

    # Выбор периода построения графика

    periods_dict = {'День': 'D', 'Неделя': 'W', 'Месяц': 'M', 'Год': 'Y'}

    period = st.selectbox(
        "Выберите период для построения графика:",
        periods_dict.keys(),
        index=2
    )
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Создание графика
    periodic_sentiment_count_all = df.groupby(
        [pd.Grouper(key='date', freq=periods_dict[period]), 'predicted_class']).size().reset_index(name='counts')

    periods_shorts = {'День': 'дням', 'Неделя': 'неделям', 'Месяц': 'месяцам', 'Год': 'годам'}

    # Создание графика с использованием Plotly
    if color_scheme_name == "Optimal":
        fig = px.bar(periodic_sentiment_count_all, x='date', y='counts', color='predicted_class',
                     title=f'Соотношение всех упоминаний по тональности по {periods_shorts[period]}',
                     color_discrete_map=color_scheme_value)
    else:
        fig = px.bar(periodic_sentiment_count_all, x='date', y='counts', color='predicted_class',
                     title=f'Соотношение всех упоминаний по тональности по {periods_shorts[period]}',
                     color_discrete_sequence=color_scheme_value)
    fig.update_xaxes(title_text='Дата поста')
    fig.update_yaxes(title_text='Количество')
    fig.update_layout(legend_title_text='Тональность')
    st.plotly_chart(fig)

    if color_scheme_name == "Optimal":
        fig = px.line(periodic_sentiment_count_all, x='date', y='counts', color='predicted_class',
                      title=f'Динамика количества упоминаний по тональности по {periods_shorts[period]}',
                      color_discrete_map=color_scheme_value)
    else:
        fig = px.line(periodic_sentiment_count_all, x='date', y='counts', color='predicted_class',
                      title=f'Динамика количества упоминаний по тональности по {periods_shorts[period]}',
                      color_discrete_sequence=color_scheme_value)
    fig.update_xaxes(title_text='Дата поста')
    fig.update_yaxes(title_text='Количество')
    fig.update_layout(legend_title_text='Тональность')
    st.plotly_chart(fig)

    chanels_list = ['Все каналы']
    chanels_list.extend(df['chanel'].unique().tolist())
    chanel_selector = st.selectbox('Выберите канал для анализа', chanels_list, index=0)
    if chanel_selector == 'Все каналы':
        # Подсчет количества упоминаний ключей по каналам
        channel_counts = df['chanel'].value_counts().reset_index()
        channel_counts.columns = ['Канал', 'Количество']

        # Создание графиков с использованием Plotly
        if color_scheme_name == "Optimal":
            color_discrete_map = color_scheme_value
        else:
            color_discrete_sequence = color_scheme_value

        fig = px.bar(channel_counts, x='Канал', y='Количество',
                     title=f'Распределение упоминаний<br>{user_input} по каналам',
                     color_discrete_sequence=color_discrete_sequence if color_scheme_name != "Optimal" else None,
                     color='Канал' if color_scheme_name == "Optimal" else None,
                     color_discrete_map=color_discrete_map if color_scheme_name == "Optimal" else None)
        fig.update_layout(xaxis_title='Канал', yaxis_title='Количество')
        fig.update_xaxes(showticklabels=False)
    else:
        # Подсчет количества упоминаний ключей по каналам
        chanel_post_counts = df[df['chanel'] == chanel_selector]['predicted_class'].value_counts().reset_index()
        chanel_post_counts.columns = ['Канал', 'Количество']

        # Создание графиков с использованием Plotly
        if color_scheme_name == "Optimal":
            color_discrete_map = color_scheme_value
        else:
            color_discrete_sequence = color_scheme_value

        fig = px.bar(chanel_post_counts, x='Канал', y='Количество',
                     title=f'Распределение упоминаний<br>{user_input} по тональности в {chanel_selector}',
                     color_discrete_sequence=color_discrete_sequence if color_scheme_name != "Optimal" else None,
                     color='Канал' if color_scheme_name == "Optimal" else None,
                     color_discrete_map=color_discrete_map if color_scheme_name == "Optimal" else None)
        fig.update_layout(xaxis_title='Канал', yaxis_title='Количество', showlegend=False)

    # Вывод графика в Streamlit
    st.plotly_chart(fig, use_container_width=True)
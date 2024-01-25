import datetime
import os
import streamlit as st
import uuid
import logging
import pandas as pd
import time


# СОЗДАНИЕ ЗАПРОСА
def create_new_task1(yet_another_row):
    try:
        # Создание запроса для обработки в папке ldb/taks
        df = pd.DataFrame({'uid': [yet_another_row[0]],
                           'starttime': [yet_another_row[1]],
                           'sources': [yet_another_row[2]],
                           'tasktype': [yet_another_row[3]],
                           'targets': [yet_another_row[4]],
                           'test': [yet_another_row[5]],
                           'parsing': [yet_another_row[6]],
                           'semantic': [yet_another_row[7]],
                           'summ': [yet_another_row[8]],
                           'status': [yet_another_row[9]],
                           'endtime': [yet_another_row[10]]})
        save_path = f'ldb/tasks/{yet_another_row[0]}.csv'
        df.to_csv(save_path)

        st.toast(f'''
    🟡 Создан task {yet_another_row[3]}
    {yet_another_row[0]} в {yet_another_row[1]}''')
        logging.info(f'Creating new task. Success.')
    except Exception as e:
        print(f'Creating new task. Failed. {e}')

    # Запись каналов в файл


def create_task2(yet_another_row):
    try:
        with open("ldb/tasks/online/online.txt", "w") as f:
            row = yet_another_row[2]
            f.write(row)

    except Exception as e:
        logging.error(f'Creating new task. Failed. {e}')


# ЧТЕНИЕ ФАЙЛОВ И ИХ СТАТУСВ
def list_files(folder_path: str):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return files


def load_data(folder_paths):
    data = []
    for folder_path in folder_paths:
        files = list_files(folder_path)
        for file in files:
            data.append({'uid': file.split('.')[0],
                         'type': 'retro',
                         'folder': folder_path.split('/')[-1]
                         })
    df = pd.DataFrame(data)
    unique_uids = df['uid'].unique()
    # types = df[df['uid'] == unique_uids].replace('tasks', 'retro')['folder']
    types = [0] * unique_uids
    # Преобразуем время в более читаемый формат
    new_data = {'uid': unique_uids, 'type': types, 'time': 0, 'parsing': [0] * len(unique_uids),
                'semantic': [0] * len(unique_uids), 'summarization': 0}

    for uid in unique_uids:
        idx = new_data['uid'].tolist().index(uid)
        parsing_done = (df['folder'] == 'posts') & (df['uid'] == uid)
        parsing_error = (df['folder'] == 'errors') & (df['uid'] == uid)
        new_data['parsing'][idx] = '🟢' if parsing_done.any() else ('🔴' if parsing_error.any() else '🟡')

        semantic_done = (df['folder'] == 'SAcompleted') & (df['uid'] == uid)
        semantic_error = (df['folder'] == 'errors') & (df['uid'] == uid)
        new_data['semantic'][idx] = '🟢' if semantic_done.any() else ('🔴' if semantic_error.any() else '🟡')

    new_df = pd.DataFrame(new_data)
    return pd.DataFrame(new_df)


st.title("Streamlit App")

col1, col2 = st.columns([1, 4])
with col1:
    tasktype = st.radio('Тип запроса:', ['retro', 'online'])
with col2:
    # Ввод переменных от пользователя
    chanel_user_input = st.text_input("Введите каналы для отслеживания", "")

# Кнопка для запуска внешнего скрипта
create_task = st.button("Запрос статистики по каналам", key='create_task')
if create_task:
    task = (
    uuid.uuid4(), datetime.time(), chanel_user_input, tasktype, 'targets', 'test', 0, 0, 0, 'in progress', 'endtime')
    print(tasktype)
    if tasktype == 'retro':
        create_new_task1(task)
    else:
        create_task2(task)

folder_paths = ['ldb/tasks', 'ldb/tasks/completed', 'ldb/tasks/errors', 'ldb/posts', 'ldb/SAcompleted',
                'ldb/tasks/online']
try:
    df = load_data(folder_paths)
except:
    df = pd.DataFrame()

# ВЫВОД СТАТУС-БАРА ПО ЗАПРОСАМ
with st.expander("task-status"):
    col3, col4, col5 = st.columns([1, 3.5, 1])
    with col3:
        st.write('🟢 -- done')
        st.write('🟡 -- in queue')
        st.write('🔴 -- error')
    with col4:
        st.write('''''')
        st.write('''
░▒█▀▀▀░█░░█▀▀▄░█▀▀▄░█░▒█░█▀▀
░▒█▀▀▀░█░░█▀▀▄░█▄▄▀░█░▒█░▀▀▄
░▒█▄▄▄░▀▀░▀▀▀▀░▀░▀▀░░▀▀▀░▀▀▀
''', unsafe_allow_html=True)
    with col5:
        refresh_button = st.button("refresh list", key='refresh_button')
        if refresh_button:
            df = load_data(folder_paths)

        delete_button = st.button('delete all', key='delete_all')
    #    if delete_all:
    #        delete_all(folder_paths)

    st.table(df)

import logging
import asyncio
import os
import time
import shutil
import pandas as pd
import csv

from funcs.data_processing import get_data
from funcs.parsing import parsing
from pathlib import Path

#путь для логирования
log_file_path = 'ldb/app.log'
logging.basicConfig(level=logging.INFO, filename=log_file_path, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

while True:
    # подсчёт файлов в tasks
    files = [f for f in os.listdir('ldb/tasks') if os.path.isfile(os.path.join('ldb/tasks', f))]

    if not files:
        time.sleep(5)
        continue

    oldest_file = min(files, key=lambda x: os.path.getctime(os.path.join('ldb/tasks', x)))
    file_path = Path('ldb/tasks') / oldest_file
    df = pd.read_csv(file_path)
    task = df.iloc[0].tolist()

    logging.info(f'Start parsing, task uid: {task[1]}')
    # попытка парсинга
    try:
        start_time = time.time()

        asyncio.run(parsing(task[3], task[1]))

        end_time = time.time()
        logging.info(f'Parsing complete. {task[1]}\nTotal time:{end_time-start_time}')
    except Exception as e:
        logging.error(f'ERROR in parsing: {e}')

        shutil.move(file_path, Path('ldb/tasks/errors') / oldest_file)

        logging.info(f'{task[1]} task was crashed in parsing')
        continue

    logging.info(f'Start semantic analysis, task uid: {task[1]}')
    # попытка семантического анализа
    try:
        start_time = time.time()

        get_data(f'ldb/posts/{task[1]}.csv')
        end_time = time.time()
        logging.info(f'Semantic analysis complete. {task[1]}\nTotal time:{end_time-start_time}')
        shutil.move(file_path, Path('ldb/tasks/completed') / oldest_file)
    except Exception as e:
        logging.error(f'ERROR in semantic analysis: {e}')
        shutil.move(file_path, Path('ldb/tasks/errors') / oldest_file)
        logging.info(f'{task[1]} was crashed in semantic analysis.')
        continue

    continue

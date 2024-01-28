import time
import os
import pandas
import csv
import datetime

import pandas as pd

from funcs.data_processing import get_data
from telethon import TelegramClient, events


config = {
    'api_id': 00000000,
    'api_hash': 'YOUR API HASH'
}

file_path = 'ldb/tasks/online/online.txt'
# Лист каналов для отслеживания
telegram_channels = {
    'from': [
        {'channel_id': '@SOMETESTCHANNEL'}
    ],
}
# Чтение списка запросов на отслеживание и добавление в список
while True:
    with open(file_path, 'r') as f:
        for line in f:
            line = line.split()
            for id in line:
                telegram_channels['from'].append({'channel_id' : id})

    listen_channels = [ch['channel_id'] for ch in telegram_channels["from"]]

    with TelegramClient('telegram_listen_news_app_me', config['api_id'], config['api_hash']) as client:
        @client.on(events.NewMessage(chats=listen_channels))
        async def get_message_handler(event):
            channel = event.message.chat.username
            text = event.raw_text
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            postlink = 'https://t.me/' + channel[1:] + '/' + str(event.message.chat.id)

            with open('ldb/posts/online.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['chanel', 'link', 'date', 'text'])
                writer.writerow([channel, postlink, current_time, text])

            try:
                get_data('ldb/posts/online.csv', modelpath='models', save_file='ldb/data/data.csv', ttype='online')

            except Exception as e:
                print(e)

        client.run_until_disconnected()

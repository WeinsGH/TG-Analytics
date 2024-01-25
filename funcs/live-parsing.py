import time
import os
import pandas
import csv
import datetime

import pandas as pd

from data_processing import get_data, summarization
from telethon import TelegramClient, events



api_id = 26146247
api_hash = 'd0a288f1b3244ff25756d1f79c83f878'


config = {
    'api_id': 26146247,
    'api_hash': 'd0a288f1b3244ff25756d1f79c83f878'
}

file_path = '../ldb/tasks/online/online.txt'
telegram_channels = {
    'from': [
        {'channel_id': '@yurydud'}
    ],
}
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

            with open('../ldb/posts/online.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['chanel', 'date', 'link', 'text'])
                writer.writerow([channel, current_time, postlink, text])

            try:
                get_data('../ldb/posts/online.csv', modelpath='../models', save_file='../ldb/data/data.csv', ttype='online')

            except Exception as e:
                print(e)

        client.run_until_disconnected()

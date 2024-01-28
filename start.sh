#!/usr/bin/env bash

# Обновление списка пакетов
sudo apt update

# Установка Python 3.10 и обновление pip
sudo apt-get install python3.10
sudo apt update
sudo apt install python3-pip
sudo apt install python3-distutils
sudo apt install screen

# Установка OCR модели
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-rus
pip install pytesseract

# Установка необходимых пакетов
pip install streamlit telethon asyncio pandas torch transformers nltk wordcloud st_pages bert-score

echo 'trying to install models..'

# Создание директорий
mkdir "models"
mkdir "models/local_tokenizer"
mkdir "summary"
mkdir "summary/tokenizer"

# Запуск скрипта get_models.py
python3 download_models.py

echo 'start.sh completed'
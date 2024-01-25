import pandas as pd
import streamlit as st
import pytesseract
from PIL import Image
import io
import time
import torch
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification, MBartForConditionalGeneration, MBartTokenizer

def get_data(text):
    def clean_text(text):
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # Удаление ссылок
        text = re.sub(r'<.*?>', ' ', text)  # Удаление HTML тегов
        return text

    text = clean_text(text)
    # так как предыдущими действиями мы скорее всего удалили только ссылки, но оставили обёртки, удаляем обёртки
    text.replace('<a href="', ' ')

    # ОПРЕДЕЛЕНИЕ ТОНАЛЬНОСТИ

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Загрузка модели и токенизатора "MonoHime/rubert-base-cased-sentiment-new"
    model = AutoModelForSequenceClassification.from_pretrained('../models').to(device)
    tokenizer = AutoTokenizer.from_pretrained(f'../models/local_tokenizer')

    # Функция для подготовки пакетов данных
    def preprocess_texts(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Инициализация массива для хранения предсказаний
    predicted_classes = torch.tensor([], dtype=torch.int64, device=device)

    # Обработка текстов пакетами и получение предсказаний
    encoded_input = preprocess_texts(text)
        # Убедитесь, что данные находятся на GPU
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        batch_predictions = model(**encoded_input)
        batch_predicted_classes = torch.argmax(batch_predictions.logits, dim=1)
        predicted_classes = torch.cat((predicted_classes, batch_predicted_classes), 0)

    # Добавление предсказаний в DataFrame
    predict = predicted_classes.tolist()
    return predict



def summarization(text):
    # Проверка доступности CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Загрузка модели и токенизатора
    model_path = '../models/summary'
    tokenizer = MBartTokenizer.from_pretrained(f'{model_path}/tokenizer')
    model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)

    # Функция для деления текста пополам
    def split_text(text, word_limit=600):
        words = text.split()
        if len(words) <= word_limit:
            return [text]
        mid = len(words) // 2
        return [' '.join(words[:mid]), ' '.join(words[mid:])]

    # Функция для суммаризации текста
    def summarize_text(text):
        # Разделение текста на части, если он слишком длинный
        text_parts = split_text(text)
        summary = ''
        for part in text_parts:
            inputs = tokenizer([part], max_length=1024, return_tensors="pt", truncation=True)
            inputs = inputs.to(device)
            summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
            summary += tokenizer.decode(summary_ids[0], skip_special_tokens=True) + ' '
        return summary.strip()

    # Применение суммаризации к тексту
    return summarize_text(text)


st.title('OCR с использованием Tesseract')

uploaded_file = st.file_uploader("Загрузите изображение с текстом на русском языке", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Конвертация в PNG
    image_png = image.convert('RGB')
    with io.BytesIO() as output:
        image_png.save(output, format="PNG")
        png_data = output.getvalue()

    st.image(image, caption='Загруженное изображение', width=200)
    st.write("")
    source = st.text_input('Введите источник')
    if st.button('Распознать текст'):
        st.write("Распознанный текст:")
        try:
            # Преобразование изображения в текст
            text = pytesseract.image_to_string(Image.open(io.BytesIO(png_data)), lang='rus')
            st.write(text)

            post_date = time.localtime()
            df = pd.DataFrame({'chanel':'image', 'link':source,'date':post_date, 'text':text, 'predicted_class':get_data(text), 'summary': summarization(text)})
            df.to_csv('../ldb/data/data.csv', mode='a', header=False, index=False)
        except Exception as e:
            st.error(f'Ошибка при распознавании текста: {e}')


import pandas as pd
import re
import torch
from bert_score import score

from transformers import AutoTokenizer, AutoModelForSequenceClassification, MBartForConditionalGeneration, MBartTokenizer

#Словарь стоп-слов
minus_words = ["наркотик", "наркоман", "наркозависимый", "наркология", "наркодилер",
               "нарконос", "наркобарон", "оружие", "пистолет", "автомат", "ружье",
               "граната", "миномет", "винтовка", "пулемет", "ракета", "бомба",
               "взрывчатка", "карабин", "снайпер", "боеприпасы", "наркоманка", "хамас", "украина", "палестина", "талибан"]


# СУММАРИЗАЦИЯ
def summarization(text):
    # Проверка доступности CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Загрузка модели и токенизатора
    model_path2 = 'models/summary'
    tokenizer2 = MBartTokenizer.from_pretrained(f'{model_path2}/tokenizer')
    model2 = MBartForConditionalGeneration.from_pretrained(model_path2).to(device)

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
            inputs = tokenizer2([part], max_length=1024, return_tensors="pt", truncation=True)
            inputs = inputs.to(device)
            summary_ids = model2.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
            summary += tokenizer2.decode(summary_ids[0], skip_special_tokens=True) + ' '
        return summary.strip()

    # Применение суммаризации к тексту
    return summarize_text(text)


def get_data(path, modelpath='models', save_file='ldb/data/data.csv', ttype='retro'):
    df = pd.read_csv(path)
    uid = path.split('/')[-1].split('.')[0]    # Читаем uid, оптимальнее не придумал
    df_minus = pd.DataFrame()
    for word in minus_words:
        df_temp = df[df['text'].str.contains(word, case=False, na=False)]
        df_minus = pd.concat([df_minus, df_temp])
    # Удаление дубликатов строк, если они есть
    df_minus = df_minus.drop_duplicates()
    # Удаление строк, содержащих любое из минус-слов
    for word in minus_words:
        df = df[~df['text'].str.contains(word, case=False, na=False)]
    def clean_text(text):
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # Удаление ссылок
        text = re.sub(r'<.*?>', ' ', text)  # Удаление HTML тегов
        return text
    df['text'] = df['text'].apply(clean_text)
    # так как предыдущими действиями мы скорее всего удалили только ссылки, но оставили обёртки, удаляем обёртки
    df['text'] = df['text'].str.replace('<a href="', ' ')
    df = df.dropna(subset=['text'])

    # ОПРЕДЕЛЕНИЕ ТОНАЛЬНОСТИ

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Загрузка модели и токенизатора "MonoHime/rubert-base-cased-sentiment-new"
    model = AutoModelForSequenceClassification.from_pretrained(modelpath).to(device)
    tokenizer = AutoTokenizer.from_pretrained(f'{modelpath}/local_tokenizer')

    # Установка размера пакета (batch size)
    batch_size = 16

    # Функция для подготовки пакетов данных
    def preprocess_texts(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")


    # Инициализация массива для хранения предсказаний
    predicted_classes = torch.tensor([], dtype=torch.int64, device=device)

    # Обработка текстов пакетами и получение предсказаний
    for i in range(0, len(df['text']), batch_size):
        batch_texts = df['text'][i:i + batch_size].tolist()
        encoded_input = preprocess_texts(batch_texts)
        # Убедитесь, что данные находятся на GPU
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            batch_predictions = model(**encoded_input)
            batch_predicted_classes = torch.argmax(batch_predictions.logits, dim=1)
            predicted_classes = torch.cat((predicted_classes, batch_predicted_classes), 0)

    # Добавление предсказаний в DataFrame
    df['predicted_class'] = predicted_classes.tolist()


    if ttype == 'retro':
        df['summary'] = df['text'].apply(summarization)
        df['bert_score_precision'] = [None]*len(df)
        df['bert_score_recall'] = [None]*len(df)
        df['bert_score_f1'] = [None]*len(df)
        df.to_csv(f'ldb/SAcompleted/{uid}.csv', index=False)
    else:
        df['summary'] = summarization(df['text'][0])
        df['bert_score_precision'] = [None]*len(df)
        df['bert_score_recall'] = [None]*len(df)
        df['bert_score_f1'] = [None]*len(df)
    df.to_csv(save_file, mode='a', header=False, index=False)




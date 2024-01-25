from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "MonoHime/rubert-base-cased-sentiment-new"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Укажите путь каталога, в который хотите сохранить веса и файлы
save_directory = 'ldb/models/summary'

# Сохранение весов модели и файлов токенизатора
model.save_pretrained(save_directory)
tokenizer.save_pretrained(f'{save_directory}/tokenizer')

print(f"Модель суммаризации установлена")

model_name = "papluca/xlm-roberta-base-language-detection"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Укажите путь каталога, в который хотите сохранить веса и файлы
save_directory = "/models"

# Сохранение весов модели и файлов токенизатора
model.save_pretrained(save_directory)
tokenizer.save_pretrained(f'{save_directory}/tokenizer')

print(f"Модель и токенизатор сохранены в {save_directory}")

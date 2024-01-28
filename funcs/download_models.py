from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "MonoHime/rubert-base-cased-sentiment-new"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Укажите путь каталога, в который хотите сохранить веса и файлы
save_directory = 'models'

# Сохранение весов модели и файлов токенизатора
model.save_pretrained(save_directory)
tokenizer.save_pretrained(f'{save_directory}/local_tokenizer')

print(f"Модель 1 установлена")

model_name = "IlyaGusev/mbart_ru_sum_gazeta"
model2 = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer2 = MBartTokenizer.from_pretrained(model_name)

# Укажите путь каталога, в который хотите сохранить веса и файлы
save_directory = "/models/summary"

# Сохранение весов модели и файлов токенизатора
model2.save_pretrained(save_directory)
tokenizer2.save_pretrained(f'{save_directory}/tokenizer')

print(f"Модель 2 и токенизатор сохранены в {save_directory}")

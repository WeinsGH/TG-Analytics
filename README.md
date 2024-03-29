# TG-Analitics

---
![pytorch](https://img.shields.io/badge/pytorch-Used-green)     ![telethon](https://img.shields.io/badge/telethon-Used-blue)  ![huggingface](https://img.shields.io/badge/huggingface-Used-yellow) ![OCR](https://img.shields.io/badge/tesseract-Used-lightgrey) ![st](https://img.shields.io/badge/streamlit-Used-red)

---
<p align="center">
  <img src="https://github.com/WeinsGH/TG-Analytics/assets/109025285/65cb5519-d3a9-487c-a2a6-205c82aa87bb">
</p>

**TG-Analitics** - это платформа для мониторинга и анализа контента telegram-каналов с помощью NLP-моделей, которая помогает понимать, оптимизировать бренд и медиа-стратегии. А ещё сравнивать показатели брендов. Вы можете отслеживать как уже существующие упоминания (посты) в режиме ретроспективы среди всех, интересующих вас источниках, так и подключить каналы для мониторинга всех последующих упоминаний.

### Как это работает

Вы можете создать произвольное число запросов, вписав интересующие вас каналы через пробел. Какое-то время понадобится на сбор и анализ языковыми моделями постов (по умолчанию 500 из каждого источника). После чего вы сможете выгрузить аналитику по конкретному запросу или по общей БД запросов. Вы можете использовать также image-to-text модель **tesseract OCR** для распознавания русского текста и добавление его в базу для аналитики.
<p align="center">
  <img src="https://github.com/WeinsGH/TG-Analytics/assets/109025285/f3dd98f5-6ab6-4d03-a375-7f15aaaeda22">
</p>

На странице с аналитикой вы можете выгрузить сводный анализ по собранной базе данных, анализ данных по конкретному ключевому слову/бренду, либо сравнительный анализ 2-х брендов на основе упоминаемости и оценке тональности упоминаний. Оценка тональности осуществляется [моделью семейства BERT с hugging face](MonoHime/rubert-base-cased-sentiment-new).
<p align="center">
  <img src="https://github.com/WeinsGH/TG-Analytics/assets/109025285/78a3232e-cf8d-4e57-a2c3-a18f74fd7289">
</p>

Вы также можете осуществить выгрузку список кратких саммари всех упоминаний за любой день. Суммаризация осуществляется также моделью семейства BERT - [MBARTRuSumGazeta](https://huggingface.co/docs/transformers/model_doc/mbart). В демонстрационных целях также выводятся bert-score метрики, если суммаризация была осуществлена (посты длиной более 50 слов)
<p align="center">
  <img src="https://github.com/WeinsGH/TG-Analytics/assets/109025285/5fad9c69-2c90-4d00-bf2d-1f753601dfb6">
</p>

---

### **Установка (Ubuntu)**
1. Клонируйте этот репозиторий в необходимую директорию. Мы рекомендуем использовать машину с gpu для ускорения обработки данных и возможности применения модели суммаризации.
2. Запустите bash-скрипт ```start.sh```. Скрипт должен установить все зависимости, создать необходимые директории, а также скачать модели и токенайзеры для их локального использования. В случае возникновения проблем с установкой конкретных зависимостей или моделей, вам придётся устанавливать их в индивидуальном порядке в соответствии с документацией.
3. Откройте config.py. Вы должны вписать сюда данные от вашего телеграм-приложения (api_hash, api_id и номер телефона) для работы парсера. Получить их можно, предварительно создав приложение на [my.telegram.org](my.telegram.org)
4. Запустите файлы 
```streamlit run app.py```
```python3 dispatcher.py```
```python3 funcs/live-parsing.py```
Вы также можете использовать скрипты ```parsing.py``` и ```live-parsing.py``` отдельно, если вам необходимо спарсить/парсить список telegram-каналов соответственно.
5. ✨Magic ✨
6. Откройте стримлит на соответствующем порту localhost. ([по умолчанию: 8501](http://localhost:8501))
7. Готово!

### Установка (Windows):
Давайте как-нибудь сами

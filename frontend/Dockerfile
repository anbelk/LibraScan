FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей
RUN pip install --no-cache-dir gradio requests

# Копируем только нужные файлы
COPY ./app/main_front.py .

EXPOSE 7860

CMD ["python", "main_front.py"]
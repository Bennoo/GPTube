# syntax=docker/dockerfile:1
FROM python:3.10.12-bullseye

# run this before copying requirements for cache efficiency
RUN pip install --upgrade pip
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src ./src

CMD [ "python", "./src/slack_bot.py"]
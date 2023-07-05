# syntax=docker/dockerfile:1
FROM python:3.10.12-bullseye

# run this before copying requirements for cache efficiency
RUN pip install --upgrade pip
COPY .devcontainer/requirements.txt .

RUN pip install -r requirements.txt
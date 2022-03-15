# Простой Flask сервер с ML моделью 

В данном учебном проекте реализован простой сервер на Flask с возможностью обучения модели.

Функционал сервера:
- Запуск из *docker* контейнера
- Возможность выбора базовой модели (Random Forest или Gradient Boosting) и настройки гиперпараметров
- Загрузка датасета 
- Просмотр табличных данных и кривой обучения

[Отчет](./Ensembels.pdf) &emsp; [Скрипты для запуска](./scripts/) &emsp; [Ноутбук с экспериментами](./trees.ipynb)  

## Пример работы сервера

![Server work](./imgs/Flask_example.gif)

##  Запуск сервера 

- Установить `Docker`
- Склонировать репозиторий `$ git clone https://github.com/PavelShtykov/trees_ensembles_prac`
- Собрать сервер `$ bash build.sh`
- Запустить сервер `$ bash run.sh`

Также сервер доступен на [DockerHub](https://hub.docker.com/repository/docker/shtykovpavel/server-test)



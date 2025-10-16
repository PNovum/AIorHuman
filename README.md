# Week 4 

## Проект

Репозиторий реализует полный ML pipeline


MLflow и Model Registry

## Установка и запуск

```bash
# 1.
git clone <repo-url>
cd week3-homework

# 2. 
cp .env.example .env

# 3.
docker compose up -d

# 4. 
curl http://127.0.0.1:5050/version
curl -u admin:adminpassword http://127.0.0.1:8080/docs

# Как поднять и протестировать

Поднять стек
```bash
docker compose up -d --build


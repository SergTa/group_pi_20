# Проектный практикум, 3 семестр Группа 20 (pi)

![Logotype](./src/models/data/ETL.png)
## Описание
Данный проект направлен на разработку алгоритма и программы для оценки технологического риска внедрений релизов программного обеспечения, предназначенного для выполнения процедур трансформации данных (ETL) и расчетов на витринах хранилища данных (DWH). Проект использует риск-ориентированный подход для оптимизации процесса тестирования и снижения затрат.

## Процесс разработки
При разработке архитектуры рассматривалось несколько различных вариантов построения алгоритма и методики для оценки технологического риска внедрений релизов программного обеспечения от использования алгоритма обработки критериев до применения моделей машинного обучения. В конечном счёте было принято решение реализовать `комплексную методику, основывающуюся на многокритериальном подходе.` 

> [!NOTE]
> *Ввиду того, что в брифе оценка важности конкретных критериев не была представлена, было принято решение о вводе `собственных шкал`, описывающих степени влияния отдельно взятых рисков на интегральный риск исходя из логических соображений.*


> [!IMPORTANT]
> *Была создана программа, реализующая алгоритм оценки технологических рисков, которая предусматривает:* 

> 1) сбор и агрегацию данных из `различных источников`: базы данных, историй внедрений, оценок команд и процессов и т.д.
> 2) наличие `интерфейса`, удобного для заполнения ответственными лицами.
> 3) реализацию `алгоритма расчёта риска` – с использованием предобученной модели для классификации риска каждого релиза по заданным критериям (с использованием различных видов классификаторов).

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/ChristopherMcCandless/group_pi_20.git
   ```
2. Установите необходимые зависимости:
    ```bash
    pip install -r requirements.txt
    ```
3. Запустите сервер:
    ```bash
    python src/web/main.py
    ```
4. Откройте файл `src/web/index.html` с помощью браузера


## Использование

### Добавление релиза
    - Перейдите на страницу "Releases" и нажмите кнопку "Add Release".

![Add](./src/models/data/releases_add.jpg)

    - Заполните форму с параметрами релиза и нажмите "Submit".
![Add one](./src/models/data/releases_add_one.jpg)

### Предсказание рисков
    - На странице "Releases" выберите релиз и нажмите кнопку "Select".

![Releases](./src/models/data/releases.jpg)

    - Нажмите кнопку "Calculate Risk" для получения предсказания рисков.
![Calc](./src/models/data/calculate.jpg)

## Структура проекта
```
/group_pi_20
│
├── /src
│   ├── /models          # Модели и скрипты для предсказания рисков
│   ├── /web             # Веб-приложение на Flask
│   └── /data            # Данные для обучения модели
│
├── example.db           # База данных SQLite
├── requirements.txt     # Зависимости проекта
└── README.md            # Этот файл
```

## Состав команды
    Аналитик данных, подготовка датасетов – Кузнецов Иван
    Руководитель проекта - Красильников Михаил
    Full Stack-разработчик – Казанцев Александр
    Тестировщик / резерв-инженер – Табакарь Сергей
    Документалист/технический писатель – Граб Яков 


## Лицензия
    Python Software Foundation License

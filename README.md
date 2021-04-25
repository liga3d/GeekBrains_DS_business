Итоговый проект курса "Машинное обучение в бизнесе"

В данном примере использовался датасет https://www.kaggle.com/blackmoon/russian-language-toxic-comments
Задача: определить является ли комментарий токсичным или нет.
Бинарная классификация

Признак один: сам комментарий на русском языке. 
Преобразования признака: TfIdf

Модели на первом уровне: LogisticRegression + среднее по предсказанным вероятностям всех моделей и стандартное отклонение
Модель на втором уровне: CatBoostClassifier
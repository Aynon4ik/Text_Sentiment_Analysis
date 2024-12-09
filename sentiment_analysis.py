from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

texts = [
    "Мне очень понравился сервис, всё отлично!",
    "Это был худший опыт в моей жизни.",
    "Среднее качество, ничего особенного.",
    "Я в восторге! Всё просто замечательно."
]

print("Анализ тональности текстов:")
for text in texts:
    result = classifier(text)[0]
    print(f"Текст: {text}")
    print(f"Тональность: {result['label']}, вероятность: {result['score']:.2f}\n")

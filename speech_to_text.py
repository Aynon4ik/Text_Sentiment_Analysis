import speech_recognition as sr
from pydub import AudioSegment


def recognize_speech_from_file(audio_file):
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file) as source:
            print("Загрузка аудио...")
            audio_data = recognizer.record(source)

        print("Распознавание речи...")
        text = recognizer.recognize_google(audio_data, language="ru-RU")
        print(f"Распознанный текст: {text}")
        return text
    except sr.UnknownValueError:
        print("Не удалось распознать речь.")
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания: {e}")


def recognize_speech_from_microphone():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Говорите что-нибудь...")
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source)

        print("Распознавание речи...")
        text = recognizer.recognize_google(audio_data, language="ru-RU")
        print(f"Распознанный текст: {text}")
        return text
    except sr.UnknownValueError:
        print("Не удалось распознать речь.")
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания: {e}")


if __name__ == "__main__":
    print("Выберите режим работы:")
    print("1 - Распознавание речи из аудиофайла")
    print("2 - Распознавание речи с микрофона")

    choice = input("Введите 1 или 2: ").strip()

    if choice == "1":
        audio_path = input("Введите путь к аудиофайлу: ").strip()
        recognize_speech_from_file(audio_path)
    elif choice == "2":
        recognize_speech_from_microphone()
    else:
        print("Некорректный выбор. Завершение программы.")

import cv2
import os
import time
import numpy as np
import re
from cv2.face import LBPHFaceRecognizer_create

# Создание директории для сохранения снимков

model = 0

cwd = os.getcwd()

while True:
    model = input("Выберите действие:\n"
    "1. Настроить модель (при первом запуске без настройки проект работать не будет)\n"
    "2. Просканировать лицо\n\nОтвет: ")
    match model:
        case "1":
            while True:
                face_id = input("Введите имя пользователя (только английские буквы): ")
                if not re.match("^[a-zA-Z]+$", face_id):
                    print("Ошибка: используйте только английские буквы, без цифр и специальных символов")
                    continue
                break
                
            answer = input('Откройте камеру и подготовьтесь к сканированию лица.\nЕсли вы готовы - введите "Y", если отказываетесь - введите N: ').lower()
            if answer == "y":
                face_dir = os.path.join(cwd, "dataset" + f"_{face_id}")
                if not os.path.exists(face_dir):
                    os.makedirs(face_dir)
                    print(f"Создана директория {face_dir}")

                cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")



                print("\n [INFO] Захват лица. Смотрите в камеру и ждите...")
                count = 0

                def show_position_message(count):
                    if count == 0:
                        print("\nСмотрите прямо в камеру")
                    elif count == 40:
                        print("\nМедленно поверните голову влево")
                    elif count == 80:
                        print("\nМедленно поверните голову вправо")
                    elif count == 120:
                        print("\nМедленно поднимите голову вверх")
                    elif count == 160:
                        print("\nМедленно опустите голову вниз")

                show_position_message(0)
                while True:
                    ret, img = cam.read()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        count += 1
                        # Сохраняем лицо
                        img_path = f"{face_dir}/User.{face_id}.{count}.jpg"
                        cv2.imwrite(img_path, gray[y:y+h, x:x+w])
                        print(f"Сохранено изображение: {img_path}")
                        
                        # Показываем сообщение о смене позиции
                        show_position_message(count)
                        
                    # Добавляем текст-подсказку на изображение
                    position_text = ""
                    if count < 40:
                        position_text = "Смотрите прямо"
                    elif count < 80:
                        position_text = "Поворот влево"
                    elif count < 120:
                        position_text = "Поворот вправо"
                    elif count < 160:
                        position_text = "Голова вверх"
                    else:
                        position_text = "Голова вниз"
                    
                    cv2.putText(img, position_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"Фото: {count}/200", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('image', img)
                    k = cv2.waitKey(100) & 0xff  # 'ESC'
                    if k == 27:
                        break
                    elif count >= 200:  # Если сохранили 200 изображений, выход.
                        break
                    
                print("\n Сканирование завершено\n\n\n")
                cam.release()
                cv2.destroyAllWindows()

                path = os.path.join(cwd, "dataset" + f"_{face_id}")  # папка с набором тренировочных фото
                recognizer = LBPHFaceRecognizer_create()


                # Функция чтения изображений из папки с тренировочными фото
                def getImagesAndLabels(path):
                    # Создаем список файлов в папке patch
                    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
                    face = []  # тут храним массив картинок
                    ids = []  # храним id лица
                    for imagePath in imagePaths:
                        img = cv2.imread(imagePath)
                        # Переводим изображение, тренер принимает изображения
                        # в оттенках серого
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        face.append(img)  # записываем тренировочное фото в массив
                        # Получаем id фото из его названия
                        id = int(os.path.split(imagePath)[-1].split(".")[2])
                        ids.append(id)  # записываем id тренировочного фото в массив
                    return face, ids


                # Чтение тренировочного набора фотографий из папки path
                faces, ids = getImagesAndLabels(path)
                # Тренируем модель распознавания
                recognizer.train(faces, np.array(ids))
                # Сохраняем результат тренировки

                model_dir = os.path.join(cwd, face_id + "_model")

                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                    print(f"Создана директория {model_dir}")

                recognizer.write(f'{model_dir}/face_{face_id}.yml')

                recognizer = LBPHFaceRecognizer_create()
                recognizer.read(f'{model_dir}/face_{face_id}.yml')

                faceCascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


            elif answer == 'n':
                print("Удачи в следующий раз...")
                print("\033[H\033[J", end="")
                time.sleep(3)

        case "2":
                print("Идет сканирование, для закрытия нажмите на ESC")
                # Тип шрифта
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Список имен для id
                names = ['None', face_id]

                cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                cam.set(3, 640)  # размер видеокадра - ширина
                cam.set(4, 480)  # размер видеокадра - высота

                # Добавляем переменную для отслеживания времени
                last_print_time = time.time()

                while True:
                    ret, img = cam.read()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2,
                        minNeighbors=5, minSize=(10, 10),)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                        # Проверяем, что лицо распознано
                        confidence_value = 100 - confidence
                        if confidence_value >= 60:  # Если уверенность больше 60%
                            id_obj = names[1]
                            confidence_str = f"{confidence_value:.2f}%"
                            
                            # Проверяем, прошло ли 10 секунд с последнего вывода
                            current_time = time.time()
                            if current_time - last_print_time >= 10:
                                print(f"На фото обнаружен: {id_obj} (уверенность: {confidence_str})")
                                last_print_time = current_time
                        else:
                            id_obj = names[0]
                            confidence_str = f"{confidence_value:.2f}%"

                        cv2.putText(img, str(id_obj), (x + 5, y - 5),
                                    font, 1, (255, 255, 255), 2)
                        cv2.putText(img, confidence_str, (x + 5, y + h - 5),
                                    font, 1, (255, 255, 0), 1)

                    cv2.imshow('camera', img)

                    k = cv2.waitKey(10) & 0xff  # 'ESC' для Выхода
                    if k == 27:
                        break
                cam.release()
                cv2.destroyAllWindows()
                
                print("Вы ввели неверное значение. Попробуйте еще раз через 3 секунды...")
                print("\033[H\033[J", end="")
                time.sleep(3)
        case _:
            print("Вы ввели неверное значение. Попробуйте еще раз через 3 секунды...")
            print("\033[H\033[J", end="")
            time.sleep(3)
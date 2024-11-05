import os
import time
from datetime import datetime, timedelta
import re
import cv2
import torch
import numpy as np
import mysql.connector
from contextlib import closing
import json
from plate_recognition.model.lpr_net import build_lprnet
from plate_recognition.process_plate import rec_plate, CHARS
from vehicle_detection.detect_car_YOLO import ObjectDetection
from plate_matching import match_car
import config


# Подключение к базе данных MySQL
def connect_to_db():
    try:
        conn = mysql.connector.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME
        )
        print("Соединение с базой данных MySQL успешно установлено.")
        return conn
    except mysql.connector.Error as e:
        print(f"Ошибка при подключении к базе данных: {e}")
        return None

import serial

SERIAL_PORT = 'COM3'  # Замените на нужный порт
BAUD_RATE = 115200
RELAY_COMMAND = "OPEN_RELAY"

def send_command_to_controller(command):
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            ser.write((command + '\n').encode())
            print(f"Команда {command} отправлена.")
    except serial.SerialException as e:
        print(f"Ошибка последовательного порта: {e}")

# Проверка номера в базе данных MySQL (таблица users)
# Проверка номера в базе данных MySQL (таблица users)
# def check_plate_in_db(conn, license_plate):
#     cursor = conn.cursor()
#     try:
#         cursor.execute("SELECT * FROM users WHERE license_plate = %s", (license_plate,))
#         result = cursor.fetchone()
        
#         if result:
#             print(f"Номер {license_plate} найден в таблице users.")
#         else:
#             print(f"Номер {license_plate} не найден в таблице users.")
        
#         conn.commit()  # Зафиксируем изменения, хотя здесь они не должны быть
#         return result
#     finally:
#         cursor.close()  # Закрываем курсор, чтобы избежать утечек
# Сохранение данных в JSON и отправка на микроконтроллер

last_saved_time = {}
def save_to_json_single(data):
    file_name = f"vehicle_data.json"
    
    # Открываем файл на запись, чтобы каждый раз сохранять только один номер
    try:
        with open(file_name, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Данные о номере {data['LicensePlate']} сохранены в файл {file_name}.")
    except Exception as e:
        print(f"Ошибка при сохранении данных в JSON: {e}")

    # # Отправляем данные на микроконтроллер через UART
    # json_data = json.dumps(data)  # Преобразуем в строку JSON
    # # ser.write(json_data.encode())  # Отправляем через UART
    # print(f"Данные о номере {data['LicensePlate']} отправлены на микроконтроллер.")

# # Функция для установки соединения с микроконтроллером через UART
# def connect_to_uart(port, baudrate=9600):
#     try:
#         ser = serial.Serial(port, baudrate, timeout=1)
#         print(f"Соединение с {port} установлено.")
#         return ser
#     except serial.SerialException as e:
#         print(f"Ошибка при подключении к {port}: {e}")
#         return None

# Добавление информации в таблицу history
# Добавляем функцию для записи в таблицу history

def flush_tables(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("FLUSH TABLES;")
        print("Таблицы успешно обновлены (FLUSH TABLES выполнен).")
    except mysql.connector.Error as e:
        print(f"Ошибка при выполнении FLUSH TABLES: {e}")
    finally:
        cursor.close()

MIN_INTERVAL = timedelta(minutes=2)

def insert_into_history(conn, license_plate):
    cursor = conn.cursor()

    # Поиск данных в таблице users по распознанному номеру
    cursor.execute("""
        SELECT id, firstName, lastName, middleName,reqistration, parkingSpot 
        FROM users 
        WHERE license_plate = %s
    """, (license_plate,))

    user_data = cursor.fetchone()

    if user_data:
        user_id, first_name, last_name, middle_name,reqistration, parking_spot = user_data

        # Проверка последней записи в history для данного номера
        cursor.execute("""
            SELECT entryDateTime, exitDateTime FROM history 
            WHERE license_plate = %s
            ORDER BY entryDateTime DESC LIMIT 1  -- Берём последнюю запись
        """, (license_plate,))

        history_data = cursor.fetchone()
        current_time = datetime.now()

        if history_data:
            entry_date_time, exit_date_time = history_data

            # Если последняя запись содержит въезд, но нет выезда
            if entry_date_time is not None and exit_date_time is None:
                time_diff = current_time - entry_date_time
                print(f"Время с момента въезда: {time_diff}")

                if time_diff >= MIN_INTERVAL:
                    # Обновляем статус на 'выезд' и сохраняем время
                    cursor.execute("""
                        UPDATE history 
                        SET exitDateTime = %s, status = 'выезд' 
                        WHERE license_plate = %s AND exitDateTime IS NULL
                    """, (current_time.strftime("%Y-%m-%d %H:%M:%S"), license_plate))

                    conn.commit()
                    print(f"Статус для номера {license_plate} обновлён на 'выезд'.")
                    data = {
                        "LicensePlate": license_plate
                    }
                    save_to_json_single(data)
                    send_command_to_controller(RELAY_COMMAND)
                    
                else:
                    print(f"Игнорируем статус выезда для номера {license_plate} - интервал меньше {MIN_INTERVAL}.")
            else:

                time_since_exit = current_time - exit_date_time
                if time_since_exit >= MIN_INTERVAL:
                    # Если запись уже завершена, создаем новую запись для нового въезда
                    cursor.execute("""
                        INSERT INTO history (id, firstName, lastName, middleName,reqistration, license_plate, parkingSpot, status, entryDateTime)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (user_id, first_name, last_name, middle_name, reqistration, license_plate, parking_spot, 'въезд', current_time.strftime("%Y-%m-%d %H:%M:%S")))

                    conn.commit()
                    print(f"Создана новая запись для номера {license_plate} с статусом 'въезд'.")
                    data = {
                        "LicensePlate": license_plate
                    }
                    save_to_json_single(data)
                    send_command_to_controller(RELAY_COMMAND)
        else:
            # Если записи нет, добавляем новую
            cursor.execute("""
                INSERT INTO history (id, firstName, lastName, middleName,reqistration, license_plate, parkingSpot, status, entryDateTime)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (user_id, first_name, last_name, middle_name, reqistration, license_plate, parking_spot, 'въезд', current_time.strftime("%Y-%m-%d %H:%M:%S")))

            conn.commit()
            print(f"Запись для номера {license_plate} добавлена в таблицу history с статусом 'въезд'.")
            data = {
                        "LicensePlate": license_plate
                    }
            save_to_json_single(data)
            send_command_to_controller(RELAY_COMMAND)
    else:
        print(f"Номер {license_plate} не найден в таблице users.")

    cursor.close()

def frame_generator(video_src: str):
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_src}.")
        return None
    else:
        print(f"Video source {video_src} opened successfully.")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            print(f"End of stream for {video_src}")
            break
    cap.release()
    return None

def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

def extract_boxes(results, frame):
    labels, cord = results
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    labls_cords = {"numbers": [], "cars": []}
    for i in range(len(labels)):
        x1, y1, x2, y2 = (int(cord[i][j] * dim) for j, dim in enumerate([x_shape, y_shape, x_shape, y_shape]))

        if labels[i] == 0:  # Номерной знак
            labls_cords["numbers"].append((x1, y1, x2, y2))
        elif labels[i] == 1:  # Автомобиль
            labls_cords["cars"].append((x1, y1, x2, y2))

    return labls_cords

def draw_number_boxes(cars_list, frame):
    for car in cars_list:
        x1_number, y1_number, x2_number, y2_number = car[0]
        number = car[1]

        cv2.rectangle(frame, (x1_number, y1_number), (x2_number, y2_number), (255, 255, 255), 2)
        cv2.putText(frame, number, (x1_number - 20, y2_number + 30), 0, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

def is_within_detection_area(coords):
    detection_area = config.DETECTION_AREA
    xc = int((coords[0] + coords[2]) / 2)
    yc = int((coords[1] + coords[3]) / 2)
    return detection_area[0][0] < xc < detection_area[1][0] and detection_area[0][1] < yc < detection_area[1][1]

def correct_perspective(image, coords):
    if coords[0] < 0 or coords[1] < 0 or coords[2] > image.shape[1] or coords[3] > image.shape[0]:
        print("Coordinates out of bounds.")
        return None

    pts1 = np.float32([
        [coords[0], coords[1]], 
        [coords[2], coords[1]], 
        [coords[2], coords[3]], 
        [coords[0], coords[3]]
    ])
    
    width, height = 200, 60
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    corrected_image = cv2.warpPerspective(image, matrix, (width, height))

    return corrected_image

# Сохранение распознанного номера в таблицу history
# def save_to_history(conn, license_plate, timestamp):
#     try:
#         cursor = conn.cursor()
#         cursor.execute("INSERT INTO history (license_plate, timestamp) VALUES (%s, %s)", (license_plate, timestamp))
#         conn.commit()
#         print(f"Номер {license_plate} сохранен в таблице history с временем {timestamp}.")
#     except mysql.connector.Error as e:
#         print(f"Ошибка при сохранении в таблицу history: {e}")

# Основная логика работы с видео и детекцией
def process_stream(stream_id, video_file_path, detector, LPRnet, conn, seen_plates, output_size):
    global last_saved_time
    for raw_frame in frame_generator(video_file_path):
        # Изменение размера до 720p (1280x720)
        proc_frame = resize_image(raw_frame, output_size)

        # Получаем результаты от детектора
        results = detector.score_frame(proc_frame)
        labls_cords = extract_boxes(results, raw_frame)

        cars = []

        for plate_coords in labls_cords["numbers"]:
            if is_within_detection_area(plate_coords):
                plate_box_image = raw_frame[plate_coords[1]:plate_coords[3], plate_coords[0]:plate_coords[2]]

                plate_box_image = correct_perspective(raw_frame, plate_coords)

                plate_text = rec_plate(LPRnet, plate_box_image)

                if re.match(r"[A-Z]{1}[0-9]{3}[A-Z]{2}[0-9]{2,3}", plate_text):
                    cars.append([plate_coords, plate_text])

                    # Проверяем, есть ли номер в базе данных
                    plate_in_db = insert_into_history(conn, plate_text)


                    current_time = datetime.now()
                    # Если номер найден в базе данных и еще не сохранен
                    if plate_in_db and plate_text not in seen_plates:
                        if plate_text not in last_saved_time or (current_time - last_saved_time[plate_text]) >= timedelta(minutes=2):
                            data = {
                                "LicensePlate": plate_text,
                            }
                            save_to_json_single(data)
                            last_saved_time[plate_text] = current_time
                            # Сохраняем в таблицу history
                            # insert_into_history(conn, plate_text)
                            seen_plates.add(plate_text)  # Добавляем номер в множество

                            send_command_to_controller(RELAY_COMMAND)

        # Отрисовка рамок вокруг номеров
        if cars:
            drawn_frame = draw_number_boxes(cars, raw_frame)
        else:
            drawn_frame = raw_frame

        yield drawn_frame

# Основная функция
def main(yolo_model_path, yolo_conf, yolo_iou, lpr_model_path, lpr_max_len, lpr_dropout_rate, device):
    cv2.startWindowThread()
    print("Initializing YOLO model...")
    detector = ObjectDetection(yolo_model_path, conf=yolo_conf, iou=yolo_iou, device=device)
    print("YOLO model initialized.")
    print("Loading LPRNet model...")
    LPRnet = build_lprnet(lpr_max_len=lpr_max_len, phase=False, class_num=len(CHARS), dropout_rate=lpr_dropout_rate)
    LPRnet.to(torch.device(device))
    LPRnet.load_state_dict(torch.load(lpr_model_path, map_location=device))
    print("LPRNet model loaded successfully.")

    conn = connect_to_db()  # Подключаемся к базе данных MySQL

    seen_plates = set()  # Множество для хранения уже сохраненных номеров

    # Размеры финального окна для отображения каждого потока
    output_size = (1280, 720)  # 720p разрешение для каждого потока

    # Потоки с двух камер
    stream_1 = process_stream(1, config.RTSP_STREAM_1, detector, LPRnet, conn, seen_plates, output_size)
    stream_2 = process_stream(2, config.RTSP_STREAM_2, detector, LPRnet, conn, seen_plates, output_size)

    while True:
        # Получаем кадры с обеих камер
        frame_1 = next(stream_1, None)
        frame_2 = next(stream_2, None)

        # Проверка, что кадры с обоих потоков получены
        if frame_1 is None or frame_2 is None:
            break

        # Уменьшаем оба кадра до половины ширины экрана
        resized_frame_1 = resize_image(frame_1, (720, 480))  # половина 720p
        resized_frame_2 = resize_image(frame_2, (720, 480))  # половина 720p

        # Объединяем два кадра горизонтально
        combined_frame = np.hstack((resized_frame_1, resized_frame_2))

        # Отображаем объединённый кадр
        cv2.imshow("Combined Video Streams", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(config.YOLO_MODEL_PATH, config.YOLO_CONF, config.YOLO_IOU, config.LPR_MODEL_PATH, config.LPR_MAX_LEN, config.LPR_DROPOUT, config.DEVICE)
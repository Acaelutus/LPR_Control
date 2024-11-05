import time
import re
import cv2
import torch
import numpy as np
import pyodbc
import json
import sqlite3
from plate_recognition.model.lpr_net import build_lprnet
from plate_recognition.process_plate import rec_plate, CHARS
from vehicle_detection.detect_car_YOLO import ObjectDetection
from plate_matching import match_car
import config

# Коннект с базой данных
def connect_to_db():
    conn = sqlite3.connect('parking.db')  # Указываем имя базы данных SQLite
    return conn


# Проверка номера в базе данных

def check_plate_in_db(conn, license_plate):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM CarOwners WHERE CarNumber = ?", (license_plate,))
    result = cursor.fetchone()
    if result:
        print(f"Номер {license_plate} найден в базе данных.")
    else:
        print(f"Номер {license_plate} не найден в базе данных.")
    return result


def save_to_json(data):
    with open('vehicle_data.json', 'a', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
        json_file.write('\n')
    print(f"Данные о номере {data['LicensePlate']} сохранены в JSON.")

def frame_generator(video_src: str):
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return None
    else:
        print(f"Video source {video_src} opened successfully.")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            print("End of stream")
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

# Основная логика работы с видео и детекцией
def main(video_file_path, yolo_model_path, yolo_conf, yolo_iou, lpr_model_path, lpr_max_len, lpr_dropout_rate, device):
    cv2.startWindowThread()
    print("Initializing YOLO model...")
    detector = ObjectDetection(yolo_model_path, conf=yolo_conf, iou=yolo_iou, device=device)
    print("YOLO model initialized.")
    print("Loading LPRNet model...")
    LPRnet = build_lprnet(lpr_max_len=lpr_max_len, phase=False, class_num=len(CHARS), dropout_rate=lpr_dropout_rate)
    LPRnet.to(torch.device(device))
    LPRnet.load_state_dict(torch.load(lpr_model_path, map_location=device))
    print("LPRNet model loaded successfully.")

    conn = connect_to_db()  # Подключаемся к базе данных
    
    seen_plates = set()  # Множество для хранения уже сохраненных номеров

    for raw_frame in frame_generator(video_file_path):
        proc_frame = resize_image(raw_frame, (1280, 720))

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
                    plate_in_db = check_plate_in_db(conn, plate_text)

                    # Если номер найден в базе данных и еще не сохранен
                    if plate_in_db and plate_text not in seen_plates:
                        # Формируем данные для сохранения
                        data = {
                            "LicensePlate": plate_text,
                            "InDatabase": True,  # Номер найден в базе данных
                            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")  # Текущее время
                        }
                        save_to_json(data)
                        seen_plates.add(plate_text)  # Добавляем номер в множество

        # Отрисовка рамок вокруг номеров
        if cars:
            drawn_frame = draw_number_boxes(cars, raw_frame)
        else:
            drawn_frame = raw_frame

        cv2.imshow("video", resize_image(drawn_frame, config.FINAL_FRAME_RES))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break



if __name__ == "__main__":
    main(
        config.FILE_PATH,
        config.YOLO_MODEL_PATH,
        config.YOLO_CONF, 
        config.YOLO_IOU,
        config.LPR_MODEL_PATH,
        config.LPR_MAX_LEN, 
        config.LPR_DROPOUT, 
        config.DEVICE
    )

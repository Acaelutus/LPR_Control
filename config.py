import os
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Path to video file or RTSP stream
# RTSP_STREAM_1 = os.environ.get('rtsp_stream', 'rtsp://admin:zxcvbn12@192.168.1.108:554/RVi/1/1')
# RTSP_STREAM_2 = os.environ.get('rtsp_stream', 'rtsp://admin:123456AS@192.168.1.107:554/cam/realmonitor?channel=1&subtype=0')
VIDEO_FILE_1 = os.environ.get('file_path', os.path.normpath("test/videos/test2.mp4"))
VIDEO_FILE_2 = os.environ.get('file_path', os.path.normpath("test/videos/rfpass.mp4"))
FILE_PATH = os.environ.get('file_path', os.path.normpath("test/videos/test2.mp4"))
UART_PORT = "COM4"  # Заменить на реальный порт


# YOLO model configurations
YOLO_MODEL_PATH = os.environ.get('yolo_model', os.path.normpath("vehicle_detection\skud.pt"))
YOLO_CONF = 0.2  # Confidence threshold
YOLO_IOU = 0.2   # Intersection Over Union threshold

# LPRNet model configurations
LPR_MODEL_PATH = os.environ.get('lpr_model', os.path.normpath("plate_recognition/model/weights/LPRNet__iteration_2000_28.09.pth"))
LPR_MAX_LEN = 9
LPR_DROPOUT = 0


# Final frame resolution for display
FINAL_FRAME_RES = (1280, 720)

# Detection area for license plate recognition
DETECTION_AREA = [(0, 0), (1280, 720)]

#DB 
DB_HOST ='localhost'
DB_USER = 'root'
DB_PASSWORD ='root'
DB_NAME = 'rfcontrol'
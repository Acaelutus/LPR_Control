import torch

class ObjectDetection:
    """
    Класс выполняет детекцию объектов на видео-фреймах, используя YOLO модель.
    """

    def __init__(self, model_path, conf, iou, device):
        self.__model_path = model_path
        self.device = device
        self.model = self.load_model()  # Загружаем локальную модель через hub
        self.model.conf = conf
        self.model.iou = iou
        self.model.to(self.device)

    def load_model(self):
        """
        Функция загружает YOLOv5 модель через PyTorch Hub с кастомными весами.
        """
        # Загрузка модели через hub, указав путь к весам
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.__model_path)
        return model

    def score_frame(self, frame):
        """
        Функция обрабатывает каждый фрейм и возвращает результаты детекции.
        """
        with torch.no_grad():
            results = self.model(frame)
        labels = results.xyxyn[0][:, -1].to("cpu").numpy()
        cords = results.xyxyn[0][:, :-1].to("cpu").numpy()
        return labels, cords

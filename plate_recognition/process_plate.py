# plate_recognition/process_plate.py
import cv2
import numpy as np
import torch

CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", 
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", 
    "W", "X", "Y", "Z", "I", "O", "_"
]

def rec_plate(lprnet, img) -> str:
    # preproc img
    image = img
    width, length, _ = image.shape
    
    # Apply augmentation (rotation and scale) to make the model robust to angles
    center = (length // 2, width // 2)
    angle = np.random.uniform(-15, 15)  # Rotate image randomly by [-15, 15] degrees
    scale = np.random.uniform(0.9, 1.1)  # Random scaling
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(image, rotation_matrix, (length, width))

    image = cv2.resize(image, (94, 24))
    image = image.astype("float32")
    image -= 127.5
    image *= 0.0078125
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)

    # Перенос изображения на то же устройство, что и модель
    device = next(lprnet.parameters()).device  # Определение устройства модели
    image = image.to(device)  # Перенос изображения на это устройство

    # forward
    preds = lprnet(image)

    # decode
    preds = preds.cpu().detach().numpy()
    label = ""
    for i in range(preds.shape[0]):
        preds = preds[i, :, :]
        preds_label = list()
        for j in range(preds.shape[1]):
            preds_label.append(np.argmax(preds[:, j], axis=0))
        pre_c = preds_label[0]
        if pre_c != len(CHARS) - 1:
            label += CHARS[pre_c]
        for c in preds_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            label += CHARS[c]
            pre_c = c
    return label

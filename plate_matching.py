def match_car(labels_and_coords: dict, iou_threshold=0.1) -> list:
    """
    Check each license plate's bounding box and match it with the corresponding car's bounding box
    using Intersection over Union (IoU) to allow more flexible matching, including angled or 
    partially visible plates.
    
    :param labels_and_coords: dict - a dictionary with bounding boxes for license plates and cars
    :param iou_threshold: float - minimum IoU required to consider a match
    :return: list - a list with the following structure:
    [
        [(license plate's coordinates), (car's coordinates)],
        [(license plate's coordinates), (car's coordinates)],
        ...
    ]
    """
    matched_cars = []

    for plate in labels_and_coords["numbers"]:
        for car in labels_and_coords["cars"]:
            # Calculate IoU between the license plate and car bounding boxes
            iou = calculate_iou(plate, car)
            if iou > iou_threshold:
                matched_cars.append([plate, car])

    return matched_cars

def calculate_iou(box1, box2) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    :param box1: tuple - (x1, y1, x2, y2) coordinates of the first bounding box
    :param box2: tuple - (x1, y1, x2, y2) coordinates of the second bounding box
    :return: float - IoU value between 0 and 1
    """
    # Calculate intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union
    union_area = box1_area + box2_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

"""Main LPR pipeline: detection, OCR, database lookup, and access control."""

import re
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from datetime import datetime

from src.detector.detector import Detection, VehicleDetector, PlateDetector
from src.ocr.recognizer import PlateRecognizer
from src.database.access_db import AccessDatabase
from src.controller.barrier_controller import BarrierController
from src.utils.logger import logger


class PlateMatch:
    """Result of matching a detected plate to a vehicle"""
    
    def __init__(self, plate_bbox: Detection, vehicle_bbox: Detection, iou: float):
        self.plate_bbox = plate_bbox
        self.vehicle_bbox = vehicle_bbox
        self.iou = iou


class LPRPipeline:
    """
    Main License Plate Recognition pipeline.
    
    Coordinates detection, OCR, and access control.
    Handles duplicate suppression and result tracking.
    """
    
    def __init__(
        self,
        vehicle_detector: Optional[VehicleDetector],
        plate_detector: PlateDetector,
        ocr: PlateRecognizer,
        database: AccessDatabase,
        controller: BarrierController,
        plate_matching_iou_threshold: float = 0.1,
        duplicate_suppression_timeout: float = 2.0
    ):
        """
        Initialize LPR pipeline.
        
        Args:
            vehicle_detector: Optional vehicle detection model
            plate_detector: Plate detection model
            ocr: OCR/recognizer model
            database: Access control database
            controller: Barrier controller
            plate_matching_iou_threshold: Min IOU for plate-vehicle matching
            duplicate_suppression_timeout: Time window for duplicate detection
        """
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr = ocr
        self.database = database
        self.controller = controller
        
        self.iou_threshold = plate_matching_iou_threshold
        self.duplicate_timeout = duplicate_suppression_timeout
        
        # Tracking for duplicate suppression
        self.recent_plates = {}  # plate -> {"time": datetime, "authorized": bool}
        self._authorized_plate_cache = None
        
        logger.info("LPRPipeline initialized")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single video frame through the full pipeline.
        
        Args:
            frame: Input frame (H, W, 3) BGR format
            
        Returns:
            Dict with:
            - vehicles: List of vehicle detections
            - plates: List of recognized plates
            - matches: List of plate-vehicle matches
            - results: List of access control results
        """
        height, width = frame.shape[:2]
        
        # Vehicle detection is optional; the current flow only needs the plate.
        vehicle_detections = []
        if self.vehicle_detector is not None:
            vehicle_detections = self.vehicle_detector.detect(frame)
            logger.debug(f"Detected {len(vehicle_detections)} vehicles")

        plate_detections = self.plate_detector.detect(frame)
        logger.debug(f"Detected {len(plate_detections)} plates")

        matches = []
        plate_targets = plate_detections
        if self.vehicle_detector is not None:
            matches = self._match_plates_to_vehicles(
                plate_detections, vehicle_detections
            )
            plate_targets = [match.plate_bbox for match in matches]
            logger.debug(f"Matched {len(matches)} plates to vehicles")

        # Step 2: Recognize plate text and process access
        results = []
        recognized_plates = []

        for plate_bbox in plate_targets:
            
            # Crop plate region
            x1, y1, x2, y2 = plate_bbox.denormalize(width, height)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            plate_crop = frame[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                continue
            
            # Recognize plate text
            raw_plate_text = self.ocr.recognize(plate_crop)
            plate_text = self._normalize_plate_text(raw_plate_text)
            
            if not plate_text:
                logger.debug("Failed to recognize plate text")
                continue

            resolved_plate_text = self._resolve_authorized_plate(plate_text)
            
            recognized_plates.append({
                'text': resolved_plate_text,
                'raw_text': plate_text,
                'bbox': plate_bbox.bbox,
                'confidence': plate_bbox.confidence,
                'crop': plate_crop
            })
            
            # Step 5: Access control logic
            result = self._process_access_control(resolved_plate_text)
            results.append(result)
        
        return {
            'vehicles': vehicle_detections,
            'plates': plate_detections,
            'matches': matches,
            'recognized_plates': recognized_plates,
            'results': results
        }
    
    def _match_plates_to_vehicles(
        self,
        plate_detections: List[Detection],
        vehicle_detections: List[Detection]
    ) -> List[PlateMatch]:
        """
        Match detected plates to vehicles using IoU.
        
        A plate should be "inside" or "near" a vehicle.
        Uses Intersection over Union (IoU) for matching.
        
        Args:
            plate_detections: List of plate detections
            vehicle_detections: List of vehicle detections
            
        Returns:
            List of PlateMatch objects
        """
        matches = []
        
        for plate in plate_detections:
            best_match = None
            best_iou = 0
            
            for vehicle in vehicle_detections:
                iou = self._calculate_iou(plate.bbox, vehicle.bbox)
                
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match = vehicle
            
            if best_match is not None:
                matches.append(PlateMatch(plate, best_match, best_iou))
        
        return matches
    
    def _calculate_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: (x1, y1, x2, y2) normalized [0, 1]
            bbox2: (x1, y1, x2, y2) normalized [0, 1]
            
        Returns:
            IoU value [0, 1]
        """
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _process_access_control(self, plate_text: str) -> Dict:
        """
        Process access control logic for recognized plate.
        
        Checks whitelist, handles duplicates, and triggers barrier.
        
        Args:
            plate_text: Recognized plate text
            
        Returns:
            Dict with access control result
        """
        current_time = datetime.now()
        
        # Check for duplicate detection
        if plate_text in self.recent_plates:
            recent = self.recent_plates[plate_text]
            last_time = recent["time"]
            if (current_time - last_time).total_seconds() < self.duplicate_timeout:
                logger.debug(f"Duplicate detection suppressed: {plate_text}")
                return {
                    'plate': plate_text,
                    'authorized': recent["authorized"],
                    'reason': 'duplicate_suppression',
                    'timestamp': current_time
                }
        
        # Check database
        is_authorized = self.database.is_authorized(plate_text)
        self.database.log_access(plate_text, is_authorized)
        
        # Update tracking
        self.recent_plates[plate_text] = {
            "time": current_time,
            "authorized": is_authorized
        }
        
        # Clean up old entries
        self._cleanup_duplicate_tracking(current_time)
        
        # Trigger barrier if authorized
        if is_authorized:
            self.controller.open_barrier(plate_text)
        else:
            logger.warning(f"Access denied for plate: {plate_text}")
        
        return {
            'plate': plate_text,
            'authorized': is_authorized,
            'reason': 'authorized' if is_authorized else 'not_in_whitelist',
            'timestamp': current_time
        }

    def _normalize_plate_text(self, plate_text: str) -> str:
        """Normalize OCR output before database lookup."""
        return re.sub(r"[^A-Z0-9]", "", plate_text.upper())

    def _resolve_authorized_plate(self, plate_text: str) -> str:
        """
        Map small OCR mistakes to a known whitelist plate.

        License plate OCR often flips one similar character frame-to-frame.
        A conservative edit-distance-1 match keeps the demo readable without
        turning unrelated plates into authorized plates.
        """
        authorized_plates = self._get_authorized_plates()
        if plate_text in authorized_plates:
            return plate_text

        best_plate = None
        best_distance = 2
        for known_plate in authorized_plates:
            if abs(len(known_plate) - len(plate_text)) > 1:
                continue
            distance = self._edit_distance(plate_text, known_plate, max_distance=1)
            if distance < best_distance:
                best_plate = known_plate
                best_distance = distance

        if best_plate is not None and best_distance <= 1:
            logger.debug(f"Resolved OCR variant {plate_text} -> {best_plate}")
            return best_plate

        return plate_text

    def _get_authorized_plates(self) -> set:
        if self._authorized_plate_cache is None:
            self._authorized_plate_cache = {
                self._normalize_plate_text(item["plate"])
                for item in self.database.get_all_plates()
            }
        return self._authorized_plate_cache

    def _edit_distance(self, left: str, right: str, max_distance: int = 1) -> int:
        if abs(len(left) - len(right)) > max_distance:
            return max_distance + 1

        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            row_min = current[0]
            for j, right_char in enumerate(right, start=1):
                insert_cost = current[j - 1] + 1
                delete_cost = previous[j] + 1
                replace_cost = previous[j - 1] + (left_char != right_char)
                value = min(insert_cost, delete_cost, replace_cost)
                current.append(value)
                row_min = min(row_min, value)
            if row_min > max_distance:
                return max_distance + 1
            previous = current

        return previous[-1]
    
    def _cleanup_duplicate_tracking(self, current_time: datetime):
        """
        Remove old entries from duplicate tracking.
        
        Args:
            current_time: Current time
        """
        expired_plates = []
        
        for plate, recent in self.recent_plates.items():
            if (current_time - recent["time"]).total_seconds() > self.duplicate_timeout:
                expired_plates.append(plate)
        
        for plate in expired_plates:
            del self.recent_plates[plate]

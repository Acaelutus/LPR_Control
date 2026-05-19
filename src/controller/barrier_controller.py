"""Barrier controller interface with simulated and serial implementations."""

from typing import Optional
from datetime import datetime
from abc import ABC, abstractmethod
from src.utils.logger import logger


class BarrierController(ABC):
    """Abstract base class for barrier control"""
    
    @abstractmethod
    def open_barrier(self, plate: str) -> bool:
        """Open the barrier"""
        pass
    
    @abstractmethod
    def close_barrier(self, plate: str) -> bool:
        """Close the barrier"""
        pass


class SimulatedBarrierController(BarrierController):
    """Simulated barrier controller for runs without hardware."""
    
    def __init__(self):
        """Initialize simulated controller"""
        self.barrier_open = False
        self.last_plate = None
        self.last_action_time = None
        logger.info("SimulatedBarrierController initialized")
    
    def open_barrier(self, plate: str) -> bool:
        """
        Simulate opening barrier.
        
        Args:
            plate: License plate that triggered access
            
        Returns:
            True if successful
        """
        self.barrier_open = True
        self.last_plate = plate
        self.last_action_time = datetime.now()
        
        logger.info("=" * 60)
        logger.info("ACCESS GRANTED")
        logger.info(f"   License Plate: {plate}")
        logger.info(f"   Time: {self.last_action_time.strftime('%H:%M:%S')}")
        logger.info("   Barrier OPENING...")
        logger.info("=" * 60)
        
        return True
    
    def close_barrier(self, plate: str) -> bool:
        """
        Simulate closing barrier.
        
        Args:
            plate: License plate
            
        Returns:
            True if successful
        """
        self.barrier_open = False
        self.last_plate = plate
        self.last_action_time = datetime.now()
        
        logger.info("-" * 60)
        logger.info("BARRIER CLOSING")
        logger.info(f"   License Plate: {plate}")
        logger.info(f"   Time: {self.last_action_time.strftime('%H:%M:%S')}")
        logger.info("-" * 60)
        
        return True


class SerialBarrierController(BarrierController):
    """ESP32 barrier controller over a serial port."""
    
    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        timeout: float = 1.0
    ):
        """
        Initialize serial barrier controller.
        
        Args:
            port: Serial port (e.g., "COM3", "/dev/ttyUSB0")
            baudrate: Baud rate for serial communication
            timeout: Serial read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        
        self._connect()
        logger.info(f"SerialBarrierController initialized on {port}")
    
    def _connect(self):
        """Establish serial connection"""
        try:
            import serial
            self.serial_conn = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout
            )
            logger.info(f"Connected to {self.port} at {self.baudrate} baud")
        except ImportError:
            raise ImportError(
                "pyserial not installed. Install with: pip install pyserial"
            )
        except Exception as e:
            logger.error(f"Failed to connect to {self.port}: {e}")
            self.serial_conn = None
    
    def _send_command(self, command: str) -> bool:
        """
        Send command to ESP32.
        
        Args:
            command: Command string
            
        Returns:
            True if successful
        """
        if self.serial_conn is None:
            logger.error("Serial connection not established")
            return False
        
        try:
            self.serial_conn.write((command + '\n').encode())
            logger.info(f"Sent command to ESP32: {command}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def open_barrier(self, plate: str) -> bool:
        """
        Open barrier via ESP32.
        
        Args:
            plate: License plate
            
        Returns:
            True if command sent successfully
        """
        logger.info(f"ACCESS GRANTED - Opening barrier for plate: {plate}")
        return self._send_command("OPEN_BARRIER")
    
    def close_barrier(self, plate: str) -> bool:
        """
        Close barrier via ESP32.
        
        Args:
            plate: License plate
            
        Returns:
            True if command sent successfully
        """
        logger.info(f"Closing barrier for plate: {plate}")
        return self._send_command("CLOSE_BARRIER")
    
    def __del__(self):
        """Close connection on cleanup"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logger.info("Serial connection closed")


def create_controller(
    controller_type: str = "simulated",
    **kwargs
) -> BarrierController:
    """
    Factory function to create appropriate controller.
    
    Args:
        controller_type: "simulated" or "serial"
        **kwargs: Additional arguments for controller
        
    Returns:
        BarrierController instance
    """
    if controller_type == "simulated":
        return SimulatedBarrierController()
    elif controller_type == "serial":
        return SerialBarrierController(
            port=kwargs.get("port", "COM3"),
            baudrate=kwargs.get("baudrate", 115200)
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

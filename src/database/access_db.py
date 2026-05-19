"""Access-control database backends for whitelist and event logging."""

import sqlite3
import json
from contextlib import closing
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from abc import ABC, abstractmethod
from src.utils.logger import logger


class AccessDatabase(ABC):
    """Abstract base class for access control databases"""
    
    @abstractmethod
    def is_authorized(self, plate: str) -> bool:
        """Check if plate is authorized"""
        pass
    
    @abstractmethod
    def add_plate(self, plate: str, owner: str = "Unknown") -> bool:
        """Add a plate to whitelist"""
        pass
    
    @abstractmethod
    def remove_plate(self, plate: str) -> bool:
        """Remove a plate from whitelist"""
        pass
    
    @abstractmethod
    def get_all_plates(self) -> List[Dict]:
        """Get all authorized plates"""
        pass
    
    @abstractmethod
    def log_access(self, plate: str, authorized: bool):
        """Log access attempt"""
        pass


class SQLiteAccessDatabase(AccessDatabase):
    """SQLite access-control database."""
    
    def __init__(self, db_path: str = "data/access_list.db"):
        """
        Initialize SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"SQLiteAccessDatabase initialized: {db_path}")
    
    def _init_database(self):
        """Create tables if they don't exist"""
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            cursor = conn.cursor()
            
            # Whitelist table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS whitelist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate TEXT UNIQUE NOT NULL,
                    owner TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Access log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate TEXT NOT NULL,
                    authorized INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def is_authorized(self, plate: str) -> bool:
        """Check if plate is authorized"""
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM whitelist WHERE plate = ?", (plate,))
            return cursor.fetchone() is not None
    
    def add_plate(self, plate: str, owner: str = "Unknown") -> bool:
        """Add plate to whitelist"""
        try:
            with closing(sqlite3.connect(str(self.db_path))) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO whitelist (plate, owner) VALUES (?, ?)",
                    (plate.upper(), owner)
                )
                conn.commit()
            logger.info(f"Added plate {plate} to whitelist")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Plate {plate} already in whitelist")
            return False
    
    def remove_plate(self, plate: str) -> bool:
        """Remove plate from whitelist"""
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM whitelist WHERE plate = ?", (plate,))
            conn.commit()
            logger.info(f"Removed plate {plate} from whitelist")
            return cursor.rowcount > 0
    
    def get_all_plates(self) -> List[Dict]:
        """Get all authorized plates"""
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT plate, owner, added_at FROM whitelist")
            rows = cursor.fetchall()
            return [
                {"plate": row[0], "owner": row[1], "added_at": row[2]}
                for row in rows
            ]
    
    def log_access(self, plate: str, authorized: bool):
        """Log access attempt"""
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO access_log (plate, authorized) VALUES (?, ?)",
                (plate, 1 if authorized else 0)
            )
            conn.commit()
        
        status = "AUTHORIZED" if authorized else "DENIED"
        logger.info(f"Access attempt: {plate} - {status}")


class JSONAccessDatabase(AccessDatabase):
    """JSON access-control database."""
    
    def __init__(self, json_path: str = "data/whitelist.json"):
        """
        Initialize JSON database.
        
        Args:
            json_path: Path to JSON whitelist file
        """
        self.json_path = Path(json_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = self.json_path.parent / "access_log.json"
        
        self._init_json()
        logger.info(f"JSONAccessDatabase initialized: {json_path}")
    
    def _init_json(self):
        """Initialize JSON files if they don't exist"""
        if not self.json_path.exists():
            self._save_whitelist({"plates": []})
        
        if not self.log_path.exists():
            self._save_log([])
    
    def _load_whitelist(self) -> Dict:
        """Load whitelist from JSON"""
        with open(self.json_path, 'r') as f:
            return json.load(f)
    
    def _save_whitelist(self, data: Dict):
        """Save whitelist to JSON"""
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_log(self) -> List:
        """Load access log from JSON"""
        with open(self.log_path, 'r') as f:
            return json.load(f)
    
    def _save_log(self, data: List):
        """Save access log to JSON"""
        with open(self.log_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def is_authorized(self, plate: str) -> bool:
        """Check if plate is authorized"""
        whitelist = self._load_whitelist()
        plates = [p["plate"] for p in whitelist["plates"]]
        return plate.upper() in plates
    
    def add_plate(self, plate: str, owner: str = "Unknown") -> bool:
        """Add plate to whitelist"""
        whitelist = self._load_whitelist()
        
        # Check if already exists
        if any(p["plate"] == plate.upper() for p in whitelist["plates"]):
            logger.warning(f"Plate {plate} already in whitelist")
            return False
        
        whitelist["plates"].append({
            "plate": plate.upper(),
            "owner": owner,
            "added_at": datetime.now().isoformat()
        })
        
        self._save_whitelist(whitelist)
        logger.info(f"Added plate {plate} to whitelist")
        return True
    
    def remove_plate(self, plate: str) -> bool:
        """Remove plate from whitelist"""
        whitelist = self._load_whitelist()
        original_count = len(whitelist["plates"])
        
        whitelist["plates"] = [
            p for p in whitelist["plates"]
            if p["plate"] != plate.upper()
        ]
        
        if len(whitelist["plates"]) < original_count:
            self._save_whitelist(whitelist)
            logger.info(f"Removed plate {plate} from whitelist")
            return True
        
        return False
    
    def get_all_plates(self) -> List[Dict]:
        """Get all authorized plates"""
        whitelist = self._load_whitelist()
        return whitelist.get("plates", [])
    
    def log_access(self, plate: str, authorized: bool):
        """Log access attempt"""
        log = self._load_log()
        log.append({
            "plate": plate,
            "authorized": authorized,
            "timestamp": datetime.now().isoformat()
        })
        self._save_log(log)
        
        status = "AUTHORIZED" if authorized else "DENIED"
        logger.info(f"Access attempt: {plate} - {status}")


def create_database(db_type: str = "sqlite", **kwargs) -> AccessDatabase:
    """
    Factory function to create appropriate database instance.
    
    Args:
        db_type: "sqlite" or "json"
        **kwargs: Additional arguments (db_path, json_path)
        
    Returns:
        AccessDatabase instance
    """
    if db_type == "sqlite":
        return SQLiteAccessDatabase(kwargs.get("db_path", "data/access_list.db"))
    elif db_type == "json":
        return JSONAccessDatabase(kwargs.get("json_path", "data/whitelist.json"))
    else:
        raise ValueError(f"Unknown database type: {db_type}")

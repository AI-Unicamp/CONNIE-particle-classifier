import os
import sqlite3
import sys
from pathlib import Path
from dataclasses import dataclass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core import SCHEMA_FILE, DATABASE_FOLDER

@dataclass
class ImageData:
    """Image data."""
    filename: str
    img_idx: int
    username: str
    date: str
    label: str


class Database:
    def __init__(self):
        self.database_name = "connie_label.db"
        self.table_name = "events"
        self.db_path = os.path.join(DATABASE_FOLDER, self.database_name)

    def create_database(self):
        """Creates a database based on current schema. """
        schema = Path(SCHEMA_FILE)
        if not schema.exists():
            raise FileNotFoundError(f"Schema file not found: {schema}")

        with open(schema, "r", encoding="utf-8") as schema_file:
            schema_str = schema_file.read()

        connection = sqlite3.connect(self.db_path)
        connection.execute(schema_str)
        connection.commit()
        connection.close()

    def insert_event_info(self, img: ImageData):
        """Insert data into database

        Args:
            img (ImageData): database data
        """
        connection = sqlite3.connect(self.db_path)
        command = f"INSERT INTO {self.table_name}\
              (filename, img_idx, username, date, label)\
                  VALUES (?, ?, ?, ?, ?);"
        cursor = connection.cursor()
        cursor.execute(command, (img.filename, img.img_idx, 
                                 img.username, img.date, img.label))
        connection.commit()
        connection.close()  
        
    def labeled_events(self):
        """ Return labeled images """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        cmd = f"SELECT filename, img_idx, username from {self.table_name}"
        cursor.execute(cmd)
        data_rows = cursor.fetchall()
        connection.close()
        return data_rows

    def search_events(self, condition: str):
        """ Return database data according to condition

        Args:
            condition (str): condition to get data

        Returns:
            tuple: all data that matches the condition
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        cmd = f"SELECT filename, img_idx, username from {self.table_name} WHERE {condition}"
        cursor.execute(cmd)
        data_rows = cursor.fetchall()
        connection.close()
        return data_rows

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

    def get_discrepant_label_votes_by_event(self):
        """ Returns a dictionary of discrepant events with label votes per event
        
        Returns:
            dict: A dictionary where keys are (filename, img_idx) tuples and
                  values are lists of (label, count) tuples representing how many
                  users assigned each label.
        """
        cmd = """
            SELECT 
                e.filename,
                e.img_idx,
                e.label,
                COUNT(*) AS user_count
            FROM events e
            WHERE (e.filename, e.img_idx) IN (
                SELECT filename, img_idx
                FROM events
                GROUP BY filename, img_idx
                HAVING COUNT(DISTINCT label) > 1
            )
            GROUP BY e.filename, e.img_idx, e.label
            ORDER BY e.filename, e.img_idx, user_count DESC;
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        cursor.execute(cmd)
        data_rows = cursor.fetchall()
        connection.close()

        # Group rows by (filename, img_idx)
        from collections import defaultdict
        events = defaultdict(list)
        for filename, img_idx, label, count in data_rows:
            events[(filename, img_idx)].append((label, count))

        return events

    def has_user_annotated_event(self, filename: str,
                                 img_idx: int,
                                 username: str) -> bool:
        """Check if a given user annotated a specific event

        Args:
            filename (str): The event's filename
            img_idx (int): The event's image index
            username (str): The username to check

        Returns:
            bool: True if user annotated the event, False otherwise
        """
        cmd = """ SELECT 1 FROM events
                  WHERE filename = ? AND img_idx = ? AND username = ?
                  LIMIT 1;
              """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        cursor.execute(cmd, (filename, img_idx, username))
        data_row = cursor.fetchone()
        connection.close()
        return data_row is not None

    def get_labels_for_event(self, filename, img_idx):
        """Return list of unique labels assigned to a given event."""
        cmd = """SELECT DISTINCT label FROM events
                WHERE filename = ? AND img_idx = ?"""
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        cursor.execute(cmd, (filename, img_idx))
        rows = cursor.fetchall()
        connection.close()
        return [row[0] for row in rows]
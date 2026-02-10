import sqlite3
from pathlib import Path
import datetime
import logging

logger = logging.getLogger(__name__)


def init_db(db_path="models/images.db"):
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    # subjects table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
        """
    )
    # images table: store image bytes as BLOB
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER,
            data BLOB,
            is_train INTEGER DEFAULT 1,
            filename TEXT,
            created_at TEXT,
            FOREIGN KEY(subject_id) REFERENCES subjects(id)
        )
        """
    )
    conn.commit()
    conn.close()


def _get_conn(db_path="models/images.db"):
    init_db(db_path)
    return sqlite3.connect(str(Path(db_path)))


def add_subject(name, db_path="models/images.db"):
    conn = _get_conn(db_path)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO subjects (name) VALUES (?)", (name,))
    conn.commit()
    cur.execute("SELECT id FROM subjects WHERE name = ?", (name,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def add_image(subject_name, image_bytes, is_train=True, filename=None, db_path="models/images.db"):
    conn = _get_conn(db_path)
    cur = conn.cursor()
    subj_id = add_subject(subject_name, db_path)
    ts = datetime.datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO images (subject_id, data, is_train, filename, created_at) VALUES (?,?,?,?,?)",
        (subj_id, image_bytes, 1 if is_train else 0, filename, ts),
    )
    conn.commit()
    conn.close()


def get_images(is_train=True, db_path="models/images.db"):
    """Return list of tuples (image_bytes, subject_name)"""
    conn = _get_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT i.data, s.name FROM images i JOIN subjects s ON i.subject_id = s.id WHERE i.is_train = ? ORDER BY i.id",
        (1 if is_train else 0,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def list_subjects(db_path="models/images.db"):
    conn = _get_conn(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM subjects ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return rows

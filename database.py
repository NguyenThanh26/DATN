import sqlite3
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class DatabaseHandler:
    """
    DatabaseHandler để quản lý cơ sở dữ liệu SQLite về phụ đề.
    """
    def __init__(self, db_path: str = "data/database.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self) -> None:
        """
        Tạo bảng subtitles nếu chưa tồn tại.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS subtitles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT,
                        origin_language TEXT,
                        translate_language TEXT,
                        use_correction INTEGER,
                        embed_subtitle INTEGER,
                        subtitle_path TEXT,
                        status TEXT
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")

    def add_file(self, file_name: str, origin_language: str, translate_language: str, use_correction: bool, embed_subtitle: str) -> int:
        """
        Thêm một bản ghi phụ đề vào cơ sở dữ liệu.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO subtitles (file_name, origin_language, translate_language, use_correction, embed_subtitle, status)
                    VALUES (?, ?, ?, ?, ?, 'PENDING')
                    """,
                    (file_name, origin_language, translate_language, use_correction, embed_subtitle)
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error adding file {file_name}: {str(e)}")
            return None

    def get_pending_files(self) -> List[Dict]:
        """
        Lấy tất cả các bản ghi chưa xử lý.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                files = conn.execute(
                    """
                    SELECT * FROM subtitles WHERE status='PENDING'
                    """
                ).fetchall()
                return [dict(file) for file in files]
        except Exception as e:
            logger.error(f"Error retrieving files: {str(e)}")
            return []

    def update_status(self, file_id: int, status: str) -> bool:
        """
        Cập nhật trạng thái của bản ghi phụ đề.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE subtitles SET status = ? WHERE id = ?
                    """,
                    (status, file_id)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating status for id {file_id}: {str(e)}")
            return False

    def update_subtitle_path(self, file_id: int, subtitle_path: str) -> bool:
        """
        Cập nhật thêm subtitle_path sau khi hoàn thành xử lý.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE subtitles SET subtitle_path = ? WHERE id = ?
                    """,
                    (subtitle_path, file_id)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating subtitle path for id {file_id}: {str(e)}")
            return False

import sqlite3
from typing import List, Dict

DB_NAME = "chat_history.db"


def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """
        )
        conn.commit()


def add_message(user_id: str, role: str, content: str) -> None:
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_messages (user_id, role, content)
            VALUES (?, ?, ?)
        """,
            (user_id, role, content),
        )  # <-- Supply parameters here
        conn.commit()


def get_chat_history(user_id: str) -> List[Dict[str, str]]:
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT role, content
                FROM chat_messages
                WHERE user_id = ?
                ORDER BY created_at ASC
        """,
            (user_id,),
        )
        rows = cursor.fetchall()
    history = []
    for row in rows:
        role, content = row
        history.append({"role": role, "content": content})
    return history


def delete_table():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        DROP TABLE chat_messages;
        """
        )

"""DB logic for storing/fetching user data."""

import psycopg2
import os
from werkzeug.security import generate_password_hash, check_password_hash


class UserRepository:
    def __init__(self):
        self.connection = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST")
        )
        self._initialize_database()

    def _initialize_database(self):
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL
                )
            """)
            self.connection.commit()

    def create_user(self, username, password, email):
        password_hash = generate_password_hash(password)
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO users (username, password_hash, email)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (username, password_hash, email))
            user_id = cursor.fetchone()[0]
            self.connection.commit()
            return user_id

    def find_user_by_username(self, username):
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT id, username, password_hash FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            return user if user else None

    def verify_password(self, stored_password_hash, password):
        return check_password_hash(stored_password_hash, password)

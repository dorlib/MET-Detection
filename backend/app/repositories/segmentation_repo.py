import psycopg2
import os

class SegmentationRepository:
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
                CREATE TABLE IF NOT EXISTS segmentation_results (
                    id SERIAL PRIMARY KEY,
                    original_image_url TEXT NOT NULL,
                    segmented_image_url TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.commit()

    def save_segmentation_result(self, original_image_url, segmented_image_url):
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO segmentation_results (original_image_url, segmented_image_url)
                VALUES (%s, %s)
                RETURNING id
            """, (original_image_url, segmented_image_url))
            result_id = cursor.fetchone()[0]
            self.connection.commit()
            return result_id

    def get_segmentation_results(self, limit=10):
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, original_image_url, segmented_image_url, timestamp
                FROM segmentation_results
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            return cursor.fetchall()

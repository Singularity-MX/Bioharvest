import pymysql
import os

DB_CONFIG = {
    "host": "localhost",
    "user": "pythondb",
    "password": os.getenv("DB_PASSWORD", "Javier117"),
    "database": "bioharvestdb",
}


def get_connection():
    return pymysql.connect(**DB_CONFIG)

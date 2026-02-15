import subprocess
from datetime import datetime
import os
from .connection import DB_CONFIG

BACKUP_DIR = "database/backup"
os.makedirs(BACKUP_DIR, exist_ok=True)


def backup_database():

    filepath = os.path.join(BACKUP_DIR, "backup_database.sql")

    command = [
        "mysqldump",
        "-h", DB_CONFIG["host"],
        "-u", DB_CONFIG["user"],
        f"--password={DB_CONFIG['password']}",
        DB_CONFIG["database"],
        "-r", filepath,
    ]

    subprocess.run(command, check=True)

    print(f"{datetime.now()} - Backup creado")

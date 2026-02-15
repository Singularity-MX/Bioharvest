from .connection import get_connection
from datetime import datetime


def save_measurement(temp, ph, r, g, b, i, filepath,
                     densidad_celular, lectura_id, nombre="NaN"):

    conn = get_connection()

    with conn.cursor() as cursor:
        query = """
        INSERT INTO bitacora
        (temperatura, ph, value_R, value_G, value_B,
         value_I, photo_src, densidad_celular, date,
         lectura_id, nombre)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        cursor.execute(query, (
            temp, ph, r, g, b, i,
            filepath,
            densidad_celular,
            datetime.now(),
            lectura_id,
            nombre
        ))

    conn.commit()
    conn.close()


def get_statistics():

    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM bitacora")
        rows = cursor.fetchall()
        columns = [c[0] for c in cursor.description]

    conn.close()

    return {"statistics": [dict(zip(columns, r)) for r in rows]}


def get_photos():

    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute("SELECT photo_src, date FROM bitacora")
        rows = cursor.fetchall()

    conn.close()

    return {
        "photos": [
            {"photo_src": r[0], "date": r[1].strftime("%Y-%m-%d %H:%M:%S")}
            for r in rows
        ]
    }

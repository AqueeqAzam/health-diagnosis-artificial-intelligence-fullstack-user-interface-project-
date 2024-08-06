import sqlite3

conn = sqlite3.connect('health.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS health_data
    (id INTEGER PRIMARY KEY, age REAL, gender INTEGER, body_temperature REAL, 
    pulse_rate INTEGER, respiration_rate INTEGER, blood_pressure REAL, 
    blood_oxygen INTEGER, weight REAL, blood_glucose INTEGER, diet_quality INTEGER, 
    health_status TEXT, recommendation TEXT)
''')

conn.commit()
conn.close()

print('Table created successfully')
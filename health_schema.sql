DROP TABLE IF EXISTS predictions;

CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age REAL,
    gender INTEGER,
    body_temperature REAL,
    pulse_rate INTEGER,
    respiration_rate INTEGER,
    blood_pressure REAL,
    blood_oxygen INTEGER,
    weight REAL,
    blood_glucose INTEGER,
    diet_quality INTEGER,
    prediction TEXT
);

# SQLite Database Design

## Table: Runs

| Column | Type |
|---------|------|
| run_id | INTEGER PRIMARY KEY |
| date | TEXT |
| runner | TEXT |
| duration | REAL |
| notes | TEXT |

---

## Table: RawSensorData

| Column | Type |
|---------|------|
| id | INTEGER PRIMARY KEY |
| run_id | INTEGER |
| timestamp | REAL |
| node | TEXT |
| ax | REAL |
| ay | REAL |
| az | REAL |
| gx | REAL |
| gy | REAL |
| gz | REAL |
| mx | REAL |
| my | REAL |
| mz | REAL |

---

## Table: Features

| Column | Type |
|---------|------|
| feature_id | INTEGER PRIMARY KEY |
| run_id | INTEGER |
| window_start | REAL |
| window_end | REAL |
| cadence | REAL |
| vertical_oscillation | REAL |
| running_smoothness | REAL |
| pelvic_tilt | REAL |
| pelvic_rotation | REAL |
| trunk_lean | REAL |
| foot_impact | REAL |
| footstrike_timing | REAL |
| symmetry | REAL |

---

## Table: Predictions

| Column | Type |
|---------|------|
| prediction_id | INTEGER PRIMARY KEY |
| run_id | INTEGER |
| prediction | TEXT |
| confidence | REAL |

---

## Table: Recommendations

| Column | Type |
|---------|------|
| recommendation_id | INTEGER PRIMARY KEY |
| prediction_id | INTEGER |
| title | TEXT |
| description | TEXT |
"""
Mock scheduling database backed by SQLite.

Two independent schedule domains live side by side:

  GENERAL APPOINTMENTS (existing demo)
    - slots: which (date, time) combos exist and whether they're booked
    - reservations: confirmed bookings (date, time, email, created_at)
    Business hours seeded for the next 14 days, Mon-Sat, 9am-5pm hourly.

  WAREHOUSE DOCK BOOKING (GigaCorp logistics extension)
    - dock_slots: which (date, time, bay) combos exist and whether booked
    - dock_reservations: confirmed dock bookings (shipment_id, carrier,
      driver_email, date, time, bay, created_at)
    Dock hours seeded for the next 14 days, every day, 3 bays, 6am-8pm
    hourly (warehouses run longer/more days than a customer-facing
    appointment desk).
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "scheduling.db"

# --- General appointment domain -------------------------------------------
BUSINESS_HOURS = ["09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]

# --- Warehouse dock domain --------------------------------------------------
DOCK_HOURS = [f"{h:02d}:00" for h in list(range(6, 20))]  # 06:00 .. 19:00
DOCK_BAYS = ["Bay-1", "Bay-2", "Bay-3"]


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # --- general appointment tables ---
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS slots (
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            booked INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (date, time)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            email TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    # --- warehouse dock tables ---
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dock_slots (
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            bay TEXT NOT NULL,
            booked INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (date, time, bay)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dock_reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shipment_id TEXT NOT NULL,
            carrier TEXT NOT NULL,
            driver_email TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            bay TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()

    today = datetime.now().date()

    # Seed general appointment slots: next 14 days, skip Sundays.
    for i in range(14):
        d = today + timedelta(days=i)
        if d.weekday() == 6:  # skip Sundays
            continue
        date_str = d.isoformat()
        for t in BUSINESS_HOURS:
            cur.execute(
                "INSERT OR IGNORE INTO slots (date, time, booked) VALUES (?, ?, 0)",
                (date_str, t),
            )

    # Seed dock slots: next 14 days, every day (warehouses run 7 days),
    # 3 bays x 14 hourly windows each.
    for i in range(14):
        d = today + timedelta(days=i)
        date_str = d.isoformat()
        for t in DOCK_HOURS:
            for bay in DOCK_BAYS:
                cur.execute(
                    "INSERT OR IGNORE INTO dock_slots (date, time, bay, booked) "
                    "VALUES (?, ?, ?, 0)",
                    (date_str, t, bay),
                )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# General appointment domain
# ---------------------------------------------------------------------------
def list_available_slots(date_str: str):
    conn = get_conn()
    rows = conn.execute(
        "SELECT time FROM slots WHERE date = ? AND booked = 0 ORDER BY time",
        (date_str,),
    ).fetchall()
    conn.close()
    return [r["time"] for r in rows]


def slot_exists(date_str: str, time_str: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM slots WHERE date = ? AND time = ?", (date_str, time_str)
    ).fetchone()
    conn.close()
    return row is not None


def is_slot_available(date_str: str, time_str: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT booked FROM slots WHERE date = ? AND time = ?",
        (date_str, time_str),
    ).fetchone()
    conn.close()
    if row is None:
        return False
    return row["booked"] == 0


def reserve(date_str: str, time_str: str, email: str) -> bool:
    """Attempt to reserve a slot. Returns False if it's already taken/doesn't exist."""
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT booked FROM slots WHERE date = ? AND time = ?",
        (date_str, time_str),
    ).fetchone()
    if row is None or row["booked"] == 1:
        conn.close()
        return False
    cur.execute(
        "UPDATE slots SET booked = 1 WHERE date = ? AND time = ?",
        (date_str, time_str),
    )
    cur.execute(
        "INSERT INTO reservations (date, time, email, created_at) VALUES (?, ?, ?, ?)",
        (date_str, time_str, email, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return True


# ---------------------------------------------------------------------------
# Warehouse dock domain
# ---------------------------------------------------------------------------
def list_available_dock_slots(date_str: str):
    """Returns list of {"time": ..., "bay": ...} dicts for open dock slots."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT time, bay FROM dock_slots WHERE date = ? AND booked = 0 "
        "ORDER BY time, bay",
        (date_str,),
    ).fetchall()
    conn.close()
    return [{"time": r["time"], "bay": r["bay"]} for r in rows]


def dock_date_exists(date_str: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM dock_slots WHERE date = ? LIMIT 1", (date_str,)
    ).fetchone()
    conn.close()
    return row is not None


def reserve_dock_slot(
    date_str: str, time_str: str, shipment_id: str, carrier: str, driver_email: str,
    bay: str | None = None,
) -> dict:
    """
    Attempt to reserve a dock slot at the given date/time.
    If `bay` is not specified, books the first available bay in that window.
    Returns {"success": True, "bay": ...} or {"success": False}.
    """
    conn = get_conn()
    cur = conn.cursor()

    if bay:
        row = cur.execute(
            "SELECT booked FROM dock_slots WHERE date = ? AND time = ? AND bay = ?",
            (date_str, time_str, bay),
        ).fetchone()
        if row is None or row["booked"] == 1:
            conn.close()
            return {"success": False}
        chosen_bay = bay
    else:
        row = cur.execute(
            "SELECT bay FROM dock_slots WHERE date = ? AND time = ? AND booked = 0 "
            "ORDER BY bay LIMIT 1",
            (date_str, time_str),
        ).fetchone()
        if row is None:
            conn.close()
            return {"success": False}
        chosen_bay = row["bay"]

    cur.execute(
        "UPDATE dock_slots SET booked = 1 WHERE date = ? AND time = ? AND bay = ?",
        (date_str, time_str, chosen_bay),
    )
    cur.execute(
        "INSERT INTO dock_reservations "
        "(shipment_id, carrier, driver_email, date, time, bay, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (shipment_id, carrier, driver_email, date_str, time_str, chosen_bay,
         datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return {"success": True, "bay": chosen_bay}


init_db()

"""
Mocked-but-functional tools used by the Booking Specialist agent.

Two independent tool sets:
  - GENERAL_TOOLS: check_availability / reserve_slot / send_booking_notification
    (customer-facing appointment booking — unchanged demo)
  - DOCK_TOOLS: check_dock_availability / reserve_dock_slot / send_driver_notification
    (GigaCorp warehouse dock-booking extension, triggered conceptually by a
    "shipment cleared customs" event: check warehouse calendar -> book a
    dock slot via a mock carrier API -> send a calendar invite to the driver)
"""

import os
import json
from datetime import datetime

import requests
from langchain_core.tools import tool

import db

# WEBHOOK_URL points at your Pipedream workflow's HTTP trigger URL.
# Set it as an env var (locally, or via .streamlit/secrets.toml) or as a
# Streamlit Community Cloud secret (Settings -> Secrets) - see the
# Pipedream setup notes in graph.py / the README for how to create it.
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")


# ---------------------------------------------------------------------------
# General appointment tools (existing demo)
# ---------------------------------------------------------------------------
@tool
def check_availability(date: str) -> str:
    """Check available appointment time slots for a given date.

    Args:
        date: The date to check, formatted as YYYY-MM-DD. Must already be
            resolved from any relative expression (e.g. "tomorrow") before
            calling this tool.
    """
    if not _is_valid_date(date):
        return json.dumps({"error": f"'{date}' is not a valid YYYY-MM-DD date."})

    slots = db.list_available_slots(date)
    if not slots:
        # distinguish "no such business day" vs "fully booked"
        any_slot = db.slot_exists(date, "09:00")
        if not any_slot:
            return json.dumps(
                {
                    "date": date,
                    "available_slots": [],
                    "note": "This date is outside business hours or a Sunday.",
                }
            )
        return json.dumps({"date": date, "available_slots": [], "note": "Fully booked."})
    return json.dumps({"date": date, "available_slots": slots})


@tool
def reserve_slot(date: str, time: str, email: str) -> str:
    """Reserve an appointment slot for a customer.

    Args:
        date: Appointment date, formatted as YYYY-MM-DD.
        time: Appointment time, formatted as HH:MM in 24-hour time (e.g. "14:00").
        email: Customer's email address to confirm the booking.
    """
    if not _is_valid_date(date):
        return json.dumps({"success": False, "error": f"'{date}' is not a valid YYYY-MM-DD date."})
    if "@" not in email:
        return json.dumps({"success": False, "error": f"'{email}' does not look like a valid email."})

    success = db.reserve(date, time, email)
    if not success:
        alternatives = db.list_available_slots(date)
        return json.dumps(
            {
                "success": False,
                "error": f"The {time} slot on {date} is unavailable.",
                "alternative_slots": alternatives,
            }
        )
    return json.dumps({"success": True, "date": date, "time": time, "email": email})


@tool
def send_booking_notification(email: str, details: str) -> str:
    """Send a mock booking confirmation notification (email/WhatsApp) via webhook.

    Args:
        email: Customer's email address.
        details: Human-readable booking details to include in the notification
            (e.g. "Appointment confirmed for 2026-07-10 at 14:00").
    """
    payload = {
        "to": email,
        "message": details,
        "sent_at": datetime.now().isoformat(),
    }
    return _dispatch_webhook(payload)


# ---------------------------------------------------------------------------
# Warehouse dock-booking tools (GigaCorp logistics extension)
# ---------------------------------------------------------------------------
@tool
def check_dock_availability(date: str) -> str:
    """Check available warehouse dock slots (across all loading bays) for a given date.

    Args:
        date: The date to check, formatted as YYYY-MM-DD. Must already be
            resolved from any relative expression (e.g. "tomorrow") before
            calling this tool.
    """
    if not _is_valid_date(date):
        return json.dumps({"error": f"'{date}' is not a valid YYYY-MM-DD date."})

    slots = db.list_available_dock_slots(date)
    if not slots:
        if not db.dock_date_exists(date):
            return json.dumps(
                {
                    "date": date,
                    "available_slots": [],
                    "note": "This date is outside the scheduling window "
                    "(dock slots are only seeded for the next 14 days).",
                }
            )
        return json.dumps(
            {"date": date, "available_slots": [], "note": "All dock bays are fully booked."}
        )
    return json.dumps({"date": date, "available_slots": slots})


@tool
def reserve_dock_slot(date: str, time: str, shipment_id: str, carrier: str, driver_email: str) -> str:
    """Reserve a warehouse dock slot for an inbound/outbound shipment via the
    (mock) carrier API. Books the first open bay in that time window.

    Args:
        date: Dock appointment date, formatted as YYYY-MM-DD.
        time: Dock appointment time, formatted as HH:MM in 24-hour time (e.g. "14:00").
        shipment_id: The shipment identifier this dock slot is for (e.g. "SHP-48213").
        carrier: Name of the carrier/trucking company delivering or collecting the shipment.
        driver_email: Driver's (or dispatcher's) email address to send the calendar invite to.
    """
    if not _is_valid_date(date):
        return json.dumps({"success": False, "error": f"'{date}' is not a valid YYYY-MM-DD date."})
    if "@" not in driver_email:
        return json.dumps(
            {"success": False, "error": f"'{driver_email}' does not look like a valid email."}
        )

    result = db.reserve_dock_slot(date, time, shipment_id, carrier, driver_email)
    if not result["success"]:
        alternatives = db.list_available_dock_slots(date)
        return json.dumps(
            {
                "success": False,
                "error": f"No dock bay available at {time} on {date}.",
                "alternative_slots": alternatives,
            }
        )
    return json.dumps(
        {
            "success": True,
            "date": date,
            "time": time,
            "bay": result["bay"],
            "shipment_id": shipment_id,
            "carrier": carrier,
            "driver_email": driver_email,
        }
    )


@tool
def send_driver_notification(driver_email: str, details: str) -> str:
    """Send a mock calendar invite / confirmation to the driver via webhook,
    simulating the 'Sends Calendar Invite to Local Driver' step.

    Args:
        driver_email: Driver's (or dispatcher's) email address.
        details: Human-readable dock booking details (e.g. "Dock Bay-2 reserved
            for shipment SHP-48213 on 2026-07-12 at 14:00, carrier: FastFreight Co.").
    """
    payload = {
        "to": driver_email,
        "message": details,
        "sent_at": datetime.now().isoformat(),
    }
    return _dispatch_webhook(payload)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _dispatch_webhook(payload: dict) -> str:
    """Send the structured notification to the configured Pipedream webhook,
    or log it locally if WEBHOOK_URL isn't set.

    NOTE: this reads the module-level WEBHOOK_URL, which is resolved once at
    import time from the environment. It intentionally does NOT reach out to
    Streamlit's st.secrets or any other secret manager here - secret-loading
    happens once, at startup, in app.py (which copies whatever it finds into
    os.environ), so this file works the same whether it's running on
    Streamlit Community Cloud, locally, or anywhere else `WEBHOOK_URL` is
    exported.
    """
    webhook_url = os.environ.get("WEBHOOK_URL", WEBHOOK_URL)

    if not webhook_url:
        print(f"[SIMULATION LOG] Webhook payload: {payload}")
        return "Notification logged locally (WEBHOOK_URL not configured)."

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code in (200, 201):
            return "Notification sent successfully via Pipedream webhook."
        return f"Webhook endpoint returned status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Failed to reach notification webhook: {str(e)}"


def _is_valid_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


GENERAL_TOOLS = [check_availability, reserve_slot, send_booking_notification]
DOCK_TOOLS = [check_dock_availability, reserve_dock_slot, send_driver_notification]

# Kept for backwards compatibility with any code importing ALL_TOOLS directly.
ALL_TOOLS = GENERAL_TOOLS

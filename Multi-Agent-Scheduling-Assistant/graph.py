"""
Multi-agent scheduling workflow built with LangGraph.

Flow (same shape for both modes):
    START -> triage
    triage --(general query)--> END  (direct answer added to messages)
    triage --(booking intent)--> booking_agent
    booking_agent --(wants to call a tool)--> booking_tools --> booking_agent (loop)
    booking_agent --(done)--> END

State persists per `thread_id` using LangGraph's SqliteSaver, so
conversation + booking-in-progress survive a page refresh.

Two modes share this exact state-machine shape but differ in tools and
system-prompt persona:

  "appointment" (default) - customer-facing appointment booking. The
      Booking Specialist collects date/time/email and uses
      check_availability / reserve_slot / send_booking_notification.

  "dock" - GigaCorp warehouse dock-booking extension. Models:
      [Shipment Clears Customs] -> [Check Warehouse Calendar]
        -> [Book Dock Slot via Carrier API] -> [Send Calendar Invite to Driver]
      The Booking Specialist collects date/time/shipment_id/carrier/
      driver_email and uses check_dock_availability / reserve_dock_slot /
      send_driver_notification.

Each mode gets its own checkpoint DB and its own thread_id namespace so
the two demos never share conversation state.
"""

import os
from datetime import datetime
from typing import Annotated, Literal, TypedDict

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:
    if "model_json_schema" in str(e):
        raise ImportError(
            "Your installed langchain-core is too old for the installed "
            "langchain-google-genai (langchain-google-genai 4.x needs "
            "langchain-core>=1.2.5). This happens when dependencies are "
            "installed/upgraded out of order or a stale environment is "
            "reused. Fix it with:\n"
            '    pip install --upgrade "langchain-core>=1.2.5,<2.0.0" '
            '"langchain-google-genai>=4.0.0,<5.0.0"\n'
            "then restart the app. On Streamlit Community Cloud, this is "
            "prevented by pinning the whole LangChain stack together in "
            "requirements.txt (already done in this repo) - if you still "
            "see this error there, use the app's 'Reboot app' option to "
            "force a clean dependency reinstall from requirements.txt, or "
            "'Clear cache' if that doesn't help."
        ) from e
    raise
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from tools import GENERAL_TOOLS, DOCK_TOOLS

# Free-tier-eligible Gemini model as of mid-2026. If Google renames/retires
# this, swap in whatever ai.google.dev/gemini-api/docs/models currently
# lists as free-tier Flash/Flash-Lite.
LLM_MODEL = "gemini-3.1-flash-lite"

MODES = ("appointment", "dock")


class SchedulingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str  # "general" | "booking"


def get_llm(temperature=0):
    """Build the Gemini chat client.

    Reads GOOGLE_API_KEY from the environment. This module never fetches
    secrets itself (no Kaggle/other secret-manager coupling) - whatever
    process starts the app (app.py, a notebook cell, a test) is responsible
    for exporting GOOGLE_API_KEY before this is called.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and export it, e.g.\n"
            "  export GOOGLE_API_KEY=your-key-here"
        )
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=temperature, google_api_key=api_key)


# --------------------------------------------------------------------------
# Triage Agent
# --------------------------------------------------------------------------
class TriageDecision(BaseModel):
    route: Literal["general", "booking"] = Field(
        description=(
            "'booking' if the user wants to schedule, check, reschedule, or "
            "book a slot, or is continuing an in-progress booking (e.g. "
            "providing a date/time/email/shipment detail). 'general' for "
            "anything else (greetings, questions about the business, small talk)."
        )
    )
    direct_reply: str = Field(
        description=(
            "If route is 'general', a short, friendly direct reply to the "
            "user. If route is 'booking', leave this empty."
        )
    )


TRIAGE_SYSTEM_BY_MODE = {
    "appointment": (
        "You are the Triage Agent for a scheduling assistant. Decide whether "
        "the latest user message should be handled by you directly ('general') "
        "or routed to the Booking Specialist ('booking'). Route to 'booking' for "
        "any scheduling, availability, rescheduling, or booking-related intent, "
        "including short follow-up replies that supply a date, time, or email "
        "during an in-progress booking. Consider the full conversation."
    ),
    "dock": (
        "You are the Triage Agent for GigaCorp's warehouse dock-booking "
        "assistant. Decide whether the latest user message should be handled "
        "by you directly ('general') or routed to the Booking Specialist "
        "('booking'). Route to 'booking' for any dock scheduling, availability, "
        "rescheduling request, a shipment-clearance event that needs a dock "
        "slot booked, or a short follow-up reply supplying a date, time, "
        "shipment ID, carrier, or driver email during an in-progress booking. "
        "Consider the full conversation."
    ),
}

TRIAGE_FALLBACK_REPLY_BY_MODE = {
    "appointment": (
        "Hi! I can help you check availability or book an appointment "
        "whenever you're ready."
    ),
    "dock": (
        "Hi! I'm the GigaCorp warehouse dock-booking assistant. Let me know "
        "a shipment that's cleared customs (or a date) and I can check dock "
        "availability and book a slot."
    ),
}


def make_triage_node(mode: str):
    system_text = TRIAGE_SYSTEM_BY_MODE[mode]
    fallback_reply = TRIAGE_FALLBACK_REPLY_BY_MODE[mode]

    def triage_node(state: SchedulingState):
        llm = get_llm()
        structured_llm = llm.with_structured_output(TriageDecision)

        system = SystemMessage(content=system_text)
        decision: TriageDecision = structured_llm.invoke([system] + state["messages"])

        if decision.route == "general":
            reply = decision.direct_reply or fallback_reply
            return {"messages": [AIMessage(content=reply)], "route": "general"}

        return {"route": "booking"}

    return triage_node


def route_after_triage(state: SchedulingState):
    return "booking_agent" if state["route"] == "booking" else END


# --------------------------------------------------------------------------
# Booking Specialist
# --------------------------------------------------------------------------
APPOINTMENT_SYSTEM_PROMPT = """You are the Booking Specialist for a scheduling assistant.
Today's date is {today} ({weekday}).

Your job:
1. Collect three pieces of information from the user: a DATE, a TIME, and an
   EMAIL address. Ask for whatever is missing, one question at a time.
2. Whenever the user gives a relative date expression ("tomorrow", "next
   Monday", "this Friday"), resolve it yourself into an absolute YYYY-MM-DD
   date using today's date above BEFORE calling any tool. Never pass a
   relative expression into a tool.
3. Once you have a candidate date, call `check_availability` to show the
   user real open slots if they haven't picked a specific time yet.
4. Once you have date + time + email, call `reserve_slot` to book it.
5. If `reserve_slot` fails (slot taken), do NOT give up or fail silently -
   look at the `alternative_slots` returned and proactively offer 2-3
   alternatives to the user, and ask them to pick one.
6. After a successful reservation, call `send_booking_notification` with a
   clear human-readable summary, then confirm the booking to the user in
   plain language (no need to mention the webhook mechanics).
7. Be concise and conversational. Never expose raw JSON to the user.
"""

DOCK_SYSTEM_PROMPT = """You are the Booking Specialist for GigaCorp's warehouse
dock-booking assistant. This automates: a shipment clears customs -> you check
warehouse dock calendar availability -> you book a dock slot via the (mock)
carrier API -> you send a calendar invite / confirmation to the driver.
Today's date is {today} ({weekday}).

Your job:
1. Collect five pieces of information: a DATE, a TIME, a SHIPMENT ID, the
   CARRIER name, and the DRIVER's (or dispatcher's) EMAIL. Ask for whatever
   is missing, one question at a time. If the user's message already looks
   like an automated customs-clearance event (it mentions a shipment ID and
   carrier and asks to book a dock slot), do not re-ask for information you
   already have - just confirm it back briefly and move straight to checking
   availability.
2. Whenever the user gives a relative date expression ("tomorrow", "next
   Monday", "this Friday"), resolve it yourself into an absolute YYYY-MM-DD
   date using today's date above BEFORE calling any tool. Never pass a
   relative expression into a tool.
3. Once you have a candidate date, call `check_dock_availability` to show
   real open dock slots (time + bay) if a specific time hasn't been chosen.
4. Once you have date + time + shipment_id + carrier + driver_email, call
   `reserve_dock_slot` to book it. You do not need to specify a bay - the
   system assigns the first open bay in that time window automatically.
5. If `reserve_dock_slot` fails (no bay available at that time), do NOT give
   up or fail silently - look at the `alternative_slots` returned and
   proactively offer 2-3 alternative time/bay combinations, and ask the user
   to pick one.
6. After a successful reservation, call `send_driver_notification` with a
   clear human-readable summary (include shipment ID, carrier, date, time,
   and assigned bay), then confirm the booking in plain language (no need to
   mention webhook mechanics).
7. Be concise and conversational. Never expose raw JSON to the user.
"""

BOOKING_SYSTEM_PROMPT_BY_MODE = {
    "appointment": APPOINTMENT_SYSTEM_PROMPT,
    "dock": DOCK_SYSTEM_PROMPT,
}

TOOLS_BY_MODE = {
    "appointment": GENERAL_TOOLS,
    "dock": DOCK_TOOLS,
}


def make_booking_agent_node(mode: str):
    tools = TOOLS_BY_MODE[mode]
    prompt_template = BOOKING_SYSTEM_PROMPT_BY_MODE[mode]

    def booking_agent_node(state: SchedulingState):
        llm = get_llm().bind_tools(tools)
        now = datetime.now()
        system = SystemMessage(
            content=prompt_template.format(
                today=now.strftime("%Y-%m-%d"), weekday=now.strftime("%A")
            )
        )
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    return booking_agent_node


def route_after_booking_agent(state: SchedulingState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "booking_tools"
    return END


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------
def build_graph(mode: str = "appointment", checkpointer=None):
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}', expected one of {MODES}")

    tools = TOOLS_BY_MODE[mode]
    graph = StateGraph(SchedulingState)

    graph.add_node("triage", make_triage_node(mode))
    graph.add_node("booking_agent", make_booking_agent_node(mode))
    graph.add_node("booking_tools", ToolNode(tools))

    graph.set_entry_point("triage")
    graph.add_conditional_edges(
        "triage", route_after_triage, {"booking_agent": "booking_agent", END: END}
    )
    graph.add_conditional_edges(
        "booking_agent",
        route_after_booking_agent,
        {"booking_tools": "booking_tools", END: END},
    )
    graph.add_edge("booking_tools", "booking_agent")

    return graph.compile(checkpointer=checkpointer)


_open_checkpointer_ctxs = []  # keep SqliteSaver context managers alive for app lifetime


def get_persistent_graph(mode: str = "appointment"):
    """Graph with SQLite-backed checkpointing so history survives refreshes.

    Each mode gets its own checkpoint file so the "appointment" and "dock"
    demos never share conversation/thread state.
    """
    from pathlib import Path

    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}', expected one of {MODES}")

    db_filename = "checkpoints.sqlite" if mode == "appointment" else f"checkpoints_{mode}.sqlite"
    db_path = Path(__file__).parent / db_filename
    conn_ctx = SqliteSaver.from_conn_string(str(db_path))
    checkpointer = conn_ctx.__enter__()  # keep the connection open for app lifetime
    _open_checkpointer_ctxs.append(conn_ctx)
    return build_graph(mode=mode, checkpointer=checkpointer)

"""
Multi-Agent Scheduling Assistant — Streamlit front end.

Two demo modes, selectable in the sidebar:
  - "General Appointment": the original customer-facing booking demo.
  - "Warehouse Dock Booking": GigaCorp logistics extension modelling
        [Shipment Clears Customs] -> [Check Warehouse Calendar]
          -> [Book Dock Slot via Carrier API] -> [Send Calendar Invite to Driver]
    Includes a "Simulate Customs Clearance" button that fires a synthetic
    automated-trigger event (as if an external customs API called this app)
    and drives the agent through the full chain without any typing. Chat
    input still works in this mode too, for manual "book a dock slot for
    shipment X" requests.

Run locally:
    export GOOGLE_API_KEY=your-free-gemini-api-key
    export WEBHOOK_URL=https://your-pipedream-endpoint.m.pipedream.net   # optional
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this whole folder (app.py, db.py, tools.py, graph.py,
       drivers.txt, requirements.txt) to a GitHub repo.
    2. On share.streamlit.io, create a new app pointing at app.py in that repo.
    3. In the app's Settings -> Secrets, add (TOML format):

           GOOGLE_API_KEY = "your-free-gemini-api-key"
           WEBHOOK_URL = "https://your-pipedream-endpoint.m.pipedream.net"

       (WEBHOOK_URL is optional - notifications are simulated/logged if unset.)
    4. Deploy. No launcher script, tunnel, or extra setup needed - Streamlit
       Cloud runs `streamlit run app.py` for you directly.
"""

import os
import random
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from graph import get_persistent_graph

# ---------------------------------------------------------------------------
# Secret loading: works on Streamlit Community Cloud *and* everywhere else.
#
# Streamlit Cloud surfaces secrets configured in the dashboard through
# `st.secrets`, not through OS environment variables. Locally (or on any
# other host), people typically just export real env vars instead. We check
# both, preferring an already-set env var, and copy anything found in
# `st.secrets` into `os.environ` so the rest of the codebase (tools.py,
# graph.py) can keep doing simple `os.environ.get(...)` lookups without
# needing to know which host it's running on.
# ---------------------------------------------------------------------------
def _load_secret_into_env(var_name: str) -> None:
    if os.environ.get(var_name):
        return  # already set (e.g. real env var / .env / docker secret)
    try:
        value = st.secrets.get(var_name)  # st.secrets is dict-like; no-op if no secrets.toml
        if value:
            os.environ[var_name] = value
    except Exception:
        # No secrets.toml configured (e.g. running purely off env vars) -
        # that's fine, downstream code handles a missing GOOGLE_API_KEY/
        # WEBHOOK_URL gracefully (a blocking error for the former, silent
        # simulation logging for the latter).
        pass


_load_secret_into_env("GOOGLE_API_KEY")
_load_secret_into_env("WEBHOOK_URL")

st.set_page_config(page_title="Scheduling Assistant", page_icon="📅", layout="centered")

MODE_LABELS = {
    "appointment": "General Appointment",
    "dock": "Warehouse Dock Booking",
}
LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}

# Sample data used to generate a plausible "customs cleared" event
SAMPLE_CARRIERS = ["FastFreight Co.", "BlueLine Logistics", "Meridian Transport", "Coastal Haulage"]

# Driver directory lives in a plain text "database" file instead of being
# hardcoded here. Format: one driver per line, "Full Name|email@example.com".
# Blank lines and lines starting with '#' are ignored.
#
# Resolved relative to this file (not the process's cwd), so it's found
# correctly regardless of where `streamlit run` is invoked from - Streamlit
# Community Cloud runs the app from the repo root, which is usually also
# this file's directory, but this keeps things robust either way.
DRIVERS_FILE = Path(__file__).resolve().parent / "drivers.txt"

def load_drivers(path: Path = DRIVERS_FILE) -> list[tuple[str, str]]:
    """Read (name, email) pairs from the drivers txt file.

    Returns an empty list (rather than raising) if the file is missing or
    empty, so the UI can show a clear warning instead of crashing.
    """
    if not path.exists():
        return []

    drivers = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            continue  # malformed line - skip rather than crash the app
        name, _, email = line.partition("|")
        name, email = name.strip(), email.strip()
        if name and "@" in email:
            drivers.append((name, email))
    return drivers


@st.cache_resource(show_spinner=False)
def load_graph(mode: str):
    return get_persistent_graph(mode=mode)


def check_api_key():
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error(
            "Missing GOOGLE_API_KEY. Get a free key at "
            "https://aistudio.google.com/apikey and set it as an environment "
            "variable (`export GOOGLE_API_KEY=...`) locally, or as a "
            "Streamlit secret named GOOGLE_API_KEY (Settings -> Secrets) "
            "if this app is deployed on Streamlit Community Cloud."
        )
        st.stop()


check_api_key()

# --------------------------------------------------------------------------
# Mode selection (sidebar) - persisted in URL query params alongside thread_id
# --------------------------------------------------------------------------
params = st.query_params
mode = params.get("mode", "appointment")
if mode not in MODE_LABELS:
    mode = "appointment"

with st.sidebar:
    st.header("Demo mode")
    selected_label = st.radio(
        "Choose a scenario",
        options=list(MODE_LABELS.values()),
        index=list(MODE_LABELS.keys()).index(mode),
        label_visibility="collapsed",
    )
    selected_mode = LABEL_TO_MODE[selected_label]
    if selected_mode != mode:
        # Switching modes starts a fresh thread for that mode's namespace
        st.query_params["mode"] = selected_mode
        st.query_params["thread_id"] = str(uuid.uuid4())[:8]
        st.rerun()

graph = load_graph(mode)

if mode == "appointment":
    st.title("📅 Scheduling Assistant")
    st.caption(
        "A Triage Agent routes you to a Booking Specialist that checks real "
        "availability, resolves dates like 'tomorrow', and negotiates "
        "alternatives if a slot is taken."
    )
else:
    st.title("🚛 GigaCorp Warehouse Dock Booking")
    st.caption(
        "Models: shipment clears customs → agent checks the warehouse dock "
        "calendar → books a dock slot via a mock carrier API → sends a "
        "calendar invite to the driver. Negotiates alternative bays/times "
        "automatically if a slot is taken."
    )

# --------------------------------------------------------------------------
# Thread persistence: the thread_id is what LangGraph's SqliteSaver keys
# state on. We keep it (and the mode) in the URL query params so a page
# refresh reloads the exact same conversation from the mode's checkpoint DB.
# --------------------------------------------------------------------------
if "thread_id" not in params:
    new_id = str(uuid.uuid4())[:8]
    st.query_params["thread_id"] = new_id
    st.query_params["mode"] = mode
    thread_id = new_id
else:
    thread_id = params["thread_id"]

config = {"configurable": {"thread_id": thread_id}}

with st.sidebar:
    st.header("Session")
    st.code(thread_id, language=None)
    st.caption(
        "This ID identifies your conversation thread. Bookmark the URL "
        "(or note this ID) to resume this exact conversation later — "
        "state is saved via LangGraph's SqliteSaver, separately per mode."
    )
    if st.button("Start a new conversation"):
        st.query_params["thread_id"] = str(uuid.uuid4())[:8]
        st.query_params["mode"] = mode
        st.rerun()

    if not os.environ.get("WEBHOOK_URL"):
        st.info(
            "WEBHOOK_URL not set — notifications will be simulated (logged) "
            "instead of actually POSTed. Set it to your Pipedream workflow's "
            "HTTP trigger URL to see real notifications delivered (see setup "
            "notes below)."
        )
        with st.expander("Pipedream setup"):
            st.markdown(
                "1. In Pipedream, create a new workflow.\n"
                "2. Add an **HTTP / Webhook** trigger — this gives you a "
                "unique URL like `https://xxxxx.m.pipedream.net`.\n"
                "3. Copy that URL into `WEBHOOK_URL` (env var or "
                "`.streamlit/secrets.toml` locally, or a Streamlit "
                "Community Cloud secret named `WEBHOOK_URL` if deployed "
                "there).\n"
                "4. Add a step after the trigger to route the payload "
                "somewhere useful — e.g. **Send Email**, **Slack: Send "
                "Message**, or **Twilio: Send SMS** — and map `to` and "
                "`message` from the incoming JSON body (`steps.trigger.event.body.to` "
                "and `...body.message`).\n"
                "5. Deploy the workflow. Every booking confirmation this app "
                "sends will now show up wherever step 4 routes it."
            )


def run_turn(human_text: str):
    """Invoke the graph with a new human message and render the result."""
    with st.chat_message("user"):
        st.markdown(human_text)

    with st.chat_message("assistant"):
        with st.spinner("Working on it..."):
            result = graph.invoke(
                {"messages": [HumanMessage(content=human_text)]}, config=config
            )
            final_messages = result["messages"]
            reply = ""
            for m in reversed(final_messages):
                if isinstance(m, AIMessage) and m.content:
                    content = m.content
                    if isinstance(content, list) and len(content) > 0:
                        # Gemini can return content as a list of blocks (text
                        # and/or tool-use dicts). Take the first text block
                        # found anywhere in the list, not just index 0 -
                        # a tool_use block can legitimately come first.
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type", "text") == "text" and block.get("text"):
                                    text_parts.append(block["text"])
                            elif isinstance(block, str):
                                text_parts.append(block)
                        reply = "\n".join(text_parts)
                    else:
                        reply = content
                    if reply:
                        break
        st.markdown(reply or "_(no response generated)_")

        tool_calls_made = [m for m in final_messages if isinstance(m, ToolMessage)]
        if tool_calls_made:
            with st.expander("Tool activity this turn"):
                for tm in tool_calls_made:
                    st.code(tm.content, language="json")


# --------------------------------------------------------------------------
# Dock mode only: "Simulate Customs Clearance" automated-trigger button
# --------------------------------------------------------------------------
if mode == "dock":
    with st.sidebar:
        st.header("Automated trigger")
        st.caption(
            "Simulates the real-world flow: an external customs API notifies "
            "this app the moment a shipment clears customs, which "
            "automatically kicks off the booking agent — no manual chat "
            "input required."
        )
        drivers = load_drivers()
        if not drivers:
            st.warning(
                f"No drivers found in `{DRIVERS_FILE.name}`. Add lines in the "
                "form `Full Name|email@example.com` to enable the simulate "
                "button."
            )
        if st.button("🛃 Simulate Customs Clearance", type="primary", disabled=not drivers):
            shipment_id = f"SHP-{random.randint(10000, 99999)}"
            carrier = random.choice(SAMPLE_CARRIERS)
            # Driver is drawn at random from the drivers.txt "database" only -
            # it's the single source of truth for name/email pairs, so the
            # email that ends up in the notification always matches a real
            # row in that file (never an independently-generated value).
            driver_name, driver_email = random.choice(drivers)
            st.session_state["pending_trigger"] = (
                f"[Automated system event] Shipment {shipment_id} has just cleared "
                f"customs. Carrier: {carrier}. Driver: {driver_name} "
                f"({driver_email}). Please check warehouse dock availability for "
                f"tomorrow and book the earliest open dock slot for this shipment, "
                f"then notify the driver."
            )
            st.rerun()

# --------------------------------------------------------------------------
# Reload & render existing history for this thread from the checkpointer
# --------------------------------------------------------------------------
existing_state = graph.get_state(config)
history = existing_state.values.get("messages", []) if existing_state.values else []

for msg in history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    # ToolMessages and tool-call-only AIMessages are intentionally not
    # rendered directly to keep the chat clean.

# Handle a pending automated-trigger event queued by the sidebar button
if st.session_state.get("pending_trigger"):
    trigger_text = st.session_state.pop("pending_trigger")
    run_turn(trigger_text)
    st.rerun()

placeholder = (
    "Try: 'I'd like to book an appointment tomorrow'"
    if mode == "appointment"
    else "Try: 'Shipment SHP-12345 cleared customs, book a dock slot tomorrow'"
)
if prompt := st.chat_input(placeholder):
    run_turn(prompt)
    st.rerun()

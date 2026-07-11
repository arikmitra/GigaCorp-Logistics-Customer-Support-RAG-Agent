# Multi-Agent Scheduling Assistant

A LangGraph multi-agent scheduling assistant with a Streamlit front end, in
two modes:

- **General Appointment** — customer-facing appointment booking (date/time/email).
- **Warehouse Dock Booking** — a GigaCorp logistics demo modelling
  `Shipment Clears Customs → Check Warehouse Calendar → Book Dock Slot via
  Carrier API → Send Calendar Invite to Driver`, including a "Simulate
  Customs Clearance" button that drives the whole chain with one click.

Both modes share the same triage → booking-specialist → tools graph shape,
persisted per conversation via LangGraph's `SqliteSaver`.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit front end — chat UI, mode switcher, secrets loading |
| `graph.py` | LangGraph state machine (triage agent, booking agent, tool node) |
| `tools.py` | Tool implementations (availability check, reservation, webhook notify) |
| `db.py` | SQLite-backed mock scheduling database, seeded on import |
| `drivers.txt` | Driver directory (`Name\|email` per line) used by the dock-booking demo |
| `requirements.txt` | Pinned dependency set |
| `.streamlit/secrets.toml.example` | Template for local secrets |

There is **no launcher script** — Streamlit Community Cloud runs `app.py`
directly with `streamlit run app.py`. (An earlier Kaggle-only version of
this project used a separate `kaggle_launcher.py` to open an ngrok tunnel;
that's not needed or wanted here, since Streamlit Cloud already gives the
app a public URL.)

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (public or private — either works with
   Streamlit Cloud).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**,
   and point it at your repo with `app.py` as the main file.
3. Before (or after) the first deploy, open the app's **Settings → Secrets**
   and paste:

   ```toml
   GOOGLE_API_KEY = "your-free-gemini-api-key"
   WEBHOOK_URL = "https://your-pipedream-endpoint.m.pipedream.net"
   ```

   Get a free Gemini API key at <https://aistudio.google.com/apikey>.
   `WEBHOOK_URL` is optional — see [Pipedream setup](#pipedream-webhook-setup-optional)
   below. Without it, booking notifications are simulated (logged) instead
   of actually sent.
4. Deploy (or reboot, if you added secrets after the first deploy). Streamlit
   Cloud installs everything from `requirements.txt` into a fresh
   environment automatically — no manual pip steps needed.

That's it. The app is reachable at the `*.streamlit.app` URL Streamlit Cloud
assigns it.

### Updating the app

Every `git push` to the branch Streamlit Cloud is tracking triggers an
automatic redeploy. If you only changed secrets (not code), use the app's
**Reboot app** button in the dashboard instead.

### A note on persistence

Streamlit Cloud's filesystem is writable but **ephemeral** — it resets on
every redeploy/reboot and isn't shared across replicas. That means:

- `scheduling.db` (mock appointment/dock availability) reseeds itself from
  scratch on every restart, via `init_db()` in `db.py`. This is expected —
  it's demo data.
- `checkpoints.sqlite` / `checkpoints_dock.sqlite` (LangGraph conversation
  state) is also lost on restart. Conversations won't survive a redeploy.
  This is fine for a demo; if you need durable multi-session history across
  restarts, swap `SqliteSaver` in `graph.py` for a checkpointer backed by an
  external database (e.g. `PostgresSaver` pointed at a hosted Postgres
  instance) — the rest of the graph code doesn't need to change.

## Run locally

```bash
pip install -r requirements.txt

# Option A: plain env vars
export GOOGLE_API_KEY=your-free-gemini-api-key
export WEBHOOK_URL=https://your-pipedream-endpoint.m.pipedream.net   # optional

# Option B: local secrets file (mirrors Streamlit Cloud's format)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml with your real key

streamlit run app.py
```

Either option works — `app.py` checks `os.environ` first, then falls back to
`st.secrets`.

## Pipedream webhook setup (optional)

Notifications ("booking confirmed", "driver notified") are sent as a POST to
`WEBHOOK_URL`. Without one configured, they're just logged to the app's
console/logs instead — the demo still works end-to-end.

1. In [Pipedream](https://pipedream.com), create a new workflow.
2. Add an **HTTP / Webhook** trigger. This gives you a unique URL like
   `https://xxxxx.m.pipedream.net`.
3. Set that URL as `WEBHOOK_URL` (env var locally, or a Streamlit Cloud
   secret).
4. Add a step after the trigger to route the payload somewhere useful — e.g.
   **Send Email**, **Slack: Send Message**, or **Twilio: Send SMS** — mapping
   `to` and `message` from the incoming JSON body
   (`steps.trigger.event.body.to` and `...body.message`).
5. Deploy the workflow. Every booking confirmation this app sends now shows
   up wherever step 4 routes it.

## Dependency notes

`requirements.txt` pins `langchain-core`, `langchain-google-genai`, and
`langgraph` together deliberately. Installing `langchain-google-genai` 4.x
against an older, independently-resolved `langchain-core` produces:

```
ImportError: cannot import name 'model_json_schema' from 'langchain_core.utils.pydantic'
```

because `langchain-google-genai` 4.x needs `langchain-core>=1.2.5` for that
symbol to exist. Keeping all three packages in the same `requirements.txt`,
resolved together by pip in one pass, avoids this. If you ever hit this
error anyway (e.g. after manually editing `requirements.txt`), use
**Manage app → Reboot app** on Streamlit Cloud to force a clean reinstall,
or **Clear cache** if that isn't enough.

`langgraph-checkpoint-sqlite` is a separate PyPI package from `langgraph`
itself (it's not bundled) — it's what provides
`from langgraph.checkpoint.sqlite import SqliteSaver`, used in `graph.py`
for conversation persistence. It's listed explicitly in `requirements.txt`.

## Editing the driver directory

The dock-booking demo's "Simulate Customs Clearance" button picks a random
driver from `drivers.txt`, one per line as `Full Name|email@example.com`.
Blank lines and lines starting with `#` are ignored. Edit that file and
redeploy (or just edit it on Streamlit Cloud's built-in file editor, if
you're editing directly from a connected repo) to change the roster.

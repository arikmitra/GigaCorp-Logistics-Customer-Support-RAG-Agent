# GigaCorp Logistics — AI Agent Portfolio

Two standalone Streamlit + LangChain/LangGraph projects built around a
fictional logistics company ("GigaCorp"): a **RAG customer-support agent**
and a **multi-agent scheduling assistant**. Each lives in its own folder
with its own dependencies, README, and deployment — pick the one you want
and follow the link below.

| Project | What it does | Folder |
|---|---|---|
| 🗨️ **GigaCorp Logistics RAG Support Agent** | Chatbot that answers customer questions (shipping, returns, billing, etc.) using Retrieval-Augmented Generation over a local FAQ file, with conversational memory | [`GigaCorp-Logistics-RAG-Support-Agent/`](GigaCorp-Logistics-RAG-Support-Agent/) |
| 📅 **Multi-Agent Scheduling Assistant** | LangGraph triage → booking-specialist → tools agent that books general appointments or, in a logistics-specific mode, warehouse dock slots for shipments clearing customs | [`Multi-Agent-Scheduling-Assistant/`](Multi-Agent-Scheduling-Assistant/) |

Both are free to run — Google's Gemini free tier, local embeddings, SQLite,
and Streamlit Community Cloud hosting all have no-cost tiers, so nothing
here requires a paid subscription.

## Live demos

- RAG Support Agent: [gigacorp-logistics-customer-support-rag-agent...streamlit.app](https://gigacorp-logistics-customer-support-rag-agent-7vygynfutjgdgkau.streamlit.app/)
- Multi-Agent Scheduling Assistant: see that folder's README for deployment steps.

## Project 1 — GigaCorp Logistics RAG Support Agent

A Streamlit chat app that answers customer questions about GigaCorp using
RAG over a local FAQ file, and remembers earlier turns so follow-up
questions ("how much does *that* cost?") resolve correctly.

**Stack**: LangChain (LCEL) → Gemini (`gemini-2.5-flash`, free tier) for
generation, `sentence-transformers` (`all-MiniLM-L6-v2`) + FAISS for
retrieval, Pydantic structured output to track which FAQ facts were used,
Streamlit for the UI.

**Flow**: user question → condense into a standalone question using chat
history → FAISS retrieval (top-4 chunks) → Gemini answers from context (or
falls back to natural small talk for greetings/thanks) → response rendered
in the chat UI.

See [`GigaCorp-Logistics-RAG-Support-Agent/README.md`](GigaCorp-Logistics-RAG-Support-Agent/README.md)
for full setup, Kaggle instructions, deployment, and troubleshooting.

## Project 2 — Multi-Agent Scheduling Assistant

A LangGraph multi-agent scheduling assistant with a Streamlit front end, in
two modes:

- **General Appointment** — customer-facing appointment booking (date, time, email).
- **Warehouse Dock Booking** — a logistics-specific demo modelling
  `Shipment Clears Customs → Check Warehouse Calendar → Book Dock Slot via
  Carrier API → Send Calendar Invite to Driver`, with a one-click "Simulate
  Customs Clearance" button.

**Stack**: LangGraph (triage agent → booking-specialist agent → tool node,
persisted per conversation via `SqliteSaver`), Gemini for the agents,
SQLite for mock scheduling data, an optional Pipedream webhook for booking
notifications, Streamlit for the UI.

See [`Multi-Agent-Scheduling-Assistant/README.md`](Multi-Agent-Scheduling-Assistant/README.md)
for full setup, secrets configuration, webhook setup, and troubleshooting
(including a known `langchain-google-genai` auth quirk).

## Repository layout

```
.
├── GigaCorp-Logistics-RAG-Support-Agent/
│   ├── app.py                       # Streamlit app + LangChain RAG + memory
│   ├── data/gigacorp_faq_final.txt  # mock FAQ knowledge base (22 sections)
│   ├── requirements.txt
│   └── README.md
├── Multi-Agent-Scheduling-Assistant/
│   ├── app.py                       # Streamlit front end, mode switcher
│   ├── graph.py                     # LangGraph state machine
│   ├── tools.py                     # tool implementations
│   ├── db.py                        # SQLite mock scheduling database
│   ├── drivers.txt                  # driver directory for dock-booking demo
│   ├── secrets.toml.example
│   ├── requirements.txt
│   └── README.md
├── LICENSE                          # CC0 1.0 Universal (public domain)
└── README.md                        # this file
```

## Getting started

Each project has its own dependencies and virtual environment — don't mix
them. From the repo root:

```bash
# RAG support agent
cd GigaCorp-Logistics-RAG-Support-Agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GOOGLE_API_KEY
streamlit run app.py

# Multi-agent scheduling assistant (separate terminal/venv)
cd Multi-Agent-Scheduling-Assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp secrets.toml.example .streamlit/secrets.toml   # fill in GOOGLE_API_KEY
streamlit run app.py
```

Both apps need a free Gemini API key from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey).

## License

[CC0 1.0 Universal](LICENSE) — public domain. Use this code however you like.

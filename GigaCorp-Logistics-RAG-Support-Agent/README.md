Check app here - https://gigacorp-logistics-customer-support-rag-agent-7vygynfutjgdgkau.streamlit.app/

# GigaCorp Customer Support RAG Agent

A Streamlit chat app that answers customer questions about a fictional
company ("GigaCorp") using Retrieval-Augmented Generation over a local FAQ
file, and remembers earlier turns in the conversation so follow-up
questions resolve correctly.

## Architecture

```
User question
     │
     ▼
┌─────────────────────────┐
│ Condense step (LLM)      │  rewrites follow-up questions ("how much to
│ uses chat_history         │  ship there?") into standalone questions
└─────────────────────────┘  using prior turns
     │
     ▼
┌─────────────────────────┐
│ FAISS retriever           │  embeds the standalone question with
│ (sentence-transformers)   │  all-MiniLM-L6-v2 and returns top-4 chunks
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ Answer step (Gemini)      │  structured output (Pydantic): a natural
│ + chat_history + context  │  conversational answer, plus which FAQ line
└─────────────────────────┘  IDs were actually used (tracked, not shown)
     │
     ▼
Streamlit chat UI (plain conversational answer, no inline citation clutter)
```

- **LLM orchestration**: LangChain (LCEL runnables), Google's free-tier `gemini-2.5-flash` via `langchain-google-genai`. (`gemini-2.0-flash` was shut down by Google on June 1, 2026 — don't use it.)
- **Knowledge base / RAG**: `data/gigacorp_faq.txt` (22 sections, 150+ facts covering shipping, returns, hours, tiers, warranty, billing, order limits, cancellations, customs, gift cards, loyalty, bulk/wholesale, promotions, subscriptions, backorders, lost packages, account security, app support, accessibility, and sustainability) is parsed into one chunk per numbered FAQ line (e.g. `1.2`, `7.4`) with section metadata, embedded with a free local `sentence-transformers` model, and indexed in FAISS.
- **Structured output with Pydantic**: the answer step uses `llm.with_structured_output(SupportAnswer)`, where `SupportAnswer` is a Pydantic model with two fields — `answer` (the natural conversational reply, no bracket tags) and `used_line_ids` (which FAQ facts were actually relied on, or empty for general chit-chat). This keeps the visible answer clean while still tracking provenance internally; the "Sources used" UI is currently disabled but the data is there if you want to re-enable it (see `answer_question()` in `app.py`).
- **Conversational fallback**: the system prompt explicitly branches — general messages (greetings, thanks, small talk) get a warm natural reply instead of being forced through the knowledge base, while genuine FAQ questions are answered only from retrieved context.
- **Conversational memory**: `st.session_state.messages` stores full turn history for the session. Each turn, prior turns are converted to LangChain `HumanMessage`/`AIMessage` objects and (a) used to rewrite the current question into a standalone one before retrieval, and (b) passed again to the final answer step so Gemini has full conversational context.

## Setup

Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (no paid plan required).

```bash
git clone <this-repo>
cd assignment1-rag-support
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GOOGLE_API_KEY
export GOOGLE_API_KEY=your-free-gemini-api-key
streamlit run app.py
```

The first run downloads the local embedding model (~90 MB) and builds the
FAISS index; this is cached via `st.cache_resource` so subsequent runs are
fast. Embeddings automatically use GPU if one is available (e.g. a Kaggle
notebook with a T4), and fall back to CPU otherwise — no code changes
needed either way. Every component (Gemini free tier, local embeddings,
FAISS, pydantic, Streamlit) is free; nothing here requires a paid
subscription.

## Running on Kaggle

`app.py` auto-detects a Kaggle environment (it checks for `/kaggle/input/gigacorp-faq/gigacorp_faq.txt` and falls back to the local `data/` folder otherwise), and pulls `GOOGLE_API_KEY` from Kaggle Secrets automatically if present.

1. **Enable internet and GPU**: Notebook Settings (right sidebar) → **Internet: On**, **Accelerator: GPU T4 x1 or x2**.
2. **Attach the FAQ file as a dataset** (or upload it directly), then confirm the real mount path:
   ```python
   !ls -R /kaggle/input/
   ```
   Adjust the `_KAGGLE_DATA_PATH` constant near the top of `app.py` if your dataset slug differs from `gigacorp-faq`.
3. **Add your Gemini key via Kaggle Secrets**: Add-ons → Secrets → add `GOOGLE_API_KEY`. `app.py` reads it automatically:
   ```python
   try:
       from kaggle_secrets import UserSecretsClient
       os.environ.setdefault("GOOGLE_API_KEY", UserSecretsClient().get_secret("GOOGLE_API_KEY"))
   except ImportError:
       pass  # not on Kaggle
   ```
4. **Write and install**:
   ```python
   %%writefile app.py
   # ... paste the full app.py contents here ...
   ```
   ```python
   !pip install -q --no-cache-dir -r requirements.txt
   ```
   If installs hang or conflict, see **Handling dependency conflicts** below. You may need to restart the kernel after installing (`import os; os._exit(0)`, then rerun cells) before the new packages are importable.

5. **Expose the Streamlit app publicly with ngrok** (Kaggle doesn't expose ports directly):
   ```python
   !pip install -q pyngrok
   from pyngrok import ngrok
   from kaggle_secrets import UserSecretsClient

   # Store your ngrok token in Kaggle Secrets as NGROK_TOKEN - never hardcode it in a cell
   ngrok.set_auth_token(UserSecretsClient().get_secret("NGROK_TOKEN"))

   public_url = ngrok.connect(8501)
   print(public_url)

   get_ipython().system_raw("streamlit run app.py --server.port 8501 --server.headless true &")
   ```
   Get a free ngrok auth token at [dashboard.ngrok.com](https://dashboard.ngrok.com) (sign up, no paid plan required) and store it in **Kaggle Secrets**, not directly in a cell — tokens pasted into notebook cells can end up in version history or shared outputs. If a token is ever pasted in plaintext anywhere, treat it as compromised and regenerate it immediately.
   The free ngrok tier gives you a randomly-assigned URL each time you reconnect and will disconnect if the Kaggle session ends or goes idle, so this is meant for live demoing/testing, not the persistent hosted deployment required by the assignment — use Streamlit Community Cloud or Render for that (see below).

## Deploying (Streamlit Community Cloud — free tier)

1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, click **New app**.
3. Point it at your repo, branch, and `app.py`.
4. In **Advanced settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your-free-gemini-api-key-here"
   ```
5. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

(Hugging Face Spaces works the same way — create a Space with the Streamlit
SDK, upload these files, and add `GOOGLE_API_KEY` under Settings → Secrets.)

## Handling dependency conflicts in requirements.txt

`requirements.txt` pins exact versions so the app behaves predictably, but pinned versions can clash with packages a host already provides (Kaggle images ship pre-installed `torch`, `protobuf`, `grpcio`, `typing-extensions`, etc. at their own fixed versions).

If `pip install -r requirements.txt` fails or the resolver hangs:

1. **Try an unpinned install first** for just the LangChain/Google stack, and let pip resolve compatible versions itself rather than fighting the exact pins:
   ```bash
   pip install -q --no-cache-dir langchain langchain-community langchain-google-genai langchain-huggingface faiss-cpu sentence-transformers pydantic
   ```
2. **If a specific package conflicts** (commonly `protobuf`, `grpcio`, or `typing-extensions` fighting a host-provided version), remove just that package's `==x.y.z` pin from `requirements.txt` and reinstall — don't unpin everything at once, only what's actually conflicting.
3. **On Kaggle specifically**, restart the kernel after any fresh install (`import os; os._exit(0)`, then rerun cells) — new packages sometimes aren't importable in the same kernel session that installed them.
4. **`pydantic` version matters**: this app uses Pydantic v2 syntax (`BaseModel`, `Field`, structured output). If a conflicting install pulls in Pydantic v1, structured output calls (`with_structured_output`) will break — pin `pydantic>=2,<3` explicitly if you hit this.
5. As a last resort, isolate the app in its own virtual environment (`python -m venv venv`) or, on Kaggle, accept the versions Kaggle already has for shared packages (`torch`, `numpy`) rather than trying to force the exact pins in `requirements.txt` — those two rarely need to match exactly for this app to work correctly.

## Testing it

Try this sequence to see RAG + memory + natural conversational fallback together:

1. "Hi!" → general chit-chat, answered directly without touching the knowledge base.
2. "Do you ship to India?" → retrieves shipping facts and answers naturally (grounded in section 1, though not shown as an inline citation).
3. "How much does that cost and how long does it take?" → resolves "that" to India shipping using memory.
4. "What if I'm a Premium member, does that change anything?" → pulls in Premium shipping benefits from section 4.
5. "Can I order 50 units of one item?" → exercises the new Section 7 (order quantities & bulk purchases) content.
6. "Thanks, that's all!" → general chit-chat again, no knowledge base lookup.

## Files

- `app.py` — Streamlit app + LangChain RAG + memory logic + Pydantic structured output
- `data/gigacorp_faq.txt` — mock knowledge base (22 sections: shipping, returns, hours, tiers, warranty, billing, order limits, cancellations, customs, gift cards, loyalty, wholesale, promotions, subscriptions, backorders, lost packages, security, app support, accessibility, sustainability)
- `requirements.txt` — pinned dependencies (see conflict-handling notes above if installing on Kaggle or another host with pre-installed packages)
- `.env.example` — required environment variable template

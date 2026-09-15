# Review Genie

A conversational AI agent built with **Streamlit** and **LangChain** that answers questions using the ReAct (Reason + Act) framework. It decides on each turn whether to search a local FAISS vector store of Amazon product data, run a live web search through Tavily, or answer directly, and it remembers the conversation within a session.

## Features

- **Chat interface** with full conversation history, built on Streamlit's native chat components.
- **Two tools the agent can call**
  - `amazon_product_search`: semantic search over a prebuilt FAISS index of Amazon product data.
  - `search_tavily`: live web search via the Tavily API.
- **Session memory**: earlier turns are summarised with `ConversationSummaryMemory`, so the agent can refer back to what you said without the prompt growing unbounded. Memory is per browser session, so users never see each other's chats.
- **Transparent reasoning**: an optional expander under each reply shows the tools the agent called, the inputs it used, and what it observed.
- **Deployment ready**: reads API keys from a local `.env` file or from Streamlit secrets, so the same code runs on a laptop and on Streamlit Community Cloud.

## Project structure

```
.
├── app.py                        # Streamlit app (UI + agent wiring)
├── faiss_index/                  # Prebuilt vector store (must be committed)
│   ├── index.faiss
│   └── index.pkl
├── .streamlit/
│   ├── config.toml               # Streamlit server and theme settings
│   └── secrets.toml.example      # Template for local secrets
├── .env.example                  # Template for local environment variables
├── requirements.txt              # Pinned Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Prerequisites

- Python 3.12 or newer
- An [OpenAI API key](https://platform.openai.com/api-keys) (used for `gpt-4o-mini` and for embeddings)
- A [Tavily API key](https://app.tavily.com/) for web search

## Run locally

1. Clone the repository and enter the folder.

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install the dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Add your API keys. Copy `.env.example` to `.env` and fill in the values.

   ```
   OPENAI_API_KEY=sk-...
   TAVILY_API_KEY=tvly-...
   ```

   Alternatively copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. Both files are ignored by git.

5. Start the app.

   ```bash
   streamlit run app.py
   ```

   Streamlit opens the app at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub. Make sure `faiss_index/` is included and that `.env` and `.streamlit/secrets.toml` are **not** (the `.gitignore` handles this).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **New app**, choose this repository and branch, and set the main file path to `app.py`.
4. Under **Advanced settings**, pick Python 3.12 or 3.13, then open the **Secrets** box and paste:

   ```toml
   OPENAI_API_KEY = "sk-..."
   TAVILY_API_KEY = "tvly-..."
   ```

5. Click **Deploy**. The first start takes a minute or two while dependencies install and the vector store loads.

To update the deployed app, push to the same branch. Streamlit Cloud redeploys automatically.

### Deploying elsewhere

The app is a plain Streamlit script, so it also runs on any host that can execute `streamlit run app.py`, for example a Docker container or a VM. Set `OPENAI_API_KEY` and `TAVILY_API_KEY` as environment variables on that host.

## How it works

1. `load_agent_components()` runs once per server process (cached with `st.cache_resource`). It loads the FAISS index with OpenAI embeddings, builds the two tools, pulls the `hwchase17/react-chat` prompt from LangChain Hub (with an embedded fallback copy), and creates the ReAct agent on `gpt-4o-mini`.
2. Each browser session gets its own `AgentExecutor` with a `ConversationSummaryMemory`, stored in `st.session_state`.
3. When you send a message, the agent loops through Thought, Action, Observation steps (up to 20 iterations) until it produces a Final Answer, which is shown in the chat along with an optional breakdown of the tool calls.

## Configuration

| Setting | Where | Default |
| --- | --- | --- |
| Model | `MODEL` in `app.py` | `gpt-4o-mini` |
| Vector store path | `FAISS_DIR` in `app.py` | `./faiss_index` |
| Max agent iterations | `max_iterations` in `get_agent_executor()` | `20` |
| Theme and server options | `.streamlit/config.toml` | light theme, headless |

## Troubleshooting

- **`OPENAI_API_KEY is not set` banner**: the app could not find the key in `.env`, the environment, or Streamlit secrets. Add it and restart.
- **Web search fails**: check `TAVILY_API_KEY`. Product search still works without it.
- **`ModuleNotFoundError` for `langchain.agents`, `langchain.memory`, or `langchain.chains`**: LangChain 1.x moved these into the `langchain-classic` package. Install the pinned versions with `pip install -r requirements.txt`.
- **Deprecation warnings in the logs** for `hub.pull` and `ConversationSummaryMemory` are expected on LangChain 1.x and do not affect the app.
- **Slow first response** after deployment: the vector store and prompt are loaded on the first request and then cached.

## License

This project is released under the [MIT License](LICENSE).

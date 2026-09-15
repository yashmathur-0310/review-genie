"""
Review Genie - Agents & ReAct Framework (Streamlit version)

A conversational agent that answers questions using two tools:
  * amazon_product_search - a FAISS vector store of Amazon product data
  * search_tavily         - live web search via Tavily

Run with:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# LangChain imports
# LangChain 1.x moved AgentExecutor / create_react_agent / memory / hub into the
# `langchain-classic` package. Fall back to the 0.3.x locations if it is absent.
# --------------------------------------------------------------------------- #
try:
    from langchain_classic import hub
    from langchain_classic.agents import AgentExecutor, create_react_agent
    from langchain_classic.memory import ConversationSummaryMemory
except ImportError:  # langchain < 1.0
    from langchain import hub
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.memory import ConversationSummaryMemory

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL = "gpt-4o-mini"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
HUB_PROMPT = "hwchase17/react-chat"

# ReAct chat prompt (same text as hub prompt `hwchase17/react-chat`), used as a
# fallback if LangChain Hub cannot be reached at startup.
FALLBACK_REACT_CHAT_PROMPT = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""


def get_secret(*names: str):
    """Return the first value found for any of `names` in the environment or
    Streamlit secrets. Lets the app run locally (.env) and on Streamlit Cloud
    (secrets.toml) without code changes."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    try:
        for name in names:
            if name in st.secrets:
                return str(st.secrets[name])
    except Exception:  # no secrets.toml present
        pass
    return None


load_dotenv()
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "Open_API_Key")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY", "Tavily_API_Key")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# --------------------------------------------------------------------------- #
# Heavy, shareable resources (vector store, tools, agent) - built once per
# server process and shared across sessions.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading vector store and tools...")
def load_agent_components():
    # Vector store of Amazon product data
    embeddings = OpenAIEmbeddings()
    vector = FAISS.load_local(
        FAISS_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="amazon_search",
        description="Search for information about Amazon products.",
    )

    @tool
    def amazon_product_search(query: str) -> str:
        """Search for information about Amazon products.
        For any questions related to Amazon products, this tool must be used."""
        return retriever_tool.invoke(query)

    @tool
    def search_tavily(query: str):
        """Executes a web search using Tavily. Use this for general questions,
        current events, or anything not related to Amazon products.

        Parameters:
            query (str): The search query entered by the user.

        Returns:
            list: Search results containing answers, raw content, and images.
        """
        search_tool = TavilySearchResults(
            max_results=5,
            include_answer=True,
            include_raw_content=True,
            include_images=True,
        )
        return search_tool.invoke(query)

    tools = [search_tavily, amazon_product_search]

    # ReAct-style prompt with a chat_history slot so session memory is used
    try:
        prompt = hub.pull(HUB_PROMPT)
    except Exception:
        prompt = PromptTemplate.from_template(FALLBACK_REACT_CHAT_PROMPT)

    llm = ChatOpenAI(model=MODEL, temperature=0)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return agent, tools, llm


# --------------------------------------------------------------------------- #
# Per-session state: chat transcript + agent executor with summary memory
# --------------------------------------------------------------------------- #
def get_agent_executor() -> AgentExecutor:
    """Return this browser session's AgentExecutor, creating it on first use.
    Memory is per session, so different users do not share history."""
    if "agent_executor" not in st.session_state:
        agent, tools, llm = load_agent_components()
        # Older turns are summarised instead of kept verbatim, which keeps the
        # prompt short as the conversation grows.
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            input_key="input",
            output_key="output",
        )
        st.session_state.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=20,
            return_intermediate_steps=True,
        )
    return st.session_state.agent_executor


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pop("agent_executor", None)


def format_steps(intermediate_steps) -> list:
    """Convert (AgentAction, observation) tuples into plain dicts we can keep
    in session state and render later."""
    steps = []
    for action, observation in intermediate_steps:
        steps.append(
            {
                "tool": getattr(action, "tool", ""),
                "tool_input": str(getattr(action, "tool_input", "")),
                "log": str(getattr(action, "log", "")).strip(),
                "observation": str(observation),
            }
        )
    return steps


def render_steps(steps: list, key_prefix: str) -> None:
    if not steps:
        return
    plural = "s" if len(steps) != 1 else ""
    with st.expander(f"Agent reasoning ({len(steps)} tool call{plural})"):
        for i, step in enumerate(steps, 1):
            st.markdown(f"**Step {i}: `{step['tool']}`**")
            if step["log"]:
                st.code(step["log"], language="text")
            st.markdown(f"*Input:* `{step['tool_input']}`")
            observation = step["observation"]
            if len(observation) > 2000:
                observation = observation[:2000] + " ... [truncated]"
            st.text_area(
                "Observation",
                observation,
                height=150,
                key=f"{key_prefix}_obs_{i}",
                disabled=True,
                label_visibility="collapsed",
            )


def clean_answer(text: str) -> str:
    """The ReAct prompt shows the answer format inside ``` fences, so the model
    often wraps its reply in them too. Strip any stray fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    if text.endswith("```"):
        text = text[: -len("```")]
    return text.strip()


def chat_with_agent(user_input: str):
    """Send the user's message to the agent and return (answer, steps)."""
    executor = get_agent_executor()
    response = executor.invoke({"input": user_input})
    if isinstance(response, dict) and "output" in response:
        answer = clean_answer(str(response["output"]))
        return answer, format_steps(response.get("intermediate_steps", []))
    return "Error: Unexpected response format", []


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Review Genie", page_icon="🤖", layout="wide")

st.title("🤖 Review Genie - Agents & ReAct Framework")
st.caption("Enter your query below and get AI-powered responses with session memory.")

with st.sidebar:
    st.header("Settings")
    show_steps = st.toggle("Show agent reasoning", value=True)
    st.button("Clear conversation", on_click=reset_conversation, use_container_width=True)

    st.divider()
    st.subheader("About")
    st.markdown(
        f"""
        **Model:** `{MODEL}`

        **Tools available to the agent**
        - `amazon_product_search` - FAISS vector store of Amazon products
        - `search_tavily` - live web search

        Conversation memory is kept per browser session and summarised as it grows.
        """
    )

    st.divider()
    st.caption(
        "OpenAI key: " + ("found" if OPENAI_API_KEY else "missing")
        + "  \nTavily key: " + ("found" if TAVILY_API_KEY else "missing")
    )

if not OPENAI_API_KEY:
    st.error(
        "`OPENAI_API_KEY` is not set. Add it to a `.env` file next to `app.py` "
        "or to `.streamlit/secrets.toml`, then restart the app."
    )
    st.stop()
if not TAVILY_API_KEY:
    st.warning(
        "`TAVILY_API_KEY` is not set. Web search will fail; only Amazon product "
        "search will work."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_steps and message.get("steps"):
            render_steps(message["steps"], key_prefix=f"msg{idx}")

# Handle a new message
if user_input := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, steps = chat_with_agent(user_input)
            except Exception as exc:  # surface agent/tool failures in the UI
                answer, steps = f"Error: {exc}", []
        st.markdown(answer)
        if show_steps:
            render_steps(steps, key_prefix=f"msg{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "steps": steps}
    )

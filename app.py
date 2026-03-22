import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit.streamlit_callback_handler import (
    LLMThoughtLabeler,
    ToolRecord,
)
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from groq import BadRequestError, APIError

import os
from dotenv import load_dotenv

load_dotenv()


def to_langchain_messages(messages, max_messages=10):
    """Convert session messages to LangChain format. Keeps last N messages to avoid context overload."""
    if max_messages and len(messages) > max_messages:
        messages = messages[-max_messages:]
    result = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            result.append(HumanMessage(content=content))
        elif role in ("assistant", "assisstant"):
            result.append(AIMessage(content=content))
    return result

## Arxiv and wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_base = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_base = WikipediaQueryRun(api_wrapper=api_wrapper)

search_base = DuckDuckGoSearchRun(name="Search")


def safe_wiki(query: str = "") -> str:
    """Search Wikipedia. Query must be a non-empty string."""
    q = (query or "").strip()
    if not q:
        return "Error: A search query is required. Provide a topic or question to look up."
    return wiki_base.invoke({"query": q})


def safe_arxiv(query: str = "") -> str:
    """Search Arxiv for research papers. Query must be a non-empty string."""
    q = (query or "").strip()
    if not q:
        return "Error: A search query is required. Provide a topic or paper to search for."
    return arxiv_base.invoke({"query": q})


def safe_search(query: str = "") -> str:
    """Search the web. Query must be a non-empty string."""
    q = (query or "").strip()
    if not q:
        return "Error: A search query is required. Provide a search term."
    return search_base.invoke({"query": q})


wiki = StructuredTool.from_function(
    func=safe_wiki,
    name="wikipedia",
    description="Search Wikipedia for general knowledge. Input MUST be a 'query' string with the topic to look up (e.g., 'machine learning', 'Albert Einstein').",
)
arxiv = StructuredTool.from_function(
    func=safe_arxiv,
    name="arxiv",
    description="Search Arxiv.org for research papers. Input MUST be a 'query' string (e.g., 'transformer neural network', paper ID like '1706.03762').",
)
search = StructuredTool.from_function(
    func=safe_search,
    name="Search",
    description="Search the web for current information. Input MUST be a 'query' string with the search term.",
)


class CustomThoughtLabeler(LLMThoughtLabeler):
    """Custom labels for agent thoughts - clearer than default 'Thinking...'."""

    @staticmethod
    def get_initial_label() -> str:
        return "🔍 **Searching for answers...**"

    @staticmethod
    def get_tool_label(tool: ToolRecord, is_complete: bool) -> str:
        input_str = tool.input_str
        name = tool.name
        emoji = "✅" if is_complete else "🔍"
        if name == "_Exception":
            emoji = "⚠️"
            name = "Parsing error"
        idx = min(60, len(input_str))
        truncated = input_str[:idx] + ("..." if len(input_str) > idx else "")
        truncated = truncated.replace("\n", " ")
        return f"{emoji} **{name}:** {truncated}"


st.title("🔎 LangChain - Chat with search")
# """
# In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
# Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
# """

## Sidebar for settings
st.sidebar.title("Settings")
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.sidebar.error("GROQ_API_KEY not found. Add it to your .env file.")
search_mode = st.sidebar.radio(
    "Search mode",
    options=["Direct Search (recommended)", "Agent with tools"],
    index=0,
    help="Direct Search: always works. Agent: LLM chooses tools but Groq may fail; falls back to Direct Search on error.",
)
model_name = "openai/gpt-oss-120b"
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        if search_mode == "Direct Search (recommended)":
            # Bypass tool calling: run searches directly and pass results to LLM
            query = prompt.strip() or "current events"
            with st.status("Searching web, Wikipedia, and Arxiv...", expanded=False):
                try:
                    web_results = safe_search(query)
                    wiki_results = safe_wiki(query)
                    arxiv_results = safe_arxiv(query)
                except Exception as e:
                    web_results = wiki_results = arxiv_results = f"Search error: {e}"

            context = f"""Search results for "{query}":

WEB SEARCH:
{web_results}

WIKIPEDIA:
{wiki_results}

ARXIV (research papers):
{arxiv_results}
"""
            try:
                llm = ChatGroq(
                    groq_api_key=api_key,
                    model_name=model_name,
                    streaming=True,
                    temperature=temperature,
                )
                messages = [
                    SystemMessage(content="You are a helpful assistant. Use the search results below to answer the user's question. Cite the sources when relevant."),
                    HumanMessage(content=f"Question: {prompt}\n\n{context}\n\nBased on the above search results, please answer the user's question."),
                ]
                response = llm.invoke(messages)
                response_text = response.content if hasattr(response, "content") else str(response)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.write(response_text)
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, an error occurred while generating the response: {str(e)}",
                })
        else:
            # Agent mode with tool calling
            tools = [search, arxiv, wiki]
            st_cb = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=False,
                thought_labeler=CustomThoughtLabeler(),
            )
            messages = to_langchain_messages(st.session_state.messages)

            max_retries = 3
            retry_temp = temperature

            for attempt in range(max_retries):
                try:
                    llm_retry = ChatGroq(
                        groq_api_key=api_key,
                        model_name=model_name,
                        streaming=True,
                        temperature=retry_temp,
                    )
                    agent_retry = create_agent(model=llm_retry, tools=tools)
                    result = agent_retry.invoke(
                        {"messages": messages},
                        config={"callbacks": [st_cb]},
                    )
                    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
                    response = (
                        ai_messages[-1].content
                        if ai_messages
                        else getattr(
                            result["messages"][-1],
                            "content",
                            str(result["messages"][-1]),
                        )
                    )
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)
                    break
                except (BadRequestError, APIError) as e:
                    err_msg = str(e).lower()
                    is_tool_error = (
                        "tool call validation failed" in err_msg
                        or "failed to call a function" in err_msg
                        or "failed_generation" in err_msg
                        or "tool_use_failed" in err_msg
                        or "missing properties" in err_msg
                    )
                    if is_tool_error and attempt < max_retries - 1:
                        retry_temp = max(0.0, retry_temp - 0.2)
                        st.warning(f"Tool call failed, retrying with temperature={retry_temp}...")
                    else:
                        # Fallback to Direct Search when Groq tool calling fails
                        st.warning("Agent tool calling failed (known Groq limitation). Falling back to Direct Search...")
                        query = prompt.strip() or "current events"
                        with st.status("Searching web, Wikipedia, and Arxiv...", expanded=False):
                            try:
                                web_results = safe_search(query)
                                wiki_results = safe_wiki(query)
                                arxiv_results = safe_arxiv(query)
                            except Exception as search_err:
                                web_results = wiki_results = arxiv_results = f"Search error: {search_err}"

                        context = f"""Search results for "{query}":
WEB SEARCH: {web_results}
WIKIPEDIA: {wiki_results}
ARXIV: {arxiv_results}"""
                        try:
                            llm = ChatGroq(
                                groq_api_key=api_key,
                                model_name=model_name,
                                streaming=True,
                                temperature=temperature,
                            )
                            messages_fb = [
                                SystemMessage(content="You are a helpful assistant. Use the search results below to answer the user's question. Cite sources when relevant."),
                                HumanMessage(content=f"Question: {prompt}\n\n{context}\n\nAnswer the user's question based on the search results above."),
                            ]
                            response = llm.invoke(messages_fb)
                            response_text = response.content if hasattr(response, "content") else str(response)
                            st.session_state.messages.append({"role": "assistant", "content": response_text})
                            st.write(response_text)
                        except Exception as fallback_err:
                            st.error(f"API Error: {fallback_err}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Sorry, an error occurred: {str(fallback_err)}. Try Direct Search mode.",
                            })
                        break
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Sorry, an error occurred: {str(e)}",
                    })
                    break


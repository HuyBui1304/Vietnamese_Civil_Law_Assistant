"""
BƯỚC 9: Agent thông minh  (Bài 3 — LangChain + LangGraph)
==========================================================
9a. Tools: search_law, calculator, web_search (DuckDuckGo)
9b. ReAct Agent (LangChain AgentExecutor)
9c. LangGraph workflow: node Router → RAG / WebSearch / Direct
9d. Memory: ConversationBufferMemory (lịch sử hội thoại)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ================================================================ #
#  9a. Định nghĩa Tools                                            #
# ================================================================ #

def make_search_law_tool(search_fn):
    """Tạo LangChain Tool để tìm kiếm trong Bộ Luật Dân Sự.

    Args:
        search_fn: Hàm nhận (query_vec, top_k) → list[dict].
                   Wrapper đã nhúng embedding bên trong.

    Returns:
        langchain_core.tools.Tool
    """
    from langchain_core.tools import Tool

    def _search(query: str) -> str:
        from embedding import embed_texts
        q_vec = embed_texts([query])[0]
        results = search_fn(q_vec, top_k=3)
        if not results:
            return "Không tìm thấy thông tin liên quan."
        return "\n\n---\n\n".join(
            f"[{r['metadata'].get('id', '?')}]\n{r['text']}" for r in results
        )

    return Tool(
        name="search_law",
        func=_search,
        description=(
            "Tìm kiếm trong Bộ Luật Dân Sự Việt Nam 2015. "
            "Input: câu hỏi hoặc từ khóa pháp luật. "
            "Dùng khi cần tra cứu quy định, điều khoản cụ thể."
        ),
    )


def make_web_search_tool():
    """Tạo DuckDuckGo web search tool."""
    from langchain_community.tools import DuckDuckGoSearchRun
    tool = DuckDuckGoSearchRun()
    tool.description = (
        "Tìm kiếm thông tin trên internet. "
        "Dùng khi câu hỏi liên quan đến sự kiện hiện tại hoặc "
        "thông tin ngoài phạm vi Bộ Luật Dân Sự."
    )
    return tool


def make_calculator_tool():
    """Tạo tool tính toán đơn giản."""
    from langchain_core.tools import Tool

    def _calc(expr: str) -> str:
        try:
            result = eval(expr, {"__builtins__": {}})  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Lỗi tính toán: {e}"

    return Tool(
        name="calculator",
        func=_calc,
        description=(
            "Tính toán biểu thức số học đơn giản. "
            "Input: biểu thức Python hợp lệ (ví dụ: '2 + 2', '365 * 24'). "
            "Dùng khi cần tính thời hạn, ngày tháng, số tiền."
        ),
    )


# ================================================================ #
#  9b. ReAct Agent (LangChain)                                     #
# ================================================================ #

def build_react_agent(search_fn, gemini_api_key: str | None = None):
    """Xây dựng ReAct Agent với LangGraph prebuilt (modern API).

    Args:
        search_fn:      Hàm search vector store (nhận query_vec, top_k).
        gemini_api_key: Gemini API key. Mặc định lấy từ .env.

    Returns:
        Compiled LangGraph ReAct agent.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent as lg_create_react_agent

    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0.1,
    )

    tools = [
        make_search_law_tool(search_fn),
        make_web_search_tool(),
        make_calculator_tool(),
    ]

    system_prompt = (
        "Bạn là trợ lý pháp luật thông minh chuyên về Bộ Luật Dân Sự Việt Nam 2015. "
        "Trả lời bằng tiếng Việt. Dùng tool search_law để tra cứu luật, "
        "web_search khi cần thông tin ngoài luật dân sự, "
        "calculator để tính toán số liệu."
    )

    agent = lg_create_react_agent(llm, tools, prompt=system_prompt)
    return agent


# ================================================================ #
#  9c. LangGraph Workflow                                          #
# ================================================================ #

def build_langgraph_workflow(search_fn, gemini_api_key: str | None = None):
    """Xây dựng LangGraph workflow với 3 node: Router → RAG / WebSearch / Direct.

    Workflow:
        START → router → rag_node / web_node / direct_node → END

    Args:
        search_fn:      Hàm search vector store.
        gemini_api_key: Gemini API key.

    Returns:
        Compiled LangGraph app.
    """
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    from langchain_google_genai import ChatGoogleGenerativeAI
    from embedding import embed_texts

    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0.1,
    )

    class AgentState(TypedDict):
        question: str
        route: str
        context: str
        answer: str

    def router_node(state: AgentState) -> AgentState:
        q = state["question"].lower()
        law_keywords = ["điều", "luật", "pháp", "quyền", "nghĩa vụ",
                        "dân sự", "hợp đồng", "tài sản", "thừa kế"]
        if any(kw in q for kw in law_keywords):
            route = "rag"
        elif any(kw in q for kw in ["tìm kiếm", "internet", "tin tức", "hiện tại"]):
            route = "web"
        else:
            route = "direct"
        return {**state, "route": route}

    def rag_node(state: AgentState) -> AgentState:
        q_vec = embed_texts([state["question"]])[0]
        results = search_fn(q_vec, top_k=4)
        context = "\n\n---\n\n".join(r["text"] for r in results)
        prompt = (
            "Bạn là trợ lý pháp luật. Dựa vào ngữ cảnh sau, trả lời bằng tiếng Việt.\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {state['question']}\n\nTRẢ LỜI:"
        )
        answer = llm.invoke(prompt).content
        return {**state, "context": context, "answer": answer}

    def web_node(state: AgentState) -> AgentState:
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            results = DuckDuckGoSearchRun().run(state["question"])
        except Exception:
            results = "(Không tìm được kết quả web)"
        prompt = (
            f"Dựa vào kết quả tìm kiếm sau, trả lời bằng tiếng Việt:\n\n"
            f"KẾT QUẢ:\n{results}\n\n"
            f"CÂU HỎI: {state['question']}\n\nTRẢ LỜI:"
        )
        answer = llm.invoke(prompt).content
        return {**state, "context": results, "answer": answer}

    def direct_node(state: AgentState) -> AgentState:
        prompt = (
            "Bạn là trợ lý pháp luật. Trả lời ngắn gọn bằng tiếng Việt.\n"
            f"CÂU HỎI: {state['question']}\n\nTRẢ LỜI:"
        )
        answer = llm.invoke(prompt).content
        return {**state, "context": "", "answer": answer}

    def route_condition(state: AgentState) -> str:
        return state["route"]

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("web", web_node)
    graph.add_node("direct", direct_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_condition,
        {"rag": "rag", "web": "web", "direct": "direct"},
    )
    graph.add_edge("rag", END)
    graph.add_edge("web", END)
    graph.add_edge("direct", END)

    return graph.compile()


# ================================================================ #
#  Run                                                             #
# ================================================================ #

def run_agent(search_fn=None, faiss_index=None, faiss_chunks=None,
              qdrant_cl=None, force_rerun: bool = False):
    """
    Khởi tạo ReAct agent + LangGraph workflow và demo một câu hỏi.

    Args:
        search_fn:    Callable(query_vec, top_k) → list[dict].
                      Nếu None, sẽ tự tạo từ Qdrant.
        faiss_index:  FAISS index (dự phòng nếu không có search_fn).
        faiss_chunks: FAISS chunks list.
        qdrant_cl:    Qdrant client (dùng làm search mặc định).

    Returns:
        react_executor (AgentExecutor),
        graph_app      (compiled LangGraph)
    """
    print(f"\n{'='*55}")
    print(f"  BƯỚC 9: AGENT (ReAct + LangGraph)")
    print(f"{'='*55}\n")

    # Tạo search_fn từ Qdrant nếu chưa có
    if search_fn is None:
        if qdrant_cl is None:
            from vectordb import load_qdrant, search_qdrant
            qdrant_cl = load_qdrant()
        from vectordb import search_qdrant

        def search_fn(q_vec, top_k=5):
            return search_qdrant(q_vec, qdrant_cl, top_k=top_k)

    # Build ReAct Agent
    print("  Đang khởi tạo ReAct Agent...")
    react_executor = build_react_agent(search_fn)
    print("  ✓ ReAct Agent sẵn sàng\n")

    # Build LangGraph
    print("  Đang khởi tạo LangGraph Workflow...")
    graph_app = build_langgraph_workflow(search_fn)
    print("  ✓ LangGraph Workflow sẵn sàng\n")

    # Demo
    demo_q = "Quyền dân sự của cá nhân được quy định như thế nào?"
    print(f"  DEMO ReAct Agent:")
    print(f"  Câu hỏi: {demo_q}\n")
    try:
        result = react_executor.invoke({"messages": [("human", demo_q)]})
        content = result["messages"][-1].content
        answer = content[0]["text"] if isinstance(content, list) else content
        print(f"\n  → Câu trả lời: {answer[:300]}...")
    except Exception as e:
        print(f"  ⚠ ReAct demo lỗi: {e}")

    print(f"\n  DEMO LangGraph:")
    try:
        state = graph_app.invoke({"question": demo_q, "route": "", "context": "", "answer": ""})
        print(f"  Route: {state['route']}")
        print(f"  → Câu trả lời: {state['answer'][:300]}...")
    except Exception as e:
        print(f"  ⚠ LangGraph demo lỗi: {e}")

    print(f"\n  ✓ Bước 9 hoàn tất!\n")
    return react_executor, graph_app

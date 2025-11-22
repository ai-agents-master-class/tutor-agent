import re
import os
from langgraph.types import Command
from langchain_core.tools import tool
from firecrawl import FirecrawlApp, ScrapeOptions

# ---------------------------------------------------------------------
# transfer_to_agent
# - LangGraph에서 특정 agent 노드로 이동하기 위한 tool
# - classification_agent → teacher_agent / quiz_agent / feynman_agent 등
#   시스템 전체 플로우 전환의 핵심 도구
# ---------------------------------------------------------------------
@tool
def transfer_to_agent(agent_name: str):
    """
    지정한 agent로 학습 흐름을 전환합니다.

    Args:
        agent_name (str):
            이동할 agent 이름.
            가능한 값:
            - 'quiz_agent'
            - 'teacher_agent'
            - 'feynman_agent'
    """

    # Command 객체를 반환함으로써 LangGraph 내부 라우팅을 트리거
    return Command(
        goto=agent_name,        # 이동할 노드 이름
        graph=Command.PARENT,   # 현재 그래프의 부모 레벨에서 라우팅 실행
        update={
            "current_agent": agent_name,  # 상태 업데이트 (현재 agent 추적)
        },
    )


# ---------------------------------------------------------------------
# web_search_tool
# - Firecrawl API 기반 웹 검색 도구
# - 검색 결과를 Markdown 포맷으로 가져오고, 후처리(cleaning)까지 수행해
#   에이전트들이 바로 연구 자료로 활용할 수 있게 제공
# ---------------------------------------------------------------------
@tool
def web_search_tool(query: str):
    """
    웹 검색 도구.

    Args:
        query (str):
            웹에서 검색할 질의 문자열.

    Returns:
        Markdown 형식의 웹페이지 내용을 담은 검색 결과 리스트.
        각 결과는:
            - title: 페이지 제목
            - url: 원본 URL
            - markdown: 정제(clean)된 본문 텍스트
    """

    # Firecrawl API 초기화 (API KEY는 환경변수에서 읽음)
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    # 검색 실행
    response = app.search(
        query=query,
        limit=5,  # 검색 결과 최대 5개
        scrape_options=ScrapeOptions(
            formats=["markdown"],  # 페이지 내용을 Markdown으로 수집
        ),
    )

    # 성공 여부 확인
    if not response.success:
        return "Error using tool."

    cleaned_chunks = []

    # --------------------------------------------------------------
    # 검색 결과 후처리
    # - Firecrawl의 markdown 결과는 종종 markdown 링크/URL/줄바꿈 등이 많아
    #   모델 연구용으로 간결하게 정리(cleaning) 필요
    # --------------------------------------------------------------
    for result in response.data:

        title = result["title"]     # 페이지 제목
        url = result["url"]         # 페이지 URL
        markdown = result["markdown"]  # Markdown 텍스트

        # 1) 역슬래시(\) 또는 불필요한 줄바꿈 제거
        cleaned = re.sub(r"\\+|\n+", "", markdown).strip()

        # 2) Markdown 링크 및 URL 제거
        #    예: [text](url), https://example.com
        cleaned = re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)

        cleaned_result = {
            "title": title,
            "url": url,
            "markdown": cleaned,
        }

        cleaned_chunks.append(cleaned_result)

    return cleaned_chunks

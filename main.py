from dotenv import load_dotenv

# .env 환경 변수 로드 (Firecrawl API KEY 등)
load_dotenv()

from langgraph.graph import START, END, StateGraph, MessagesState

# 각 에이전트 불러오기
from agents.classification_agent import classification_agent
from agents.teacher_agent import teacher_agent
from agents.feynman_agent import feynman_agent
from agents.quiz_agent import quiz_agent


# ---------------------------------------------------------------------
# TutorState
# - 전체 멀티에이전트 시스템에서 유지되는 공통 state 정의
# - MessagesState를 상속하여 메시지 히스토리 + current_agent를 관리
# ---------------------------------------------------------------------
class TutorState(MessagesState):
    current_agent: str   # 현재 활성화된 에이전트 이름


# ---------------------------------------------------------------------
# router_check
# - START 지점 혹은 노드 전환 시, 현재 어떤 agent로 이동해야 하는지 결정하는 함수
# - transfer_to_agent tool이 state["current_agent"]를 변경하면
#   라우팅이 이 값에 따라 자동 재조정됨
# ---------------------------------------------------------------------
def router_check(state: TutorState):
    current_agent = state.get("current_agent", "classification_agent")
    return current_agent


# ---------------------------------------------------------------------
# LangGraph 그래프 생성
# ---------------------------------------------------------------------
graph_builder = StateGraph(TutorState)

# ---------------------------------------------------------------------
# 1) classification_agent 노드 정의
# - 이 에이전트는 '초기 라우터'이자 '학습자 프로파일러'
# - 이 에이전트에서 teacher / quiz / feynman 중 하나로 분기
# ---------------------------------------------------------------------
graph_builder.add_node(
    "classification_agent",
    classification_agent,
    destinations=(
        "quiz_agent",
        "teacher_agent",
        "feynman_agent",
    ),
)

# ---------------------------------------------------------------------
# 2) 개별 학습형 에이전트 노드 추가
# ---------------------------------------------------------------------
graph_builder.add_node("teacher_agent", teacher_agent)
graph_builder.add_node("feynman_agent", feynman_agent)
graph_builder.add_node("quiz_agent", quiz_agent)

# ---------------------------------------------------------------------
# 3) 라우팅 규칙 추가
# - START 지점에서 router_check 함수가 current_agent 값을 읽고
#   해당 agent로 이동하도록 라우팅 수행
# - 즉, 최초에는 classification_agent로 이동하며,
#   이후 transfer_to_agent 에 의해 current_agent가 변경되면 즉시 그 agent로 이동
# ---------------------------------------------------------------------
graph_builder.add_conditional_edges(
    START,
    router_check,
    [
        "teacher_agent",
        "feynman_agent",
        "classification_agent",
        "quiz_agent",
    ],
)

# ---------------------------------------------------------------------
# 4) classification_agent → END (그래프 종료)
# - classification_agent가 흐름을 끝내고 싶을 때 사용
# - 다른 agent들은 명시적으로 END로 연결되지 않으므로 계속 진행함
# ---------------------------------------------------------------------
graph_builder.add_edge("classification_agent", END)

# ---------------------------------------------------------------------
# 그래프 빌드 → compile()
# 최종 Tutor Graph 객체 생성
# ---------------------------------------------------------------------
graph = graph_builder.compile()

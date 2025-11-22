from langchain_core.tools import tool
from typing import Literal, List
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# Question 모델
# - 하나의 객관식 문제를 표현하는 데이터 구조
# - Pydantic으로 검증/직렬화를 지원
# ---------------------------------------------------------------------
class Question(BaseModel):
    question: str = Field(description="문제의 질문 텍스트")
    options: List[str] = Field(
        description="보기 4개(A, B, C, D). 반드시 4개여야 함."
    )
    correct_answer: str = Field(
        description="정답(반드시 options 중 하나와 동일해야 함)"
    )
    explanation: str = Field(
        description="정답이 맞는 이유와 오답이 틀린 이유를 포함한 상세 해설"
    )


# ---------------------------------------------------------------------
# Quiz 모델
# - 하나의 퀴즈 세트를 구성하는 데이터 구조
# - topic + 여러 Question 인스턴스로 구성됨
# ---------------------------------------------------------------------
class Quiz(BaseModel):
    topic: str = Field(description="퀴즈의 주요 주제")
    questions: List[Question] = Field(description="문제 리스트")


# ---------------------------------------------------------------------
# generate_quiz 함수
# - LangChain Tool로 등록되는 함수
# - 연구 텍스트(research_text)를 기반으로 구조화된 객관식 퀴즈 생성
# - Quiz 모델 형태로 리턴
# ---------------------------------------------------------------------
@tool
def generate_quiz(
    research_text: str,
    topic: str,
    difficulty: Literal["easy", "medium", "hard"],
    num_questions: int,
):
    """
    연구 정보를 기반으로 구조화된 객관식 퀴즈를 생성합니다.

    Args:
        research_text (str):
            - 해당 주제에 대한 연구 정보 또는 참고 텍스트
            - 웹 검색 결과(raw text), 연구 요약, 일반 텍스트 등 활용 가능
            - 비어 있어도 생성 가능하지만 정확도가 낮아질 수 있음

        topic (str):
            - 퀴즈 주제
            - 예: "파이썬 프로그래밍", "세계 2차 대전", "광합성"

        difficulty ("easy" | "medium" | "hard"):
            퀴즈 난이도
            - "easy": 기초 개념, 정의, 단순 사실 중심
            - "medium": 개념 관계, 응용력 평가
            - "hard": 분석·종합 수준의 심화 문제

        num_questions (int):
            생성할 문제 수 (1~30)
            일반적으로:
            - short: 3-5개
            - medium: 6-10개
            - long: 11-15개

    Returns:
        Quiz:
            - topic: 퀴즈 주제
            - questions: Question 객체 리스트
                - question: 문제 텍스트
                - options: 보기 4개
                - correct_answer: 정답
                - explanation: 상세 해설

    예시:
        research_info = "Machine learning is a subset of AI..."
        quiz = generate_quiz(research_info, "Machine Learning", "medium", 5)
    """

    # GPT-4o 기반 챗 모델 초기화
    model = init_chat_model("openai:gpt-4o")

    # 출력값을 Quiz 형태의 구조화된 JSON으로 제한
    structured_model = model.with_structured_output(Quiz)

    # ---------------------------------------------------------------
    # prompt (한국어 번역)
    # 연구 정보를 기반으로 문제 생성 규칙을 명확하게 전달
    # ---------------------------------------------------------------
    prompt = f"""
    아래 제공한 연구 정보를 기반으로,
    주제: "{topic}"
    난이도: "{difficulty}"
    문제 수: {num_questions}개의 객관식 퀴즈를 생성해주세요.

    <연구정보>
    {research_text}
    </연구정보>

    위 연구정보를 활용해 가장 정확하고 신뢰성 있는 문제를 구성해야 합니다.
    보기 옵션은 반드시 4개(A, B, C, D)로 구성해주세요.
    """

    # 모델 호출 → Quiz 형태의 구조화된 객체 반환
    quiz = structured_model.invoke(prompt)

    return quiz

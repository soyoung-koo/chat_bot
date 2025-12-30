import streamlit as st
import base64
import os
import requests
import re
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ============================================================
# 페이지 및 기본 설정
# ============================================================
st.set_page_config(
    page_title="광고 리스크검증 챗봇",
    page_icon="✅",
    layout="wide",
)

# Document 폴더 자동 생성
if not os.path.exists("Document"):
    os.makedirs("Document")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# ============================================================
# 커스텀 CSS (All-White & Clean Blue 테마)
# ============================================================
st.markdown(
    """
<style>
    :root{
        --bg: #ffffff;
        --panel: #ffffff;
        --border: #edf0f4;

        /* 규제/검수 툴 톤: 딥블루(신뢰) + 상태색(그린/앰버/레드) */
        --primary: #1e3a8a;      /* deep blue */
        --primary-2: #2563eb;    /* hover accent */
        --text: #111827;         /* slate-900 */
        --muted: #6b7280;        /* gray-500 */

        --ok-bg: #e8f5e9;
        --ok-tx: #1b5e20;

        --warn-bg: #fff7ed;      /* amber-ish */
        --warn-tx: #9a3412;

        --danger-bg: #fef2f2;
        --danger-tx: #b91c1c;

        --info-bg: #eff6ff;
        --info-tx: #1d4ed8;
    }

    .stApp { background-color: var(--bg); color: var(--text); }

    [data-testid="stSidebar"] {
        background-color: var(--panel);
        border-right: 1px solid var(--border);
    }

    /* 말풍선: “규제 툴” 느낌으로 더 단정하게 */
    .user-box {
        background-color: var(--primary);
        color: white;
        padding: 14px 16px;
        border-radius: 18px 18px 6px 18px;
        margin: 10px 0 10px 18%;
        box-shadow: 0 6px 14px rgba(17,24,39,0.08);
        font-size: 15px;
        line-height: 1.5;
    }

    .ai-box {
        background-color: #f9fafb;
        color: var(--text);
        padding: 14px 16px;
        border-radius: 18px 18px 18px 6px;
        margin: 10px 18% 10px 0;
        border: 1px solid var(--border);
        box-shadow: 0 4px 10px rgba(17,24,39,0.04);
        font-size: 15px;
        line-height: 1.6;
    }

    /* 버튼: 기본은 “아웃라인”, hover는 primary 채움 */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid rgba(30,58,138,0.35);
        background-color: white;
        color: var(--primary);
        font-weight: 650;
        transition: all 0.2s ease;
        padding: 0.55rem 0.8rem;
        box-shadow: 0 2px 6px rgba(17,24,39,0.03);
    }
    .stButton>button:hover {
        background-color: var(--primary);
        color: white;
        border-color: var(--primary);
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(30,58,138,0.15);
    }

    /* 입력창: 포커스에 primary 적용 */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: rgba(37,99,235,0.55) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
        outline: none !important;
    }

    /* 검색 결과: 좌측 라인 강조는 유지하되 primary 톤으로 정리 */
    .search-result {
        background-color: #f9fafb;
        border-radius: 12px;
        padding: 14px 16px;
        margin: 10px 0;
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
    }
    .source-link {
        color: var(--primary-2);
        font-size: 0.9em;
    }

    /* 모드 배지: 상태색을 확실하게 */
    .mode-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 10px;
        border: 1px solid rgba(17,24,39,0.06);
    }
    .mode-rag { background-color: var(--info-bg); color: var(--primary); }
    .mode-web { background-color: var(--info-bg); color: var(--info-tx); }
    .mode-llm { background-color: var(--warn-bg); color: var(--warn-tx); }

    /* (선택) 위험도 배지 추가하려면 나중에 이 클래스만 써도 됨 */
    .risk-low  { background-color: var(--ok-bg); color: var(--ok-tx); }
    .risk-mid  { background-color: var(--warn-bg); color: var(--warn-tx); }
    .risk-high { background-color: var(--danger-bg); color: var(--danger-tx); }

</style>
""",
    unsafe_allow_html=True,
)



def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None


# ============================================================
# RAG: 인덱싱 함수
# ============================================================
def perform_indexing():
    with st.spinner("Document 폴더 내 문서를 인덱싱 중입니다..."):
        try:
            loader = PyPDFDirectoryLoader("Document/")
            documents = loader.load()
            if not documents:
                st.warning("Document 폴더에 PDF 파일이 없습니다.")
                return
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            st.session_state.vector_store = vectorstore
            st.success(f"인덱싱 완료! 총 {len(splits)}개의 지식 조각을 생성했습니다.")
        except Exception as e:
            st.error(f"인덱싱 중 오류 발생: {e}")


# ============================================================
# 웹 검색 함수
# ============================================================
def search_naver_blog(query: str, num_results: int = 10) -> list:
    """네이버 블로그 검색 API"""
    url = "https://openapi.naver.com/v1/search/blog.json"
    headers = {
        "X-Naver-Client-Id": st.secrets["NAVER_CLIENT_ID"],
        "X-Naver-Client-Secret": st.secrets["NAVER_CLIENT_SECRET"],
    }
    params = {
        "query": query,
        "display": num_results,
        "sort": "sim",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()

        search_results = []
        for item in results.get("items", []):
            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
            description = re.sub(r"<[^>]+>", "", item.get("description", ""))
            search_results.append(
                {
                    "title": title,
                    "link": item.get("link", ""),
                    "snippet": description,
                    "source": "네이버 블로그",
                    "date": item.get("postdate", ""),
                }
            )
        return search_results
    except Exception as e:
        return []


def search_naver_cafe(query: str, num_results: int = 10) -> list:
    """네이버 카페 검색 API"""
    url = "https://openapi.naver.com/v1/search/cafearticle.json"
    headers = {
            "X-Naver-Client-Id": st.secrets["NAVER_CLIENT_ID"],
            "X-Naver-Client-Secret": st.secrets["NAVER_CLIENT_SECRET"],
    }
    params = {"query": query, "display": num_results, "sort": "sim"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()

        search_results = []
        for item in results.get("items", []):
            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
            description = re.sub(r"<[^>]+>", "", item.get("description", ""))
            search_results.append(
                {
                    "title": title,
                    "link": item.get("link", ""),
                    "snippet": description,
                    "source": "네이버 카페",
                    "cafe_name": item.get("cafename", ""),
                }
            )
        return search_results
    except Exception as e:
        return []


def search_web(query: str, sources: list, num_results: int = 5) -> list:
    """네이버 블로그 + 카페 통합 검색"""
    all_results = []
    if "네이버 블로그" in sources:
        all_results.extend(search_naver_blog(query, num_results))
    if "네이버 카페" in sources:
        all_results.extend(search_naver_cafe(query, num_results))
    return all_results


# ============================================================
# 질문 분류 함수
# ============================================================
def classify_query(query: str, has_vector_store: bool) -> str:
    """
    질문을 분류하여 RAG / AUTO로 분기
    - 규제 기준·법·가이드·사례 설명 → RAG
    - 광고 문구 분석/수정, 이미지 프롬프트, 광고 세팅 등 실행 요청 → AUTO
    """

    # 규제/법/가이드 설명용 키워드 (RAG 사용)
    rag_keywords = [
        # 기관 / 법
        "식약처", "식품의약품안전처",
        "표시광고법", "식품표시광고법", "건강기능식품법",
        "법", "법률", "조항",

        # 규제 / 가이드
        "표시·광고", "표시광고",
        "규제", "기준", "가이드", "가이드라인",

        # 심의 / 제재
        "광고심의", "사전심의", "심의",
        "허위과대광고", "과대광고", "부당광고",
        "위반사례", "행정처분", "적발"
    ]

    query_lower = query.lower()

    # RAG 키워드 체크
    for keyword in rag_keywords:
        if keyword in query_lower:
            return "RAG"

    # 그 외 질문은 AUTO
    # (광고카피 분석, 대체 문구, 이미지 프롬프트, 광고 세팅 등)
    return "AUTO"


def determine_search_need(query: str, api_key: str) -> dict:
    """
    LLM을 사용하여 질문이 웹 검색이 필요한지 판단
    Returns: {"need_search": bool, "reason": str, "search_query": str}
    """
    llm = ChatOpenAI(
        model="gpt-5-mini",
        api_key=api_key,
        temperature=1,
    )
    
    classification_prompt = f"""
당신은 '웹 검색 필요 여부'를 판단하는 분류기입니다.
반드시 JSON 형식으로만 응답하세요. (다른 텍스트 절대 금지)

[웹 검색이 필요한 경우]
- 최근(최신) 규제 동향 확인이 필요한 질문
  예) "최근 허위·과대광고 적발 사례", "행정처분/보도자료", "가이드라인 개정 여부", "협회 공지/FAQ 변경"
- 특정 기관/페이지의 최신 문서나 공지 링크가 필요한 질문
  예) "식약처 공지 링크", "건강기능식품협회 심의 절차 최신 안내"
- 특정 브랜드/제품 관련 최근 이슈/기사 확인이 필요한 질문

[웹 검색이 필요 없는 경우]
- 일반적인 규제 원칙/기준 설명 (질병 치료 표현 금지, 의약품 오인 금지, 과장·단정 표현 등)
- 사용자가 제공한 광고카피를 바탕으로 위험요소 분석 및 대체 문구 제안
- AI 이미지 프롬프트 작성, 광고 세팅 추천 등 실행 가이드
- 코딩/문서 인덱싱/RAG 관련 도움

질문: "{query}"

아래 JSON 중 하나로만 응답:
{{"need_search": true, "reason": "이유(한 문장)", "search_query": "검색어(짧게)"}}
또는
{{"need_search": false, "reason": "이유(한 문장)", "search_query": ""}}
"""
    
    try:
        response = llm.invoke([HumanMessage(content=classification_prompt)])
        result_text = response.content.strip()
        
        # ```json 등의 마크다운 제거
        if "```" in result_text:
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*', '', result_text)
            result_text = result_text.strip()
        
        # JSON 파싱 시도
        result = json.loads(result_text)
        
        # 필수 키 검증
        if "need_search" not in result:
            result["need_search"] = False
        if "reason" not in result:
            result["reason"] = "자동 판단"
        if "search_query" not in result:
            result["search_query"] = ""
            
        return result
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 텍스트에서 판단 시도
        result_lower = response.content.lower() if response else ""
        if "true" in result_lower or "필요" in result_lower:
            return {"need_search": True, "reason": "웹 검색 필요로 판단", "search_query": query}
        return {"need_search": False, "reason": "AI 직접 답변 가능", "search_query": ""}
    except Exception as e:
        # 기타 오류 시 기본값 반환
        return {"need_search": False, "reason": f"판단 중 오류: {str(e)}", "search_query": ""}


# ============================================================
# 대표 질문용 미리 정의된 답변
# ============================================================
PREDEFINED_ANSWERS = {

    "✍️ 이 문구를 광고 심의 통과 가능하게 수정해줘": """
네, 가능합니다. 😊  
위반 가능성이 있는 표현을 **식약처 가이드라인 기준에 맞는 안전한 문구**로 수정해 드립니다.

수정 시 적용 원칙은 다음과 같습니다.
* ❌ 질병의 예방·치료·개선 표현 제거 (예: 치료, 완치, 예방, 회복)
* ❌ 의약품으로 오인될 수 있는 표현 제거 (예: 처방, 전문의/약사 추천)
* ❌ 과장·단정적 표현 완화 (예: 100%, 즉각, 무조건, 확실)
* ✅ **‘도움이 될 수 있음’**, **‘건강 유지에 도움’** 등 허용 표현으로 전환
* ✅ 필요 시 개인차/생활습관 병행 문구 보강

📌 예시:
- 변경 전: “장염 개선에 효과”
- 변경 후: “장 건강 유지에 도움을 줄 수 있음(개인차가 있을 수 있음)”

👉 수정이 필요한 **광고 문구를 그대로 붙여주세요.**
    """,

    "🎨 해당 카피에 어울리는 AI 이미지 제작 프롬프트 작성해줘": """
가능합니다. 아래 정보를 주시면 **광고용 이미지 생성 프롬프트를 디테일하게** 만들어 드립니다.

필수로 확인하는 요소:
1) **제품/카테고리**: 예) 장건강, 이너뷰티, 피로·에너지 등  
2) **타깃**: 성별/연령/상황(직장인, 육아맘 등)  
3) **톤앤무드**: 클린/프리미엄/내추럴/활력 등  
4) **금지 요소(규제 리스크 회피)**: 병원, 의사, 약, 전후(Before/After), 과도한 신체 변화 연출 등

📌 출력은 보통 이렇게 드려요:
- 추천 스타일/무드
- 구성(배경/소품/조명/구도)
- 네거티브 프롬프트(피해야 할 요소)
- 최종 프롬프트(한글/영문)

👉 광고 카피(문구)를 붙여주시면, 그 톤에 맞춰 바로 생성해 드릴게요.
    """,

    "📊 이 광고 카피에 어울리는 광고 세팅을 추천해줘": """
좋아요. 광고 카피의 **목표(인지/트래픽/전환)**와 **제품 카테고리**에 맞춰
매체별로 현실적인 세팅을 추천해 드립니다.

기본으로 제안하는 항목:
1) **추천 매체**: Meta / 네이버 검색 / GFA / 유튜브 등  
2) **캠페인 목표**: 전환(구매/장바구니) vs 트래픽 vs 도달  
3) **타겟팅**: 연령·성별·관심사(카테고리 기반) / 리타겟팅 구조  
4) **크리에이티브 타입**: 단일 이미지/카루셀/숏폼/UGC 톤  
5) **랜딩/심의 주의 포인트**: 카피·상세페이지에서 컷 될 요소 체크

👉 아래 중 아는 것만 알려주면 더 정확해요:
- 판매채널(자사몰/스마트스토어/쿠팡)
- 광고 목적(매출 vs 유입 vs 인지도)
- 타겟(성별/연령)
- 예산(일 예산 대략)

우선은 광고 카피를 붙여주시면, 카피 톤에 맞춰 **바로 추천 세팅**을 드릴게요.
    """,
}



# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    logo_b64 = get_base64_image("SeSAC_logo.png")
    if logo_b64:
        st.markdown(
            f'<img src="data:image/png;base64,{logo_b64}" width="100%">',
            unsafe_allow_html=True,
        )
    else:
        st.title("🏛️ SeSAC AI")

    st.divider()
    
    # 지식 데이터베이스 섹션
    st.subheader("📚 지식 데이터베이스")
    if st.button("규제문서 인덱싱 시작"):
        perform_indexing()
    if st.session_state.vector_store:
        st.caption("✅ 규제문서 학습 완료 (광고 심의기준 적용 중)")

    st.divider()
    
    # 웹 검색 설정 섹션
    st.subheader("🔍 웹 검색 설정")
    search_sources = st.multiselect(
        "검색 소스",
        ["네이버 블로그", "네이버 카페"],
        default=["네이버 블로그", "네이버 카페"],
    )
    num_results = st.slider("소스별 검색 결과 수", 3, 15, 5)
    
    st.divider()
    
    # AI 페르소나 설정
    st.subheader("AI 페르소나 설정")
    system_instruction = st.text_area(
        "AI 역할 정의:",
        value="""
너는 건강기능식품 광고 카피 규제 전문 AI다.
식약처 가이드라인, 건강기능식품법, 표시·광고법을 기반으로 광고 문구의 위반 가능성, 위험 유형, 심의 리스크를 분석한다.

제공된 [Context]와 내부 규제 문서를 우선적으로 참고하여 답변하며, 추측이나 과도한 해석은 피하고 보수적인 규제 기준으로 판단한다.

답변 시 다음 원칙을 따른다.
1. 위반 가능 표현을 명확히 지적한다.
2. 왜 문제가 되는지 규제 관점에서 설명한다.
3. 광고 심의 통과 가능성이 높은 대체 문구를 제안한다.
4. 법률 자문이 아닌 가이드 목적의 분석임을 명확히 한다.
""",
        height=150,
    )
    
    st.divider()
    
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.search_history = []
        st.rerun()
    
    # 통계 표시
    st.divider()
    st.subheader("📊 사용 통계")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("대화 수", len(st.session_state.messages) // 2)
    with col2:
        st.metric("웹 검색", len(st.session_state.search_history))

# ============================================================
# 메인 화면
# ============================================================
st.markdown(
    "<h2 style='color: #0066cc;'>건기식 광고 리스크검증 챗봇</h2>", unsafe_allow_html=True
)
st.caption("🚀 RAG + AI 규제 판단 | 광고 카피 검수부터 크리에이티브·세팅 추천까지 한 번에")

st.markdown("### 자주 묻는 질문")
col1, col2, col3 = st.columns(3)
q1 = "✍️ 이 문구를 광고 심의 통과 가능하게 수정해줘"
q2 = "🎨 해당 카피에 어울리는 AI 이미지 제작 프롬프트 작성해줘"
q3 = "📊 이 광고 카피에 어울리는 광고 세팅을 추천해줘"

clicked_q = None
if col1.button("✍️ 문구 수정"):
    clicked_q = q1
if col2.button("🎨 소재 프롬프트"):
    clicked_q = q2
if col3.button("📊 세팅 추천"):
    clicked_q = q3

st.divider()

# 대화 기록 표시
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(
            f'<div class="user-box">{msg.content}</div>',
            unsafe_allow_html=True
        )
    elif isinstance(msg, AIMessage):
        st.markdown(
            f'<div class="ai-box">{msg.content}</div>',
            unsafe_allow_html=True
        )
# 사용자 입력
user_input = st.chat_input(
    "광고 카피 또는 요청을 입력해주세요. (예: 이 문구 심의 통과 가능하게 바꿔줘 / 이미지 프롬프트 만들어줘)"
)
final_query = clicked_q if clicked_q else user_input

if final_query:
    st.markdown(f'<div class="user-box">{final_query}</div>', unsafe_allow_html=True)
    st.session_state.messages.append(HumanMessage(content=final_query))

    # 답변 생성 로직
    if final_query in PREDEFINED_ANSWERS:
        # 미리 정의된 답변
        ai_content = PREDEFINED_ANSWERS[final_query]
        mode_badge = '<span class="mode-badge mode-rag">🧩 대표 질문</span>'
    else:
        # 질문 분류
        query_type = classify_query(final_query, st.session_state.vector_store is not None)
        
        try:
            if query_type == "RAG":
                # RAG 모드 (규제/가이드 문서 기반)
                mode_badge = '<span class="mode-badge mode-rag">📚 규제 RAG 모드</span>'
                
                context = ""
                if st.session_state.vector_store:
                    docs = st.session_state.vector_store.similarity_search(final_query, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])

                llm = ChatOpenAI(
                    model="gpt-5-mini",
                    api_key=st.secrets["OPENAI_API_KEY"],
                    streaming=True,
                    temperature=1,
                )

                full_system_prompt = f"{system_instruction}\n\n[Context]\n{context if context else '관련 문서 없음'}"
                prompt = [
                    SystemMessage(content=full_system_prompt)
                ] + st.session_state.messages

                with st.spinner("답변 생성 중..."):
                    response = llm.invoke(prompt)
                    ai_content = response.content
                    
            else:
                # AUTO 모드: 질문 유형에 따라 필요 시 웹 검색 + 종합 답변
                with st.spinner("질문 분석 중..."):
                    search_decision = determine_search_need(final_query, st.secrets["OPENAI_API_KEY"])
                
                if search_decision["need_search"]:
                    # 웹 검색 모드
                    mode_badge = '<span class="mode-badge mode-web">🔍 웹 검색 모드</span>'
                    
                    search_query = search_decision["search_query"] if search_decision["search_query"] else final_query
                    
                    with st.status(f"🔍 웹에서 '{search_query}' 검색 중...", expanded=True) as status:
                        all_results = []
                        seen_links = set()
                        
                        # 검색 실행
                        results = search_web(search_query, search_sources, num_results)
                        
                        for result in results:
                            if result["link"] not in seen_links:
                                seen_links.add(result["link"])
                                all_results.append(result)
                        
                        st.write(f"✅ {len(all_results)}개의 결과를 찾았습니다.")
                        st.caption(f"💡 판단 이유: {search_decision['reason']}")
                        status.update(label="검색 완료!", state="complete")
                    
                    # 검색 결과 표시
                    if all_results:
                        with st.expander("📑 검색된 원본 자료 보기", expanded=False):
                            for i, result in enumerate(all_results[:10], 1):
                                st.markdown(
                                    f"""
                                <div class="search-result">
                                    <strong>{i}. {result['title']}</strong><br>
                                    <span class="source-link">🔗 <a href="{result['link']}" target="_blank">{result['source']}</a></span><br>
                                    <small>{result['snippet'][:200]}...</small>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )
                        
                        # 검색 기록 저장
                        st.session_state.search_history.append({
                            "query": search_query,
                            "results_count": len(all_results),
                        })
                    
                    # 웹 검색 결과를 컨텍스트로 구성
                    web_context = ""
                    for i, result in enumerate(all_results, 1):
                        web_context += f"\n[결과 {i}]\n"
                        web_context += f"제목: {result['title']}\n"
                        web_context += f"출처: {result['source']}\n"
                        web_context += f"링크: {result['link']}\n"
                        web_context += f"내용: {result['snippet']}\n"
                    
                    # LLM으로 웹 검색 결과 분석
                    llm = ChatOpenAI(
                        model="gpt-5-mini",
                        api_key=st.secrets["OPENAI_API_KEY"],
                        streaming=True,
                        temperature=1,
                    )
                    
                    web_system_prompt = f"""{system_instruction}

아래는 사용자 질문과 관련된 웹 검색 결과입니다. 이 정보를 바탕으로 **광고/규제 관점에서** 종합적으로 분석하여 답변해주세요.

- 웹 검색 결과는 참고 자료이며, 확실하지 않은 내용은 단정하지 말고 '추정' 또는 '추가 확인 필요'라고 표시하세요.
- 건강기능식품 광고는 의약품 오인/질병 치료 암시 표현이 민감하므로, 관련 표현은 보수적으로 안내하세요.
- 답변 말미에 참고한 출처를 **제목 + 링크** 형태로 3~5개 표시하세요.


[웹 검색 결과]
{web_context if web_context else '검색 결과 없음'}"""

                    prompt = [
                        SystemMessage(content=web_system_prompt)
                    ] + st.session_state.messages
                    
                    with st.spinner("답변 생성 중..."):
                        response = llm.invoke(prompt)
                        ai_content = response.content
                else:
                    # 일반 LLM 모드 (웹 검색 불필요)
                    mode_badge = '<span class="mode-badge" style="background-color:#fff3e0;color:#e65100;">🧠 AI 직접 답변</span>'
                    
                    llm = ChatOpenAI(
                        model="gpt-5-mini",
                        api_key=st.secrets["OPENAI_API_KEY"],
                        streaming=True,
                        temperature=1,
                    )
                    
                    # 일반 답변용 시스템 프롬프트 (웹 검색 언급 제거)
                    general_system_prompt = "너는 친절하고 유능한 AI 어시스턴트야. 사용자의 질문에 정확하고 도움이 되는 답변을 제공해줘."

                    prompt = [
                        SystemMessage(content=general_system_prompt)
                    ] + st.session_state.messages

                    with st.spinner("답변 생성 중..."):
                        response = llm.invoke(prompt)
                        ai_content = response.content
                    
        except Exception as e:
            ai_content = f"오류가 발생했습니다: {e}"
            mode_badge = '<span class="mode-badge" style="background-color:#ffebee;color:#c62828;">⚠️ 오류</span>'

    # 답변 표시
    st.markdown(mode_badge, unsafe_allow_html=True)
    st.markdown(f'<div class="ai-box">{ai_content}</div>', unsafe_allow_html=True)
    st.session_state.messages.append(AIMessage(content=ai_content))

# 하단 안내
st.divider()
st.caption(
    """
💡 **사용 안내**  
- **광고카피 분석**: 입력한 광고 문구를 기준으로 위반 가능성·위험 유형을 분석합니다.  
- **규제 근거 판단**: 건강기능식품 표시·광고 관련 법령 및 가이드라인 문서를 기반으로 검토합니다.  
- **대체 문구 제안**: 심의 리스크를 낮춘 안전한 광고 문구를 자동으로 추천합니다.  
- **실행 가이드 제공**: 광고에 활용 가능한 AI 이미지 프롬프트 및 매체별 광고 세팅을 함께 제안합니다.  
- **최신 이슈 확인**: 필요 시 웹 검색을 통해 최근 규제 사례 및 참고 정보를 반영합니다.
"""
)
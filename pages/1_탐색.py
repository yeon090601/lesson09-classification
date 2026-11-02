
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------
# 기본 설정
# ---------------------------------

APP_TITLE = "뇌졸중 예측 실습실"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "뇌졸중_예측_실습실.json"

COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]


# ---------------------------------
# 페이지 설정
# ---------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 뇌졸중 예측 실습실")

st.markdown("""
이 실습실에서는 뇌졸중 관련 데이터를 살펴보고
데이터의 구조와 특징을 알아봅니다.

각 행은 한 사람의 정보를 나타내며,
`stroke` 열은 뇌졸중 발생 여부를 나타냅니다.

- `stroke = 1`: 뇌졸중 발생
- `stroke = 0`: 뇌졸중 미발생
- `bmi`: 일부 값이 비어 있을 수 있습니다.
""")


# ---------------------------------
# 데이터 불러오기
# ---------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    missing_columns = [
        col for col in COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"데이터에 필요한 열이 없습니다: {missing_columns}"
        )

    return df[COLUMNS].copy()


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# ---------------------------------
# 기존 기록 불러오기
# ---------------------------------

def load_notes():

    if not RECORD_FILE.exists():
        return {}, ""

    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as file:
            saved = json.load(file)

        meanings = saved.get("meanings", {})
        source = saved.get("source", "")

        if not isinstance(meanings, dict):
            meanings = {}

        if not isinstance(source, str):
            source = ""

        return meanings, source

    except (OSError, json.JSONDecodeError):
        st.warning("기존 기록을 읽지 못했습니다.")
        return {}, ""


# ---------------------------------
# 기록 저장하기
# ---------------------------------

def save_notes(meanings, source):

    RECORD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    saved = {
        "meanings": meanings,
        "source": source
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as file:
        json.dump(
            saved,
            file,
            ensure_ascii=False,
            indent=4
        )


saved_meanings, saved_source = load_notes()


# ---------------------------------
# 1. 데이터 통계
# ---------------------------------

st.subheader("📊 데이터 한눈에 보기")

total_people = len(df)
total_columns = len(df.columns)

stroke_values = pd.to_numeric(
    df["stroke"],
    errors="coerce"
)

stroke_people = int(
    (stroke_values == 1).sum()
)

stroke_percent = (
    stroke_people / total_people * 100
    if total_people > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "전체 사람 수",
    f"{total_people:,}명"
)

col2.metric(
    "열 개수",
    f"{total_columns}개"
)

col3.metric(
    "뇌졸중 발생 인원",
    f"{stroke_people:,}명"
)

col4.metric(
    "뇌졸중 발생 비율",
    f"{stroke_percent:.2f}%"
)


st.divider()


# ---------------------------------
# 2. 열 이름과 데이터 구조
# ---------------------------------

st.subheader("📋 데이터 열 살펴보기")

st.write("""
각 열이 어떤 정보를 의미하는지 교재를 참고하여
'우리말 뜻' 칸에 직접 입력해 보세요.
""")

table_data = []

for col in df.columns:

    unique_count = df[col].nunique(dropna=True)
    missing_count = int(df[col].isna().sum())

    if pd.api.types.is_numeric_dtype(df[col]):
        value_type = f"수치형 ({unique_count}종류)"
    else:
        value_type = f"범주형 ({unique_count}종류)"

    table_data.append({
        "열 이름": col,
        "우리말 뜻": saved_meanings.get(col, ""),
        "값의 종류": value_type,
        "빈 값 개수": missing_count
    })


meaning_df = pd.DataFrame(table_data)

edited_df = st.data_editor(
    meaning_df,
    use_container_width=True,
    hide_index=True,
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름"
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고하여 직접 입력하세요."
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d"
        )
    },
    key="meaning_editor"
)


st.divider()


# ---------------------------------
# 3. 데이터 처음 다섯 줄
# ---------------------------------

st.subheader("🔎 데이터 처음 다섯 줄")

st.write("""
실제 데이터의 처음 다섯 행입니다.
원본 데이터를 그대로 보여 줍니다.
""")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


st.divider()


# ---------------------------------
# 4. 데이터 출처 입력
# ---------------------------------

st.subheader("📚 데이터 출처")

source_text = st.text_area(
    "교재에 있는 데이터 출처를 입력하세요.",
    value=saved_source,
    placeholder="데이터 출처를 여기에 입력하세요.",
    height=120
)


# ---------------------------------
# 5. 저장 버튼
# ---------------------------------

if st.button(
    "💾 우리말 뜻과 데이터 출처 저장",
    type="primary"
):

    meanings = {}

    for _, row in edited_df.iterrows():

        value = row["우리말 뜻"]

        if pd.isna(value):
            value = ""

        meanings[str(row["열 이름"])] = str(value)

    try:
        save_notes(meanings, source_text)

        st.success(
            "저장되었습니다! "
            "새로고침해도 입력한 내용을 다시 불러옵니다."
        )

    except OSError as e:
        st.error("파일 저장에 실패했습니다.")
        st.exception(e)


st.caption(
    f"저장 파일 위치: {RECORD_FILE.as_posix()}"
)

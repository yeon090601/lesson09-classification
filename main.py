
import streamlit as st
import pandas as pd
import json
from pathlib import Path

# =====================================
# 기본 설정
# =====================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고 뇌졸중과 관련된 특성을 탐구하는 실습 공간입니다.")

# =====================================
# 데이터 및 기록 파일 경로
# =====================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "yeon090601/data/refs/heads/main/stroke.csv"
)

RECORD_DIR = Path(__file__).parent / "내_기록"
RECORD_FILE = RECORD_DIR / "stroke_records.json"

RECORD_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# JSON 기록 불러오기 및 저장
# =====================================

def load_records():
    if RECORD_FILE.exists():
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    return {}


def save_records(meaning_dict, source_text):
    records = {
        "meanings": meaning_dict,
        "data_source": source_text
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=4
        )


# =====================================
# 데이터 불러오기
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.write(e)
    st.stop()


# =====================================
# 기존 기록 불러오기
# =====================================

if "records_loaded" not in st.session_state:
    saved_records = load_records()

    st.session_state.meanings = saved_records.get(
        "meanings",
        {}
    )

    st.session_state.data_source = saved_records.get(
        "data_source",
        ""
    )

    st.session_state.records_loaded = True


# =====================================
# 첫 화면 소개
# =====================================

st.header("📊 데이터 소개")

st.write(
    """
    이 데이터는 사람들의 건강 상태와 생활 습관에 관한 정보를
    담고 있으며, 뇌졸중 발생 여부를 나타내는 `stroke` 열을 포함합니다.

    각 열의 의미와 값의 종류를 직접 확인하면서
    뇌졸중과 관련된 데이터를 이해해 봅시다.
    """
)

st.divider()


# =====================================
# 주요 데이터 통계
# =====================================

total_people = len(df)
total_columns = len(df.columns)
stroke_people = int((df["stroke"] == 1).sum())

stroke_ratio = (
    stroke_people / total_people * 100
    if total_people > 0 else 0
)

st.subheader("📌 데이터 주요 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="전체 사람 수",
        value=f"{total_people:,}명"
    )

with col2:
    st.metric(
        label="전체 열 개수",
        value=f"{total_columns}개"
    )

with col3:
    st.metric(
        label="뇌졸중 발생 인원",
        value=f"{stroke_people:,}명"
    )

with col4:
    st.metric(
        label="뇌졸중 발생 비율",
        value=f"{stroke_ratio:.2f}%"
    )

st.divider()


# =====================================
# 열별 데이터 정보 표
# =====================================

st.subheader("📝 열 이름과 데이터 정보")

st.write(
    "각 열의 우리말 뜻을 교재를 참고하여 직접 작성해 보세요. "
    "입력한 내용은 자동으로 저장됩니다."
)

saved_meanings = st.session_state.meanings

table_data = []

for column in df.columns:

    unique_count = df[column].nunique(dropna=True)

    if pd.api.types.is_numeric_dtype(df[column]):
        value_type = f"숫자형 (고유값 {unique_count}종)"
    else:
        values = df[column].dropna().unique().tolist()

        if len(values) <= 10:
            value_type = ", ".join(str(v) for v in values)
        else:
            value_type = f"문자형 (고유값 {unique_count}종)"

    table_data.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": value_type,
        "빈 값 개수": int(df[column].isna().sum())
    })

info_df = pd.DataFrame(table_data)

edited_df = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
            width="medium"
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            width="large",
            help="교재를 참고하여 우리말 뜻을 입력하세요."
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류",
            width="large"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d"
        )
    },
    key="meaning_editor"
)

# 입력한 우리말 뜻 저장
new_meanings = {}

for _, row in edited_df.iterrows():
    new_meanings[row["열 이름"]] = row["우리말 뜻"]

st.session_state.meanings = new_meanings

save_records(
    st.session_state.meanings,
    st.session_state.data_source
)

st.success("우리말 뜻이 자동 저장되었습니다.")

st.divider()


# =====================================
# 데이터 첫 5행 표시
# =====================================

st.subheader("🔎 데이터 미리 보기")

st.write("실제 CSV 데이터의 처음 다섯 행입니다.")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)

st.divider()


# =====================================
# 데이터 출처 기록
# =====================================

st.subheader("📚 데이터 출처")

st.write(
    "교재에 제시된 데이터 출처를 아래에 직접 작성하세요. "
    "입력한 내용은 자동으로 저장됩니다."
)

source_text = st.text_area(
    "데이터 출처 입력",
    value=st.session_state.data_source,
    placeholder="교재에 적힌 데이터 출처를 입력하세요.",
    height=120,
    key="source_editor"
)

st.session_state.data_source = source_text

save_records(
    st.session_state.meanings,
    st.session_state.data_source
)

st.success("데이터 출처가 자동 저장되었습니다.")

st.caption(
    "저장 위치: 내_기록/stroke_records.json"
)

st.divider()

st.caption(
    "※ 본 앱은 데이터 분석 실습용이며, 실제 뇌졸중 진단이나 "
    "개인의 의료적 위험도 판단에 사용할 수 없습니다."
)

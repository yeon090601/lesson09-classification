
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------
# 기본 설정
# ---------------------------------

APP_TITLE = "뇌졸중 예측 실습실"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

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

st.set_page_config(
    page_title=f"탐색 | {APP_TITLE}",
    page_icon="🔎",
    layout="wide",
)


# ---------------------------------
# 데이터 불러오기
# ---------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    missing_columns = [
        col for col in COLUMNS
        if col not in df.columns
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


# 분석에 사용할 숫자형 데이터 변환
df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce"
)

df["avg_glucose_level"] = pd.to_numeric(
    df["avg_glucose_level"],
    errors="coerce"
)

df["bmi"] = pd.to_numeric(
    df["bmi"],
    errors="coerce"
)

df["stroke"] = pd.to_numeric(
    df["stroke"],
    errors="coerce"
)

df["hypertension"] = pd.to_numeric(
    df["hypertension"],
    errors="coerce"
)

df["heart_disease"] = pd.to_numeric(
    df["heart_disease"],
    errors="coerce"
)


# ---------------------------------
# 페이지 제목
# ---------------------------------

st.title("🔎 데이터 탐색")

st.markdown("""
뇌졸중 데이터의 여러 특성을 그래프와 표로 살펴봅니다.

나이, 평균 혈당, 고혈압, 심장병, 체질량지수,
흡연 상태와 뇌졸중 발생 사이의 관계를 탐색해 보세요.
""")

st.divider()


# ---------------------------------
# 1. 나이와 평균 혈당의 분포
# ---------------------------------

st.header("1. 나이와 평균 혈당의 분포")

st.write("""
히스토그램을 이용해 사람들의 나이와 평균 혈당이
어떤 범위에 많이 분포하는지 확인합니다.
""")

col1, col2 = st.columns(2)

with col1:

    age_data = df.dropna(subset=["age"])

    fig_age = px.histogram(
        age_data,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        },
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )


with col2:

    glucose_data = df.dropna(
        subset=["avg_glucose_level"]
    )

    fig_glucose = px.histogram(
        glucose_data,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        },
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )


st.divider()


# ---------------------------------
# 2. 뇌졸중 여부에 따른 나이와 혈당 비교
# ---------------------------------

st.header("2. 뇌졸중 여부에 따른 비교")

st.write("""
뇌졸중을 겪은 사람과 겪지 않은 사람을 나누어
나이와 평균 혈당의 분포를 상자그림으로 비교합니다.
""")

comparison_df = df[
    df["stroke"].isin([0, 1])
].copy()

comparison_df["뇌졸중 여부"] = comparison_df[
    "stroke"
].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

col1, col2 = st.columns(2)

with col1:

    age_box_data = comparison_df.dropna(
        subset=["age"]
    )

    fig_age_box = px.box(
        age_box_data,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={
            "age": "나이",
            "뇌졸중 여부": "그룹"
        },
    )

    fig_age_box.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True
    )


with col2:

    glucose_box_data = comparison_df.dropna(
        subset=["avg_glucose_level"]
    )

    fig_glucose_box = px.box(
        glucose_box_data,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={
            "avg_glucose_level": "평균 혈당",
            "뇌졸중 여부": "그룹"
        },
    )

    fig_glucose_box.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True
    )


# 두 그룹의 평균값 표
st.subheader("두 그룹의 평균값 비교")

mean_table = (
    comparison_df
    .groupby("뇌졸중 여부")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reindex([
        "뇌졸중 없음",
        "뇌졸중 있음"
    ])
    .reset_index()
)

mean_table = mean_table.rename(
    columns={
        "뇌졸중 여부": "그룹",
        "age": "평균 나이",
        "avg_glucose_level": "평균 혈당"
    }
)

mean_table["평균 나이"] = mean_table[
    "평균 나이"
].round(2)

mean_table["평균 혈당"] = mean_table[
    "평균 혈당"
].round(2)

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ---------------------------------
# 3. 고혈압 여부에 따른 뇌졸중 비율
# ---------------------------------

st.header("3. 고혈압과 심장병에 따른 뇌졸중 비율")

st.write("""
고혈압이 있는 사람과 없는 사람,
심장병이 있는 사람과 없는 사람의
뇌졸중 발생 비율을 비교합니다.
""")


def make_stroke_rate_table(data, column, labels):

    temp = data[
        data[column].isin([0, 1])
        & data["stroke"].isin([0, 1])
    ].copy()

    result = (
        temp.groupby(column)["stroke"]
        .agg(["count", "mean"])
        .reset_index()
    )

    result["구분"] = result[column].map(labels)
    result["뇌졸중 비율"] = result["mean"] * 100

    result = result.rename(
        columns={
            "count": "전체 사람 수"
        }
    )

    return result[
        ["구분", "전체 사람 수", "뇌졸중 비율"]
    ]


binary_labels = {
    0: "없음",
    1: "있음"
}

col1, col2 = st.columns(2)

with col1:

    hypertension_table = make_stroke_rate_table(
        df,
        "hypertension",
        binary_labels
    )

    fig_hypertension = px.bar(
        hypertension_table,
        x="구분",
        y="뇌졸중 비율",
        text="뇌졸중 비율",
        title="고혈압 여부에 따른 뇌졸중 비율",
        labels={
            "구분": "고혈압",
            "뇌졸중 비율": "뇌졸중 비율 (%)"
        },
    )

    fig_hypertension.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_hypertension.update_layout(
        yaxis_title="뇌졸중 비율 (%)",
        xaxis_title="고혈압 여부",
        yaxis_range=[
            0,
            max(
                5,
                hypertension_table["뇌졸중 비율"].max() * 1.2
            ) if not hypertension_table.empty else 5
        ]
    )

    st.plotly_chart(
        fig_hypertension,
        use_container_width=True
    )

    st.dataframe(
        hypertension_table.style.format({
            "뇌졸중 비율": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


with col2:

    heart_table = make_stroke_rate_table(
        df,
        "heart_disease",
        binary_labels
    )

    fig_heart = px.bar(
        heart_table,
        x="구분",
        y="뇌졸중 비율",
        text="뇌졸중 비율",
        title="심장병 여부에 따른 뇌졸중 비율",
        labels={
            "구분": "심장병",
            "뇌졸중 비율": "뇌졸중 비율 (%)"
        },
    )

    fig_heart.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_heart.update_layout(
        yaxis_title="뇌졸중 비율 (%)",
        xaxis_title="심장병 여부",
        yaxis_range=[
            0,
            max(
                5,
                heart_table["뇌졸중 비율"].max() * 1.2
            ) if not heart_table.empty else 5
        ]
    )

    st.plotly_chart(
        fig_heart,
        use_container_width=True
    )

    st.dataframe(
        heart_table.style.format({
            "뇌졸중 비율": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ---------------------------------
# 4. BMI 결측값과 뇌졸중 비율
# ---------------------------------

st.header("4. BMI가 비어 있는 사람들의 뇌졸중 비율")

st.write("""
체질량지수(bmi)가 비어 있는 사람들을 따로 구분합니다.
이 사람들의 뇌졸중 비율과 전체 데이터의 뇌졸중 비율을 비교합니다.
""")

missing_bmi_df = df[
    df["bmi"].isna()
].copy()

valid_stroke_df = df[
    df["stroke"].isin([0, 1])
].copy()

missing_bmi_stroke_df = missing_bmi_df[
    missing_bmi_df["stroke"].isin([0, 1])
].copy()

missing_bmi_people = len(missing_bmi_df)

missing_bmi_stroke_count = int(
    (missing_bmi_stroke_df["stroke"] == 1).sum()
)

missing_bmi_stroke_rate = (
    missing_bmi_stroke_df["stroke"].mean() * 100
    if len(missing_bmi_stroke_df) > 0
    else 0
)

overall_stroke_count = int(
    (valid_stroke_df["stroke"] == 1).sum()
)

overall_stroke_rate = (
    valid_stroke_df["stroke"].mean() * 100
    if len(valid_stroke_df) > 0
    else 0
)

bmi_table = pd.DataFrame({
    "구분": [
        "BMI가 비어 있는 사람",
        "전체 사람"
    ],
    "사람 수": [
        missing_bmi_people,
        len(df)
    ],
    "뇌졸중 발생 인원": [
        missing_bmi_stroke_count,
        overall_stroke_count
    ],
    "뇌졸중 비율": [
        missing_bmi_stroke_rate,
        overall_stroke_rate
    ]
})

st.dataframe(
    bmi_table.style.format({
        "사람 수": "{:,}명",
        "뇌졸중 발생 인원": "{:,}명",
        "뇌졸중 비율": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)


st.divider()


# ---------------------------------
# 5. 흡연 상태별 사람 수
# ---------------------------------

st.header("5. 흡연 상태별 사람 수")

st.write("""
흡연 상태(smoking_status)에 어떤 값이 있는지 확인하고,
각 상태에 해당하는 사람 수를 세어 봅니다.
""")

smoking_counts = (
    df["smoking_status"]
    .fillna("결측값")
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_counts,
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"전체 데이터 {len(df):,}명 중 흡연 상태별 인원입니다. "
    "결측값도 별도 항목으로 표시했습니다."
)

st.divider()

st.caption(
    "데이터 출처: "
    "https://github.com/greatsong/modudata"
)


import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ==========================================
# 1. 기본 설정
# ==========================================

APP_TITLE = "뇌졸중 예측 실습실"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

AVAILABLE_FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
]

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

st.set_page_config(
    page_title=f"채점표 | {APP_TITLE}",
    page_icon="📊",
    layout="wide",
)


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    required_columns = [
        "id",
        "age",
        "avg_glucose_level",
        "bmi",
        "hypertension",
        "heart_disease",
        "stroke",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"필요한 열이 없습니다: {missing_columns}"
        )

    return df[required_columns].copy()


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# ==========================================
# 3. 화면 제목과 속성 선택
# ==========================================

st.title("📊 채점표")

st.markdown("""
테스트 데이터를 이용하여 두 분류 모델이 얼마나 정확하게
뇌졸중 환자를 찾아내는지 확인합니다.

- **양성**: 뇌졸중 발생 (`stroke = 1`)
- **음성**: 뇌졸중 미발생 (`stroke = 0`)

혼동행렬과 정확도, 재현율, 정밀도, F1 점수를 살펴봅니다.
""")

st.divider()

st.subheader("① 입력 속성 고르기")

selected_features = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요.",
    options=AVAILABLE_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_NAMES[x],
)

if len(selected_features) < 2:
    st.warning(
        "입력 속성을 두 개 이상 선택해야 모델을 만들 수 있습니다."
    )
    st.stop()

st.write(
    "선택한 속성: "
    + ", ".join(
        FEATURE_NAMES[x] for x in selected_features
    )
)


# ==========================================
# 4. 데이터 정렬 및 훈련/테스트 분리
# ==========================================

st.divider()
st.subheader("② 데이터 나누기")

df = df.sort_values(
    "id",
    kind="stable"
).reset_index(drop=True)

X_all = df[selected_features].copy()

y_all = pd.to_numeric(
    df["stroke"],
    errors="coerce"
)

valid_target = y_all.isin([0, 1])

X_all = X_all.loc[valid_target].reset_index(drop=True)
y_all = y_all.loc[valid_target].astype(int).reset_index(drop=True)

# 10명씩 묶어서 각 묶음의 앞 3명을 테스트 데이터로 분리
test_mask = pd.Series(
    [i % 10 < 3 for i in range(len(X_all))]
)

X_test_raw = X_all.loc[test_mask].copy()
X_train_raw = X_all.loc[~test_mask].copy()

y_test = y_all.loc[test_mask].copy()
y_train = y_all.loc[~test_mask].copy()

X_train_raw = X_train_raw.reset_index(drop=True)
X_test_raw = X_test_raw.reset_index(drop=True)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

test_people = len(X_test_raw)
test_positive_people = int(y_test.sum())

st.write(
    f"테스트 데이터: **{test_people:,}명** / "
    f"그중 실제 뇌졸중 환자: **{test_positive_people:,}명**"
)

col1, col2 = st.columns(2)

col1.metric(
    "훈련 데이터",
    f"{len(X_train_raw):,}명"
)

col2.metric(
    "테스트 데이터",
    f"{test_people:,}명"
)

if test_people != 1533:
    st.warning(
        f"테스트 데이터가 {test_people:,}명입니다. "
        "예상 인원 1,533명과 다릅니다."
    )


# ==========================================
# 5. 결측값 처리
# ==========================================

st.divider()
st.subheader("③ 결측값 처리")

X_train = X_train_raw.copy()
X_test = X_test_raw.copy()

if "bmi" in selected_features:

    # 훈련 데이터의 중앙값만 사용
    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error(
            "훈련 데이터의 체질량지수 값이 없어 "
            "중앙값을 계산할 수 없습니다."
        )
        st.stop()

    train_missing = int(X_train["bmi"].isna().sum())
    test_missing = int(X_test["bmi"].isna().sum())

    X_train["bmi"] = X_train["bmi"].fillna(
        bmi_median
    )

    X_test["bmi"] = X_test["bmi"].fillna(
        bmi_median
    )

    st.write(
        f"BMI 결측값은 훈련 데이터 중앙값 "
        f"**{bmi_median:.2f}**으로 채웠습니다."
    )

    st.caption(
        f"훈련 데이터: {train_missing:,}개 / "
        f"테스트 데이터: {test_missing:,}개"
    )

else:

    st.write(
        "체질량지수를 선택하지 않았으므로 "
        "BMI 결측값 처리를 하지 않았습니다."
    )


# BMI 이외의 결측값 검사
if X_train.isna().any().any() or X_test.isna().any().any():

    missing_info = pd.concat(
        [
            X_train.isna().sum().rename("훈련 데이터"),
            X_test.isna().sum().rename("테스트 데이터"),
        ],
        axis=1
    )

    st.error(
        "선택한 입력 속성에 처리되지 않은 결측값이 있습니다."
    )

    st.dataframe(missing_info)
    st.stop()


# ==========================================
# 6. 표준화
# ==========================================

st.divider()
st.subheader("④ 데이터 크기 맞추기")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

st.write("""
훈련 데이터에서 평균과 표준편차를 계산하고,
같은 기준을 테스트 데이터에 적용했습니다.
""")


# ==========================================
# 7. 두 모델 학습
# ==========================================

st.divider()
st.subheader("⑤ 분류 모델 학습")

# 로지스틱 회귀
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(
    X_train_scaled,
    y_train
)

st.success("두 모델의 학습이 완료되었습니다.")


# ==========================================
# 8. 기준 모델
# ==========================================

majority_class = int(
    y_train.value_counts().idxmax()
)

# 훈련 데이터에서 더 많은 쪽만 계속 예측
baseline_predictions = [
    majority_class
] * len(y_test)


# ==========================================
# 9. 테스트 데이터 예측
# ==========================================

logistic_predictions = logistic_model.predict(
    X_test_scaled
)

tree_predictions = tree_model.predict(
    X_test_scaled
)


# ==========================================
# 10. 혼동행렬 계산
# ==========================================

def make_confusion_matrix(y_true, y_pred):

    # 양성 = 1, 음성 = 0
    tn = int(
        ((y_true == 0) & (y_pred == 0)).sum()
    )

    fp = int(
        ((y_true == 0) & (y_pred == 1)).sum()
    )

    fn = int(
        ((y_true == 1) & (y_pred == 0)).sum()
    )

    tp = int(
        ((y_true == 1) & (y_pred == 1)).sum()
    )

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


logistic_cm = make_confusion_matrix(
    y_test,
    logistic_predictions
)

tree_cm = make_confusion_matrix(
    y_test,
    tree_predictions
)

baseline_cm = make_confusion_matrix(
    y_test,
    baseline_predictions
)


# ==========================================
# 11. 혼동행렬 표 만들기
# ==========================================

def show_confusion_matrix(title, cm):

    st.subheader(title)

    tn = cm["TN"]
    fp = cm["FP"]
    fn = cm["FN"]
    tp = cm["TP"]

    actual_negative = tn + fp
    actual_positive = fn + tp

    predicted_negative = tn + fn
    predicted_positive = fp + tp

    total = tn + fp + fn + tp

    confusion_df = pd.DataFrame(
        [
            [
                f"TN: {tn:,}명",
                f"FP: {fp:,}명",
                f"{actual_negative:,}명"
            ],
            [
                f"FN: {fn:,}명",
                f"TP: {tp:,}명",
                f"{actual_positive:,}명"
            ],
            [
                f"{predicted_negative:,}명",
                f"{predicted_positive:,}명",
                f"{total:,}명"
            ]
        ],
        index=[
            "실제: 뇌졸중 아님",
            "실제: 뇌졸중",
            "열 합계"
        ],
        columns=[
            "예측: 뇌졸중 아님",
            "예측: 뇌졸중",
            "행 합계"
        ]
    )

    st.dataframe(
        confusion_df,
        use_container_width=True
    )

    st.caption(
        "TN: 실제 아님 / 예측 아님 · "
        "FP: 실제 아님 / 예측 뇌졸중 · "
        "FN: 실제 뇌졸중 / 예측 아님 · "
        "TP: 실제 뇌졸중 / 예측 뇌졸중"
    )


# ==========================================
# 12. 혼동행렬 표시
# ==========================================

st.divider()
st.header("⑥ 테스트 데이터 혼동행렬")

show_confusion_matrix(
    "로지스틱 회귀 (확률로 답하는 모델)",
    logistic_cm
)

show_confusion_matrix(
    "의사결정트리 (질문으로 답하는 모델)",
    tree_cm
)


# ==========================================
# 13. 성능 지표 계산
# ==========================================

def calculate_metrics(cm):

    tn = cm["TN"]
    fp = cm["FP"]
    fn = cm["FN"]
    tp = cm["TP"]

    total = tn + fp + fn + tp

    # 정확도
    accuracy = (
        (tp + tn) / total
        if total > 0
        else None
    )

    # 재현율: 실제 뇌졸중 환자 중 찾아낸 비율
    recall_denominator = tp + fn

    recall = (
        tp / recall_denominator
        if recall_denominator > 0
        else None
    )

    # 정밀도: 뇌졸중이라고 예측한 사람 중 실제 환자 비율
    precision_denominator = tp + fp

    if precision_denominator == 0:
        precision = None
        f1 = None

    else:
        precision = tp / precision_denominator

        if recall is None or precision + recall == 0:
            f1 = None
        else:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1,
    }


logistic_metrics = calculate_metrics(
    logistic_cm
)

tree_metrics = calculate_metrics(
    tree_cm
)

baseline_metrics = calculate_metrics(
    baseline_cm
)


# ==========================================
# 14. 성능 지표 비교 표
# ==========================================

st.divider()
st.header("⑦ 모델 성능 비교")

st.write("""
모든 지표는 테스트 데이터로 계산했습니다.

- 정확도: 전체 테스트 데이터 중 맞게 예측한 비율
- 재현율: 실제 뇌졸중 환자 중 찾아낸 비율
- 정밀도: 뇌졸중이라고 예측한 사람 중 실제 환자 비율
- F1: 정밀도와 재현율의 조화평균
""")

def format_metric(value):

    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


metrics_table = pd.DataFrame(
    [
        {
            "모델": "로지스틱 회귀 (확률로 답하는 모델)",
            "정확도": format_metric(logistic_metrics["정확도"]),
            "재현율": format_metric(logistic_metrics["재현율"]),
            "정밀도": format_metric(logistic_metrics["정밀도"]),
            "F1": format_metric(logistic_metrics["F1"]),
        },
        {
            "모델": "의사결정트리 (질문으로 답하는 모델)",
            "정확도": format_metric(tree_metrics["정확도"]),
            "재현율": format_metric(tree_metrics["재현율"]),
            "정밀도": format_metric(tree_metrics["정밀도"]),
            "F1": format_metric(tree_metrics["F1"]),
        },
        {
            "모델": "훈련용에서 많은 쪽으로만 답하는 모델",
            "정확도": format_metric(baseline_metrics["정확도"]),
            "재현율": format_metric(baseline_metrics["재현율"]),
            "정밀도": format_metric(baseline_metrics["정밀도"]),
            "F1": format_metric(baseline_metrics["F1"]),
        },
    ]
)

st.dataframe(
    metrics_table,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 15. 놓친 환자와 헛짚은 사람
# ==========================================

st.divider()
st.header("⑧ 놓친 환자와 헛짚은 사람")

st.write("""
- **놓친 환자(FN)**: 실제 뇌졸중 환자이지만 뇌졸중이 아니라고 예측한 사람
- **헛짚은 사람(FP)**: 실제 뇌졸중 환자가 아니지만 뇌졸중이라고 예측한 사람

아래에서 확인할 모델을 선택할 수 있습니다.
""")

selected_model_name = st.selectbox(
    "목록을 확인할 모델",
    [
        "로지스틱 회귀 (확률로 답하는 모델)",
        "의사결정트리 (질문으로 답하는 모델)",
        "훈련용에서 많은 쪽으로만 답하는 모델",
    ]
)

if selected_model_name.startswith("로지스틱 회귀"):

    selected_predictions = logistic_predictions
    selected_cm = logistic_cm

elif selected_model_name.startswith("의사결정트리"):

    selected_predictions = tree_predictions
    selected_cm = tree_cm

else:

    selected_predictions = baseline_predictions
    selected_cm = baseline_cm


# ==========================================
# 16. 실제 결과와 예측 결과 연결
# ==========================================

# 원본 ID와 기본 정보 보관
test_details = df.loc[
    valid_target
].copy()

test_details = test_details.loc[
    test_mask
].reset_index(drop=True)

test_details["실제 결과"] = y_test
test_details["예측 결과"] = selected_predictions

# 번호는 원본 데이터의 id
# 나이, 혈당, 고혈압, 심장병 표시
detail_columns = [
    "id",
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
    "실제 결과",
    "예측 결과",
]

# 놓친 환자: 실제 1, 예측 0
missed_patients = test_details[
    (test_details["실제 결과"] == 1)
    & (test_details["예측 결과"] == 0)
].copy()

# 헛짚은 사람: 실제 0, 예측 1
false_alarm_people = test_details[
    (test_details["실제 결과"] == 0)
    & (test_details["예측 결과"] == 1)
].copy()

# 나이가 많은 순서
missed_patients = missed_patients.sort_values(
    "age",
    ascending=False,
    na_position="last"
)

false_alarm_people = false_alarm_people.sort_values(
    "age",
    ascending=False,
    na_position="last"
)


# ==========================================
# 17. 목록 표시 함수
# ==========================================

def prepare_display_table(data):

    display_df = data[detail_columns].copy()

    display_df = display_df.rename(
        columns={
            "id": "번호",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부",
        }
    )

    display_df["고혈압 여부"] = display_df[
        "고혈압 여부"
    ].map({
        0: "없음",
        1: "있음"
    })

    display_df["심장병 여부"] = display_df[
        "심장병 여부"
    ].map({
        0: "없음",
        1: "있음"
    })

    return display_df[
        [
            "번호",
            "나이",
            "평균 혈당",
            "고혈압 여부",
            "심장병 여부",
        ]
    ]


# ==========================================
# 18. 두 목록 나란히 표시
# ==========================================

missed_count = len(missed_patients)
false_alarm_count = len(false_alarm_people)

left_col, right_col = st.columns(2)

with left_col:

    st.subheader(
        f"놓친 환자 (FN) · 전체 {missed_count:,}명"
    )

    st.caption(
        "실제 뇌졸중 환자이지만 뇌졸중이 아니라고 예측한 사람"
    )

    if missed_count > 0:

        st.dataframe(
            prepare_display_table(
                missed_patients.head(20)
            ),
            use_container_width=True,
            hide_index=True
        )

        if missed_count > 20:
            st.caption(
                f"전체 {missed_count:,}명 중 "
                "나이가 많은 앞의 20명만 표시했습니다."
            )

    else:

        st.success("놓친 환자가 없습니다.")


with right_col:

    st.subheader(
        f"헛짚은 사람 (FP) · 전체 {false_alarm_count:,}명"
    )

    st.caption(
        "실제 뇌졸중 환자가 아니지만 뇌졸중이라고 예측한 사람"
    )

    if false_alarm_count > 0:

        st.dataframe(
            prepare_display_table(
                false_alarm_people.head(20)
            ),
            use_container_width=True,
            hide_index=True
        )

        if false_alarm_count > 20:
            st.caption(
                f"전체 {false_alarm_count:,}명 중 "
                "나이가 많은 앞의 20명만 표시했습니다."
            )

    else:

        st.success("헛짚은 사람이 없습니다.")


# ==========================================
# 19. 혼동행렬과 목록 인원 일치 확인
# ==========================================

st.divider()

st.write(
    f"선택한 모델의 혼동행렬에서 FN은 "
    f"**{selected_cm['FN']:,}명**, "
    f"놓친 환자 목록은 **{missed_count:,}명**입니다. "
    f"FP는 **{selected_cm['FP']:,}명**, "
    f"헛짚은 사람 목록은 **{false_alarm_count:,}명**입니다."
)

if (
    selected_cm["FN"] == missed_count
    and selected_cm["FP"] == false_alarm_count
):
    st.success(
        "놓친 환자 목록의 인원은 FN과 같고, "
        "헛짚은 사람 목록의 인원은 FP와 같습니다."
    )
else:
    st.error(
        "목록 인원과 혼동행렬 값이 일치하지 않습니다. "
        "데이터 분리와 예측 결과를 확인해 주세요."
    )


# ==========================================
# 20. 마무리 안내
# ==========================================

st.divider()

st.caption("""
이 페이지는 분류 모델을 이해하기 위한 교육용 실습입니다.
실제 의료 진단이나 치료 판단을 대신할 수 없습니다.
""")

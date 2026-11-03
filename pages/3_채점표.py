
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

APP_TITLE = "뇌졸중 예측 실습실"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
]

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수(BMI)",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부",
}

MODEL_LOGISTIC = "로지스틱 회귀(확률로 답하는 모델)"
MODEL_TREE = "의사결정트리(질문으로 답하는 모델)"

st.set_page_config(
    page_title=f"{APP_TITLE} - 채점표",
    page_icon="🧠",
    layout="wide",
)

st.title("📋 채점표")
st.write(
    "테스트 데이터로 두 모델의 예측 결과를 확인하고 "
    "혼동행렬과 평가 지표를 비교합니다."
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

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
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"데이터에 필요한 열이 없습니다: {missing_columns}"
        )

    return df


try:
    df = load_data()
except Exception as error:
    st.error(f"데이터를 불러오지 못했습니다: {error}")
    st.stop()


# --------------------------------------------------
# 사용할 특성 선택
# --------------------------------------------------

st.subheader("1. 예측에 사용할 특성 선택")

selected_features = st.multiselect(
    "모델 학습에 사용할 항목을 선택하세요.",
    options=FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
    ],
    format_func=lambda x: FEATURE_LABELS[x],
)

if len(selected_features) < 2:
    st.warning("특성을 두 개 이상 선택해야 합니다.")
    st.stop()


# --------------------------------------------------
# 데이터 정리 및 훈련용·테스트용 데이터 나누기
# --------------------------------------------------

work_df = df.copy()

for column in FEATURES + ["stroke", "id"]:
    work_df[column] = pd.to_numeric(
        work_df[column],
        errors="coerce",
    )

work_df = work_df.dropna(subset=["stroke"])
work_df = work_df[
    work_df["stroke"].isin([0, 1])
]

work_df = work_df.sort_values("id").reset_index(drop=True)

X_all = work_df[selected_features].copy()
y_all = work_df["stroke"].astype(int).copy()

# 정렬한 데이터에서 10명 중 앞의 3명을 테스트용으로 사용
test_mask = pd.Series(
    [i % 10 < 3 for i in range(len(work_df))]
)

X_test = X_all.loc[test_mask].copy()
X_train = X_all.loc[~test_mask].copy()

y_test = y_all.loc[test_mask].copy()
y_train = y_all.loc[~test_mask].copy()

# BMI를 선택한 경우에만 훈련용 데이터의 중앙값으로 결측값 처리
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("훈련용 데이터에서 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    X_train.loc[:, "bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test.loc[:, "bmi"] = X_test["bmi"].fillna(bmi_median)

# 선택한 특성에 처리되지 않은 결측값이 있는지 확인
if X_train.isna().any().any() or X_test.isna().any().any():
    st.error(
        "선택한 특성에 결측값이 남아 있습니다. "
        "데이터를 확인해 주세요."
    )
    st.stop()

if len(X_train) == 0 or len(X_test) == 0:
    st.error("훈련용 또는 테스트용 데이터가 없습니다.")
    st.stop()


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
)

logistic_model.fit(X_train_scaled, y_train)
tree_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test_scaled)
tree_predictions = tree_model.predict(X_test)

# 훈련용 데이터에서 더 많은 쪽만 예측하는 기준 모델
majority_class = int(y_train.value_counts().idxmax())

baseline_predictions = [
    majority_class
] * len(y_test)


# --------------------------------------------------
# 테스트 데이터 정보
# --------------------------------------------------

test_count = len(y_test)
test_stroke_count = int((y_test == 1).sum())

st.subheader("2. 테스트 데이터")

st.write(
    f"테스트용 사람 수: **{test_count:,}명** "
    f"/ 그중 실제 뇌졸중 환자: **{test_stroke_count:,}명**"
)


# --------------------------------------------------
# 혼동행렬 계산
# --------------------------------------------------

def make_confusion_matrix(actual, predicted):
    """
    행: 실제 정답
    열: 모델 예측

                 예측 0       예측 1
    실제 0        TN           FP
    실제 1        FN           TP
    """

    actual = list(actual)
    predicted = list(predicted)

    tn = sum(
        1 for a, p in zip(actual, predicted)
        if a == 0 and p == 0
    )

    fp = sum(
        1 for a, p in zip(actual, predicted)
        if a == 0 and p == 1
    )

    fn = sum(
        1 for a, p in zip(actual, predicted)
        if a == 1 and p == 0
    )

    tp = sum(
        1 for a, p in zip(actual, predicted)
        if a == 1 and p == 1
    )

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def show_confusion_matrix(title, result):
    tn = result["TN"]
    fp = result["FP"]
    fn = result["FN"]
    tp = result["TP"]

    # 행 합계
    actual_negative = tn + fp
    actual_positive = fn + tp

    # 열 합계
    predicted_negative = tn + fn
    predicted_positive = fp + tp

    total = tn + fp + fn + tp

    # 표에 표시할 데이터
    matrix_df = pd.DataFrame(
        [
            [
                f"TN: {tn:,}명",
                f"FP: {fp:,}명",
                f"{actual_negative:,}명",
            ],
            [
                f"FN: {fn:,}명",
                f"TP: {tp:,}명",
                f"{actual_positive:,}명",
            ],
            [
                f"{predicted_negative:,}명",
                f"{predicted_positive:,}명",
                f"{total:,}명",
            ],
        ],
        index=[
            "실제: 뇌졸중 아님 (0)",
            "실제: 뇌졸중 (1)",
            "열 합계",
        ],
        columns=[
            "예측: 뇌졸중 아님 (0)",
            "예측: 뇌졸중 (1)",
            "행 합계",
        ],
    )

    st.markdown(f"#### {title}")

    st.caption(
        "TN: 정상인을 정상으로 예측 / "
        "FP: 정상인을 뇌졸중으로 예측 / "
        "FN: 환자를 정상으로 예측 / "
        "TP: 환자를 환자로 예측"
    )

    # 혼동행렬을 표로 직접 표시
    st.table(matrix_df)


# --------------------------------------------------
# 혼동행렬 화면
# --------------------------------------------------

st.subheader("3. 모델별 혼동행렬")

col1, col2 = st.columns(2)

logistic_cm = make_confusion_matrix(
    y_test,
    logistic_predictions,
)

tree_cm = make_confusion_matrix(
    y_test,
    tree_predictions,
)

with col1:
    show_confusion_matrix(
        MODEL_LOGISTIC,
        logistic_cm,
    )

with col2:
    show_confusion_matrix(
        MODEL_TREE,
        tree_cm,
    )


# --------------------------------------------------
# 평가 지표 계산
# --------------------------------------------------

def calculate_metrics(result):
    tn = result["TN"]
    fp = result["FP"]
    fn = result["FN"]
    tp = result["TP"]

    total = tn + fp + fn + tp

    accuracy = (
        (tp + tn) / total
        if total > 0
        else None
    )

    # 재현율 = 실제 뇌졸중 환자 중 맞게 예측한 비율
    actual_positive = tp + fn

    recall = (
        tp / actual_positive
        if actual_positive > 0
        else None
    )

    # 정밀도 = 뇌졸중이라고 예측한 사람 중 실제 환자 비율
    predicted_positive = tp + fp

    if predicted_positive == 0:
        precision = None
        f1 = None
    else:
        precision = tp / predicted_positive

        if recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0
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


def format_metric(value):
    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# --------------------------------------------------
# 기준 모델 혼동행렬 및 평가 지표
# --------------------------------------------------

baseline_cm = make_confusion_matrix(
    y_test,
    baseline_predictions,
)

baseline_metrics = calculate_metrics(baseline_cm)
logistic_metrics = calculate_metrics(logistic_cm)
tree_metrics = calculate_metrics(tree_cm)


# --------------------------------------------------
# 평가 지표 비교표
# --------------------------------------------------

st.subheader("4. 평가 지표 비교")

st.caption(
    "정확도 = 전체 예측 중 맞힌 비율 / "
    "재현율 = TP ÷ (TP + FN) / "
    "정밀도 = TP ÷ (TP + FP) / "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)"
)

metrics_df = pd.DataFrame(
    [
        {
            "모델": MODEL_LOGISTIC,
            "정확도": format_metric(logistic_metrics["정확도"]),
            "재현율": format_metric(logistic_metrics["재현율"]),
            "정밀도": format_metric(logistic_metrics["정밀도"]),
            "F1": format_metric(logistic_metrics["F1"]),
        },
        {
            "모델": MODEL_TREE,
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
    metrics_df,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 놓친 환자와 헛짚은 사람 목록
# --------------------------------------------------

st.subheader("5. 놓친 환자와 헛짚은 사람")

selected_model = st.selectbox(
    "목록을 확인할 모델을 선택하세요.",
    options=[
        MODEL_LOGISTIC,
        MODEL_TREE,
    ],
)

if selected_model == MODEL_LOGISTIC:
    selected_predictions = logistic_predictions
    selected_cm = logistic_cm
else:
    selected_predictions = tree_predictions
    selected_cm = tree_cm


# 테스트 데이터의 원래 정보와 예측 결과 연결
test_details = work_df.loc[
    test_mask
].copy().reset_index(drop=True)

test_details["실제값"] = y_test.reset_index(drop=True)
test_details["예측값"] = list(selected_predictions)

# FN: 실제 1인데 0이라고 예측한 사람
missed_patients = test_details[
    (test_details["실제값"] == 1)
    & (test_details["예측값"] == 0)
].copy()

# FP: 실제 0인데 1이라고 예측한 사람
false_alarms = test_details[
    (test_details["실제값"] == 0)
    & (test_details["예측값"] == 1)
].copy()

# 나이가 많은 순서로 정렬
missed_patients = missed_patients.sort_values(
    "age",
    ascending=False,
)

false_alarms = false_alarms.sort_values(
    "age",
    ascending=False,
)


def make_person_table(data):
    result = data[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease",
        ]
    ].copy()

    result.columns = [
        "번호",
        "나이",
        "평균 혈당",
        "고혈압 여부",
        "심장병 여부",
    ]

    result = result.rename_axis("순번").reset_index(drop=True)
    result.index = result.index + 1

    return result


col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"#### 놓친 환자 (FN): {len(missed_patients):,}명"
    )

    st.caption(
        "실제로 뇌졸중 환자이지만, 모델이 아니라고 예측한 사람"
    )

    if len(missed_patients) == 0:
        st.info("놓친 환자가 없습니다.")
    else:
        st.dataframe(
            make_person_table(
                missed_patients.head(20)
            ),
            use_container_width=True,
        )

        if len(missed_patients) > 20:
            st.caption(
                f"전체 {len(missed_patients):,}명 중 "
                "나이가 많은 순서로 앞의 20명만 표시했습니다."
            )


with col2:
    st.markdown(
        f"#### 헛짚은 사람 (FP): {len(false_alarms):,}명"
    )

    st.caption(
        "실제로 뇌졸중 환자가 아니지만, 모델이 환자라고 예측한 사람"
    )

    if len(false_alarms) == 0:
        st.info("헛짚은 사람이 없습니다.")
    else:
        st.dataframe(
            make_person_table(
                false_alarms.head(20)
            ),
            use_container_width=True,
        )

        if len(false_alarms) > 20:
            st.caption(
                f"전체 {len(false_alarms):,}명 중 "
                "나이가 많은 순서로 앞의 20명만 표시했습니다."
            )


# --------------------------------------------------
# 목록과 혼동행렬의 FN, FP 수 비교
# --------------------------------------------------

st.divider()

st.subheader("6. 목록과 혼동행렬의 인원수 확인")

fn_list_count = len(missed_patients)
fp_list_count = len(false_alarms)

fn_matrix_count = selected_cm["FN"]
fp_matrix_count = selected_cm["FP"]

st.write(
    f"놓친 환자 목록: **{fn_list_count:,}명** "
    f"/ 혼동행렬 FN: **{fn_matrix_count:,}명**"
)

st.write(
    f"헛짚은 사람 목록: **{fp_list_count:,}명** "
    f"/ 혼동행렬 FP: **{fp_matrix_count:,}명**"
)

if (
    fn_list_count == fn_matrix_count
    and fp_list_count == fp_matrix_count
):
    st.success(
        "확인 완료: 두 목록의 인원수가 "
        "혼동행렬의 FN, FP 인원수와 일치합니다."
    )
else:
    st.error(
        "목록과 혼동행렬의 인원수가 일치하지 않습니다. "
        "데이터 분할과 예측 결과를 확인해 주세요."
    )

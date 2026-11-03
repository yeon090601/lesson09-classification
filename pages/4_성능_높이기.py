
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. 기본 설정
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
    page_title=f"{APP_TITLE} - 성능 높이기",
    page_icon="🧠",
    layout="wide",
)

st.title("🚀 성능 높이기")

st.write(
    "뇌졸중 환자는 전체 사람 중에서 매우 적습니다. "
    "이 페이지에서는 환자와 비환자의 학습 가중치를 조정하기 전과 "
    "조정한 후의 예측 결과를 비교합니다."
)


# --------------------------------------------------
# 2. 데이터 불러오기
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
# 3. 입력 특성 선택
# --------------------------------------------------

st.subheader("1. 예측에 사용할 속성 선택")

selected_features = st.multiselect(
    "모델의 입력으로 사용할 속성을 고르세요.",
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
    st.warning("속성을 두 개 이상 선택해 주세요.")
    st.stop()


# --------------------------------------------------
# 4. 데이터 정리 및 분할
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

# 채점표 페이지와 같은 방법으로 10명 중 앞의 3명을 테스트용으로 분리
test_mask = pd.Series(
    [i % 10 < 3 for i in range(len(work_df))]
)

X_test = X_all.loc[test_mask].copy()
X_train = X_all.loc[~test_mask].copy()

y_test = y_all.loc[test_mask].copy()
y_train = y_all.loc[~test_mask].copy()

if len(X_train) == 0 or len(X_test) == 0:
    st.error("훈련용 또는 테스트용 데이터가 없습니다.")
    st.stop()


# --------------------------------------------------
# 5. 결측값 처리
# --------------------------------------------------

if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("훈련용 데이터에서 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    X_train.loc[:, "bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test.loc[:, "bmi"] = X_test["bmi"].fillna(bmi_median)

if X_train.isna().any().any() or X_test.isna().any().any():
    missing_names = [
        FEATURE_LABELS[column]
        for column in selected_features
        if X_train[column].isna().any()
        or X_test[column].isna().any()
    ]

    st.error(
        "선택한 속성에 결측값이 남아 있습니다: "
        + ", ".join(missing_names)
        + ". BMI 이외의 결측값을 확인해 주세요."
    )
    st.stop()


# --------------------------------------------------
# 6. 데이터 정보
# --------------------------------------------------

st.subheader("2. 데이터 분할 결과")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric("전체 사람 수", f"{len(work_df):,}명")

with info2:
    st.metric("훈련용 사람 수", f"{len(X_train):,}명")

with info3:
    st.metric("테스트용 사람 수", f"{len(X_test):,}명")

st.write(
    f"훈련용 뇌졸중 환자: **{int((y_train == 1).sum()):,}명** / "
    f"훈련용 전체: **{len(y_train):,}명**"
)

st.write(
    f"테스트용 뇌졸중 환자: **{int((y_test == 1).sum()):,}명** / "
    f"테스트용 전체: **{len(y_test):,}명**"
)

st.caption(
    "데이터는 ID 순서로 정렬한 뒤 10명 중 앞의 3명을 테스트용, "
    "나머지 7명을 훈련용으로 사용합니다."
)


# --------------------------------------------------
# 7. 모델 학습
# --------------------------------------------------

st.subheader("3. 가중치 조정 전후 모델 학습")

# 로지스틱 회귀에 사용할 표준화
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 가중치 조정 전
logistic_before = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
)


# 가중치 조정 후
logistic_after = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced",
)

tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced",
)


# 학습
logistic_before.fit(X_train_scaled, y_train)
logistic_after.fit(X_train_scaled, y_train)

tree_before.fit(X_train, y_train)
tree_after.fit(X_train, y_train)


# 테스트 데이터 예측
predictions = {
    MODEL_LOGISTIC: {
        "before": logistic_before.predict(X_test_scaled),
        "after": logistic_after.predict(X_test_scaled),
    },
    MODEL_TREE: {
        "before": tree_before.predict(X_test),
        "after": tree_after.predict(X_test),
    },
}

models = {
    MODEL_LOGISTIC: {
        "before": logistic_before,
        "after": logistic_after,
        "scaled": True,
    },
    MODEL_TREE: {
        "before": tree_before,
        "after": tree_after,
        "scaled": False,
    },
}

st.success(
    "두 모델의 학습을 완료했습니다. "
    "가중치 조정 전후에 같은 테스트 데이터를 사용합니다."
)


# --------------------------------------------------
# 8. 혼동행렬 계산
# --------------------------------------------------

def make_confusion_matrix(actual, predicted):
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


# --------------------------------------------------
# 9. 평가 지표 계산
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

    actual_positive = tp + fn

    recall = (
        tp / actual_positive
        if actual_positive > 0
        else None
    )

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
# 10. 모델별 전후 결과 계산
# --------------------------------------------------

results = {}

for model_name in [MODEL_LOGISTIC, MODEL_TREE]:

    before_cm = make_confusion_matrix(
        y_test,
        predictions[model_name]["before"],
    )

    after_cm = make_confusion_matrix(
        y_test,
        predictions[model_name]["after"],
    )

    before_metrics = calculate_metrics(before_cm)
    after_metrics = calculate_metrics(after_cm)

    results[model_name] = {
        "before_cm": before_cm,
        "after_cm": after_cm,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
    }


# --------------------------------------------------
# 11. 혼동행렬과 평가 지표 표
# --------------------------------------------------

st.subheader("4. 가중치 조정 전후 성능 비교")

st.write(
    "각 모델의 혼동행렬 네 칸과 네 가지 평가 지표를 "
    "가중치 조정 전후로 비교합니다."
)

st.caption(
    "TN: 정상인을 정상으로 예측 / "
    "FP: 정상인을 환자로 예측 / "
    "FN: 환자를 정상으로 예측 / "
    "TP: 환자를 환자로 예측"
)

st.caption(
    "정확도 = (TP + TN) ÷ 전체 / "
    "재현율 = TP ÷ (TP + FN) / "
    "정밀도 = TP ÷ (TP + FP) / "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)"
)


def make_comparison_table(model_name):
    data = results[model_name]

    before_cm = data["before_cm"]
    after_cm = data["after_cm"]

    before_metrics = data["before_metrics"]
    after_metrics = data["after_metrics"]

    rows = [
        {
            "구분": "TN",
            "가중치 조정 전": before_cm["TN"],
            "가중치 조정 후": after_cm["TN"],
        },
        {
            "구분": "FP",
            "가중치 조정 전": before_cm["FP"],
            "가중치 조정 후": after_cm["FP"],
        },
        {
            "구분": "FN",
            "가중치 조정 전": before_cm["FN"],
            "가중치 조정 후": after_cm["FN"],
        },
        {
            "구분": "TP",
            "가중치 조정 전": before_cm["TP"],
            "가중치 조정 후": after_cm["TP"],
        },
        {
            "구분": "정확도",
            "가중치 조정 전": format_metric(
                before_metrics["정확도"]
            ),
            "가중치 조정 후": format_metric(
                after_metrics["정확도"]
            ),
        },
        {
            "구분": "재현율",
            "가중치 조정 전": format_metric(
                before_metrics["재현율"]
            ),
            "가중치 조정 후": format_metric(
                after_metrics["재현율"]
            ),
        },
        {
            "구분": "정밀도",
            "가중치 조정 전": format_metric(
                before_metrics["정밀도"]
            ),
            "가중치 조정 후": format_metric(
                after_metrics["정밀도"]
            ),
        },
        {
            "구분": "F1",
            "가중치 조정 전": format_metric(
                before_metrics["F1"]
            ),
            "가중치 조정 후": format_metric(
                after_metrics["F1"]
            ),
        },
    ]

    return pd.DataFrame(rows)


col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### {MODEL_LOGISTIC}")
    st.dataframe(
        make_comparison_table(MODEL_LOGISTIC),
        use_container_width=True,
        hide_index=True,
    )

with col2:
    st.markdown(f"#### {MODEL_TREE}")
    st.dataframe(
        make_comparison_table(MODEL_TREE),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# 12. 환자 발견 및 오류 인원 변화
# --------------------------------------------------

st.subheader("5. 환자 발견 결과 변화")

def show_change(label, before, after):
    st.write(
        f"**{label}:** {before:,}명 → {after:,}명"
    )


for model_name in [MODEL_LOGISTIC, MODEL_TREE]:

    before_cm = results[model_name]["before_cm"]
    after_cm = results[model_name]["after_cm"]

    st.markdown(f"#### {model_name}")

    found_before = before_cm["TP"]
    found_after = after_cm["TP"]

    missed_before = before_cm["FN"]
    missed_after = after_cm["FN"]

    false_before = before_cm["FP"]
    false_after = after_cm["FP"]

    col1, col2, col3 = st.columns(3)

    with col1:
        show_change(
            "찾아낸 환자",
            found_before,
            found_after,
        )

    with col2:
        show_change(
            "놓친 환자",
            missed_before,
            missed_after,
        )

    with col3:
        show_change(
            "헛짚은 사람",
            false_before,
            false_after,
        )


# --------------------------------------------------
# 13. 그래프용 예측 확률 함수
# --------------------------------------------------

def get_probabilities(model, rows, feature_names, scaled):
    input_df = pd.DataFrame(
        rows,
        columns=feature_names,
    )

    if scaled:
        input_data = scaler.transform(input_df)
    else:
        input_data = input_df

    return model.predict_proba(input_data)[:, 1]


# --------------------------------------------------
# 14. 그래프 범위 계산
# --------------------------------------------------

def make_axis_range(series):
    minimum = float(series.min())
    maximum = float(series.max())

    if minimum == maximum:
        padding = max(abs(minimum) * 0.1, 1)
    else:
        padding = (maximum - minimum) * 0.05

    return (
        minimum - padding,
        maximum + padding,
    )


def make_values(minimum, maximum, count):
    if count <= 1:
        return [minimum]

    return [
        minimum + (maximum - minimum) * i / (count - 1)
        for i in range(count)
    ]


# --------------------------------------------------
# 15. 실제 테스트 데이터 산점도
# --------------------------------------------------

def add_actual_points_2d(fig, x_feature, y_feature):
    actual = y_test.reset_index(drop=True)
    x_values = X_test[x_feature].reset_index(drop=True)
    y_values = X_test[y_feature].reset_index(drop=True)

    for label, color, name in [
        (0, "#2878B5", "실제 뇌졸중 아님"),
        (1, "#E45756", "실제 뇌졸중"),
    ]:
        indices = [
            i for i in range(len(actual))
            if int(actual.iloc[i]) == label
        ]

        fig.add_trace(
            go.Scatter(
                x=[x_values.iloc[i] for i in indices],
                y=[y_values.iloc[i] for i in indices],
                mode="markers",
                name=name,
                marker={
                    "color": color,
                    "size": 7,
                    "opacity": 0.65,
                },
            )
        )


def add_actual_points_3d(
    fig,
    x_feature,
    y_feature,
    z_feature,
):
    actual = y_test.reset_index(drop=True)

    x_values = X_test[x_feature].reset_index(drop=True)
    y_values = X_test[y_feature].reset_index(drop=True)
    z_values = X_test[z_feature].reset_index(drop=True)

    for label, color, name in [
        (0, "#2878B5", "실제 뇌졸중 아님"),
        (1, "#E45756", "실제 뇌졸중"),
    ]:
        indices = [
            i for i in range(len(actual))
            if int(actual.iloc[i]) == label
        ]

        fig.add_trace(
            go.Scatter3d(
                x=[x_values.iloc[i] for i in indices],
                y=[y_values.iloc[i] for i in indices],
                z=[z_values.iloc[i] for i in indices],
                mode="markers",
                name=name,
                marker={
                    "color": color,
                    "size": 4,
                    "opacity": 0.7,
                },
            )
        )


# --------------------------------------------------
# 16. 2차원 결정 경계 계산
# --------------------------------------------------

def make_2d_probability_grid(
    model,
    x_feature,
    y_feature,
    scaled,
):
    x_min, x_max = make_axis_range(X_test[x_feature])
    y_min, y_max = make_axis_range(X_test[y_feature])

    x_values = make_values(x_min, x_max, 55)
    y_values = make_values(y_min, y_max, 55)

    rows = []

    for y_value in y_values:
        for x_value in x_values:
            row = {
                feature: float(X_train[feature].median())
                for feature in selected_features
            }

            row[x_feature] = x_value
            row[y_feature] = y_value

            rows.append(
                [row[feature] for feature in selected_features]
            )

    probabilities = get_probabilities(
        model,
        rows,
        selected_features,
        scaled,
    )

    probability_grid = []

    for row_index in range(len(y_values)):
        start = row_index * len(x_values)
        end = start + len(x_values)

        probability_grid.append(
            probabilities[start:end].tolist()
        )

    return x_values, y_values, probability_grid


# --------------------------------------------------
# 17. 2차원 그래프
# --------------------------------------------------

def draw_2d_graph(model_name, x_feature, y_feature):
    model_info = models[model_name]

    before_model = model_info["before"]
    after_model = model_info["after"]
    scaled = model_info["scaled"]

    x_values, y_values, before_grid = make_2d_probability_grid(
        before_model,
        x_feature,
        y_feature,
        scaled,
    )

    _, _, after_grid = make_2d_probability_grid(
        after_model,
        x_feature,
        y_feature,
        scaled,
    )

    fig = go.Figure()

    add_actual_points_2d(
        fig,
        x_feature,
        y_feature,
    )

    before_flat = [
        value
        for row in before_grid
        for value in row
    ]

    after_flat = [
        value
        for row in after_grid
        for value in row
    ]

    before_min = min(before_flat)
    before_max = max(before_flat)

    after_min = min(after_flat)
    after_max = max(after_flat)

    before_exists = (
        before_min <= 0.5 <= before_max
        and before_min != before_max
    )

    after_exists = (
        after_min <= 0.5 <= after_max
        and after_min != after_max
    )

    if before_exists:
        fig.add_trace(
            go.Contour(
                x=x_values,
                y=y_values,
                z=before_grid,
                name="가중치 조정 전 0.5 경계선",
                showscale=False,
                contours={
                    "start": 0.5,
                    "end": 0.5,
                    "size": 0.01,
                    "coloring": "lines",
                },
                line={
                    "color": "#F2A541",
                    "width": 4,
                    "dash": "dash",
                },
                hoverinfo="skip",
            )
        )

    if after_exists:
        fig.add_trace(
            go.Contour(
                x=x_values,
                y=y_values,
                z=after_grid,
                name="가중치 조정 후 0.5 경계선",
                showscale=False,
                contours={
                    "start": 0.5,
                    "end": 0.5,
                    "size": 0.01,
                    "coloring": "lines",
                },
                line={
                    "color": "#8E44AD",
                    "width": 4,
                },
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"{model_name} - 2차원 산점도와 결정 경계",
        xaxis_title=FEATURE_LABELS[x_feature],
        yaxis_title=FEATURE_LABELS[y_feature],
        height=650,
        legend_title="구분",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    if not before_exists:
        st.info(
            "가중치 조정 전의 0.5 경계선은 "
            "현재 그림 범위 안에 나타나지 않습니다."
        )

    if not after_exists:
        st.info(
            "가중치 조정 후의 0.5 경계선은 "
            "현재 그림 범위 안에 나타나지 않습니다."
        )

    st.caption(
        "경계선은 선택하지 않은 나머지 속성을 "
        "훈련용 데이터의 중앙값으로 고정한 상태에서 계산했습니다."
    )


# --------------------------------------------------
# 18. 3차원 결정 평면 계산
# --------------------------------------------------

def make_3d_boundary(
    model,
    x_feature,
    y_feature,
    z_feature,
    scaled,
):
    x_min, x_max = make_axis_range(X_test[x_feature])
    y_min, y_max = make_axis_range(X_test[y_feature])
    z_min, z_max = make_axis_range(X_test[z_feature])

    x_values = make_values(x_min, x_max, 25)
    y_values = make_values(y_min, y_max, 25)
    z_values = make_values(z_min, z_max, 55)

    rows = []

    for y_value in y_values:
        for x_value in x_values:
            for z_value in z_values:
                row = {
                    feature: float(X_train[feature].median())
                    for feature in selected_features
                }

                row[x_feature] = x_value
                row[y_feature] = y_value
                row[z_feature] = z_value

                rows.append(
                    [
                        row[feature]
                        for feature in selected_features
                    ]
                )

    probabilities = get_probabilities(
        model,
        rows,
        selected_features,
        scaled,
    )

    boundary_grid = []
    crossing_count = 0

    position = 0

    for y_index in range(len(y_values)):
        boundary_row = []

        for x_index in range(len(x_values)):
            current_probs = probabilities[
                position:position + len(z_values)
            ]

            position += len(z_values)

            crossing = None

            for i in range(len(z_values) - 1):
                p1 = float(current_probs[i])
                p2 = float(current_probs[i + 1])

                if (p1 - 0.5) * (p2 - 0.5) <= 0:
                    if p1 == p2:
                        crossing = (
                            z_values[i] + z_values[i + 1]
                        ) / 2
                    else:
                        ratio = (0.5 - p1) / (p2 - p1)

                        crossing = (
                            z_values[i]
                            + ratio * (
                                z_values[i + 1] - z_values[i]
                            )
                        )

                    break

            if crossing is not None:
                crossing_count += 1

            boundary_row.append(crossing)

        boundary_grid.append(boundary_row)

    return (
        x_values,
        y_values,
        boundary_grid,
        crossing_count,
    )


# --------------------------------------------------
# 19. 3차원 그래프
# --------------------------------------------------

def draw_3d_graph(model_name, x_feature, y_feature, z_feature):
    model_info = models[model_name]

    before_model = model_info["before"]
    after_model = model_info["after"]
    scaled = model_info["scaled"]

    (
        x_values,
        y_values,
        before_surface,
        before_count,
    ) = make_3d_boundary(
        before_model,
        x_feature,
        y_feature,
        z_feature,
        scaled,
    )

    (
        _,
        _,
        after_surface,
        after_count,
    ) = make_3d_boundary(
        after_model,
        x_feature,
        y_feature,
        z_feature,
        scaled,
    )

    fig = go.Figure()

    add_actual_points_3d(
        fig,
        x_feature,
        y_feature,
        z_feature,
    )

    if before_count > 0:
        fig.add_trace(
            go.Surface(
                x=x_values,
                y=y_values,
                z=before_surface,
                name="가중치 조정 전 0.5 평면",
                showscale=False,
                opacity=0.48,
                colorscale=[
                    [0, "#F2A541"],
                    [1, "#F2A541"],
                ],
                connectgaps=False,
                hoverinfo="skip",
            )
        )

    if after_count > 0:
        fig.add_trace(
            go.Surface(
                x=x_values,
                y=y_values,
                z=after_surface,
                name="가중치 조정 후 0.5 평면",
                showscale=False,
                opacity=0.48,
                colorscale=[
                    [0, "#8E44AD"],
                    [1, "#8E44AD"],
                ],
                connectgaps=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"{model_name} - 3차원 산점도와 결정 평면",
        height=750,
        scene={
            "xaxis_title": FEATURE_LABELS[x_feature],
            "yaxis_title": FEATURE_LABELS[y_feature],
            "zaxis_title": FEATURE_LABELS[z_feature],
        },
        legend_title="구분",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    if before_count == 0:
        st.info(
            "가중치 조정 전의 0.5 평면은 "
            "현재 그림 범위 안에서 찾을 수 없습니다."
        )

    if after_count == 0:
        st.info(
            "가중치 조정 후의 0.5 평면은 "
            "현재 그림 범위 안에서 찾을 수 없습니다."
        )

    st.caption(
        "3차원 평면은 선택한 세 속성을 축으로 사용합니다. "
        "선택하지 않은 나머지 속성은 훈련용 중앙값으로 고정했습니다. "
        "마우스로 그래프를 돌려 여러 방향에서 확인할 수 있습니다."
    )


# --------------------------------------------------
# 20. 산점도와 결정 경계 선택
# --------------------------------------------------

st.subheader("6. 산점도와 0.5 결정 경계 비교")

st.write(
    "점의 색은 실제 뇌졸중 여부를 나타냅니다. "
    "주황색은 가중치 조정 전, 보라색은 가중치 조정 후의 "
    "0.5 결정 경계입니다."
)

if len(selected_features) == 2:

    x_feature = selected_features[0]
    y_feature = selected_features[1]

    st.write(
        f"가로축: **{FEATURE_LABELS[x_feature]}** / "
        f"세로축: **{FEATURE_LABELS[y_feature]}**"
    )

    for model_name in [MODEL_LOGISTIC, MODEL_TREE]:
        draw_2d_graph(
            model_name,
            x_feature,
            y_feature,
        )

else:

    st.write("3차원 그래프에 사용할 세 가지 속성을 고르세요.")

    axis_features = st.multiselect(
        "3차원 그래프의 축 선택",
        options=selected_features,
        default=selected_features[:3],
        format_func=lambda x: FEATURE_LABELS[x],
        key="three_d_features",
    )

    if len(axis_features) != 3:
        st.warning("3차원 그래프를 그리려면 속성을 정확히 세 개 골라야 합니다.")
    else:
        x_feature = axis_features[0]
        y_feature = axis_features[1]
        z_feature = axis_features[2]

        st.write(
            f"가로축: **{FEATURE_LABELS[x_feature]}** / "
            f"세로축: **{FEATURE_LABELS[y_feature]}** / "
            f"높이축: **{FEATURE_LABELS[z_feature]}**"
        )

        for model_name in [MODEL_LOGISTIC, MODEL_TREE]:
            draw_3d_graph(
                model_name,
                x_feature,
                y_feature,
                z_feature,
            )


# --------------------------------------------------
# 21. 마무리 안내
# --------------------------------------------------

st.divider()

st.subheader("7. 결과 해석하기")

st.markdown(
    """
    - **찾아낸 환자(TP)**: 실제 환자를 뇌졸중이라고 올바르게 예측한 사람입니다.
    - **놓친 환자(FN)**: 실제 환자인데 뇌졸중이 아니라고 예측한 사람입니다.
    - **헛짚은 사람(FP)**: 실제 환자가 아닌데 뇌졸중이라고 예측한 사람입니다.
    - **재현율**: 실제 환자를 얼마나 많이 찾아냈는지 보여 줍니다.
    - **정밀도**: 환자라고 예측한 사람 중 실제 환자가 얼마나 되는지 보여 줍니다.

    가중치를 조정하면 놓친 환자가 줄어들 수 있지만,
    헛짚은 사람이 늘어날 수도 있습니다.
    정확도 하나만 보지 말고 재현율과 정밀도도 함께 비교해 보세요.
    """
)

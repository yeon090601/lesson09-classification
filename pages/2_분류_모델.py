
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    page_title=f"분류 모델 | {APP_TITLE}",
    page_icon="🤖",
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
# 3. 화면 제목과 입력 속성 선택
# ==========================================

st.title("🤖 분류 모델")

st.markdown("""
이 페이지에서는 두 가지 분류 모델을 학습하고 성능을 비교합니다.

- **로지스틱 회귀**: 확률로 답하는 모델
- **의사결정트리**: 질문으로 답하는 모델

뇌졸중이 발생한 사람(`stroke=1`)을 양성으로,
발생하지 않은 사람(`stroke=0`)을 음성으로 분류합니다.
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
# 4. 데이터 정리와 훈련/테스트 분리
# ==========================================

st.divider()
st.subheader("② 훈련 데이터와 테스트 데이터 나누기")

# 번호 순서로 정렬
df = df.sort_values(
    "id",
    kind="stable"
).reset_index(drop=True)

# 입력 속성과 정답 준비
X_all = df[selected_features].copy()

y_all = pd.to_numeric(
    df["stroke"],
    errors="coerce"
)

# 정답이 비어 있거나 0, 1이 아닌 행은 제외
valid_target = y_all.isin([0, 1])

X_all = X_all.loc[valid_target].reset_index(drop=True)
y_all = y_all.loc[valid_target].astype(int).reset_index(drop=True)

# 10명씩 묶어서 각 묶음의 앞 3명을 테스트 데이터로 사용
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

st.write(
    f"전체 사용 데이터: **{len(X_all):,}명**"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "훈련 데이터",
    f"{len(X_train_raw):,}명"
)

col2.metric(
    "테스트 데이터",
    f"{len(X_test_raw):,}명"
)

col3.metric(
    "테스트 데이터 목표",
    "1,533명"
)

if len(X_test_raw) != 1533:
    st.warning(
        f"현재 데이터의 테스트 인원은 {len(X_test_raw):,}명입니다. "
        "데이터 행 수가 예상과 다를 수 있습니다."
    )


# ==========================================
# 5. 결측값 처리
# ==========================================

st.divider()
st.subheader("③ 결측값 처리")

X_train = X_train_raw.copy()
X_test = X_test_raw.copy()

if "bmi" in selected_features:

    # 중앙값은 훈련 데이터에서만 계산
    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error(
            "훈련 데이터에 체질량지수 값이 없어 "
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
        f"체질량지수의 훈련 데이터 중앙값 **{bmi_median:.2f}**으로 "
        "훈련용과 테스트용의 빈 값을 채웠습니다."
    )

    st.caption(
        f"훈련 데이터에서 채운 값: {train_missing:,}개 / "
        f"테스트 데이터에서 채운 값: {test_missing:,}개"
    )

else:
    st.write(
        "체질량지수를 선택하지 않았으므로 BMI 결측값 처리를 하지 않았습니다."
    )


# 선택한 다른 속성에 결측값이 있는지 확인
if X_train.isna().any().any() or X_test.isna().any().any():

    missing_info = pd.concat(
        [
            X_train.isna().sum().rename("훈련 데이터"),
            X_test.isna().sum().rename("테스트 데이터"),
        ],
        axis=1
    )

    st.error(
        "선택한 속성 중 BMI 이외의 결측값이 있습니다. "
        "해당 값의 처리 방법을 먼저 확인해야 합니다."
    )

    st.dataframe(missing_info)
    st.stop()


# ==========================================
# 6. 크기 맞추기
# ==========================================

st.divider()
st.subheader("④ 데이터 크기 맞추기")

scaler = StandardScaler()

# 훈련 데이터로만 평균과 표준편차 계산
X_train_scaled = scaler.fit_transform(X_train)

# 테스트 데이터에는 훈련 데이터의 기준 적용
X_test_scaled = scaler.transform(X_test)

st.write("""
훈련 데이터의 평균과 표준편차를 기준으로
입력 속성의 크기를 맞췄습니다.

테스트 데이터에는 훈련 데이터에서 계산한 기준만 적용했습니다.
""")


# ==========================================
# 7. 로지스틱 회귀 모델
# ==========================================

st.divider()
st.subheader("⑤ 로지스틱 회귀 (확률로 답하는 모델)")

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_train_accuracy = logistic_model.score(
    X_train_scaled,
    y_train
)

logistic_test_accuracy = logistic_model.score(
    X_test_scaled,
    y_test
)


# ==========================================
# 8. 의사결정트리 모델
# ==========================================

st.divider()
st.subheader("⑥ 의사결정트리 (질문으로 답하는 모델)")

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(
    X_train_scaled,
    y_train
)

tree_train_accuracy = tree_model.score(
    X_train_scaled,
    y_train
)

tree_test_accuracy = tree_model.score(
    X_test_scaled,
    y_test
)


# ==========================================
# 9. 아무 입력도 보지 않는 기준 모델
# ==========================================

# 훈련 데이터에서 더 많은 정답을 항상 예측
majority_class = int(
    y_train.value_counts().idxmax()
)

baseline_train_accuracy = (
    y_train == majority_class
).mean()

baseline_test_accuracy = (
    y_test == majority_class
).mean()


# ==========================================
# 10. 정확도 카드
# ==========================================

st.divider()
st.header("⑦ 모델 정확도 비교")

st.write("""
카드의 큰 숫자는 테스트 데이터의 정확도입니다.
아래에는 훈련 데이터와 테스트 데이터의 정확도를 함께 표시합니다.
""")

card1, card2, card3 = st.columns(3)

with card1:

    st.metric(
        "로지스틱 회귀",
        f"{logistic_test_accuracy * 100:.2f}%"
    )

    st.caption(
        f"훈련 데이터: {logistic_train_accuracy * 100:.2f}%"
    )

    st.caption(
        f"테스트 데이터: {logistic_test_accuracy * 100:.2f}%"
    )


with card2:

    st.metric(
        "의사결정트리",
        f"{tree_test_accuracy * 100:.2f}%"
    )

    st.caption(
        f"훈련 데이터: {tree_train_accuracy * 100:.2f}%"
    )

    st.caption(
        f"테스트 데이터: {tree_test_accuracy * 100:.2f}%"
    )


with card3:

    baseline_name = (
        "뇌졸중"
        if majority_class == 1
        else "뇌졸중 아님"
    )

    st.metric(
        "입력을 보지 않는 기준 모델",
        f"{baseline_test_accuracy * 100:.2f}%"
    )

    st.caption(
        f"훈련 데이터: {baseline_train_accuracy * 100:.2f}%"
    )

    st.caption(
        f"테스트 데이터: {baseline_test_accuracy * 100:.2f}%"
    )

    st.caption(
        f"항상 '{baseline_name}'으로 답합니다."
    )


st.info(
    "기준 모델은 입력 속성을 하나도 사용하지 않고 "
    "훈련 데이터에서 더 많은 쪽의 정답만 반복해서 예측합니다."
)


# ==========================================
# 11. 두 속성 선택
# ==========================================

st.divider()
st.header("⑧ 두 속성으로 모델의 판단 살펴보기")

st.write("""
선택한 입력 속성 중 두 가지를 골라 가로축과 세로축으로 사용합니다.

점의 색은 테스트 데이터의 실제 뇌졸중 여부를 나타냅니다.
배경의 옅은 색은 의사결정트리가 나눈 영역을 나타냅니다.
선은 로지스틱 회귀가 확률 0.5를 기준으로 나눈 경계입니다.
""")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:

    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x],
    )

with axis_col2:

    remaining_features = [
        feature for feature in selected_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축 속성",
        options=remaining_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x],
    )


# ==========================================
# 12. 나머지 속성의 고정값
# ==========================================

# 테스트 데이터 중앙값을 사용
fixed_values = X_test.median()

fixed_features = [
    feature for feature in selected_features
    if feature not in [x_feature, y_feature]
]

st.subheader("그림을 그리기 위한 나머지 속성의 고정값")

if fixed_features:

    fixed_table = pd.DataFrame({
        "속성": [
            FEATURE_NAMES[feature]
            for feature in fixed_features
        ],
        "고정한 값": [
            round(float(fixed_values[feature]), 3)
            for feature in fixed_features
        ],
        "기준": [
            "테스트 데이터의 중앙값"
            for feature in fixed_features
        ]
    })

    st.dataframe(
        fixed_table,
        use_container_width=True,
        hide_index=True
    )

else:

    st.write(
        "선택한 입력 속성이 두 개뿐이므로 "
        "고정할 나머지 속성이 없습니다."
    )


# ==========================================
# 13. 2차원 격자 생성
# ==========================================

plot_df = X_test.copy()

plot_df["실제 결과"] = y_test.map({
    0: "뇌졸중 아님",
    1: "뇌졸중"
})

x_min = float(plot_df[x_feature].min())
x_max = float(plot_df[x_feature].max())

y_min = float(plot_df[y_feature].min())
y_max = float(plot_df[y_feature].max())

# 모든 값이 같을 때도 그래프를 그릴 수 있도록 범위 확장
if x_min == x_max:
    x_min -= 1
    x_max += 1

if y_min == y_max:
    y_min -= 1
    y_max += 1

resolution = 80

x_values = [
    x_min + (x_max - x_min) * i / (resolution - 1)
    for i in range(resolution)
]

y_values = [
    y_min + (y_max - y_min) * i / (resolution - 1)
    for i in range(resolution)
]

# 두 축 외의 속성은 테스트 데이터 중앙값으로 고정
base_row = fixed_values.to_dict()

grid_rows = []

for y_value in y_values:

    for x_value in x_values:

        row = base_row.copy()

        row[x_feature] = x_value
        row[y_feature] = y_value

        grid_rows.append(row)

grid_raw = pd.DataFrame(
    grid_rows,
    columns=selected_features
)

# 학습할 때 사용한 scaler 적용
grid_scaled = scaler.transform(grid_raw)


# ==========================================
# 14. 로지스틱 회귀의 확률 경계 계산
# ==========================================

positive_index = list(
    logistic_model.classes_
).index(1)

logistic_probabilities = logistic_model.predict_proba(
    grid_scaled
)[:, positive_index]

logistic_matrix = [
    logistic_probabilities[
        i * resolution:(i + 1) * resolution
    ].tolist()
    for i in range(resolution)
]

prob_min = float(min(logistic_probabilities))
prob_max = float(max(logistic_probabilities))

boundary_visible = (
    prob_min <= 0.5 <= prob_max
)


# ==========================================
# 15. 의사결정트리의 영역 계산
# ==========================================

tree_grid_predictions = tree_model.predict(
    grid_scaled
)

tree_matrix = [
    tree_grid_predictions[
        i * resolution:(i + 1) * resolution
    ].tolist()
    for i in range(resolution)
]


# ==========================================
# 16. 산점도와 두 모델의 판단 영역
# ==========================================

fig = go.Figure()

# 의사결정트리의 판단 영역
fig.add_trace(
    go.Heatmap(
        x=x_values,
        y=y_values,
        z=tree_matrix,
        zmin=0,
        zmax=1,
        colorscale=[
            [0, "rgba(100, 160, 220, 0.35)"],
            [1, "rgba(240, 120, 120, 0.35)"]
        ],
        opacity=0.35,
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 영역",
    )
)

# 로지스틱 회귀의 확률 0.5 경계
if boundary_visible:

    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=logistic_matrix,
            contours={
                "start": 0.5,
                "end": 0.5,
                "size": 1,
                "coloring": "none",
                "showlabels": False,
            },
            line={
                "color": "black",
                "width": 3,
                "dash": "dash",
            },
            showscale=False,
            hoverinfo="skip",
            name="로지스틱 회귀 확률 0.5 경계",
        )
    )


# 실제 테스트 데이터 점
fig.add_trace(
    go.Scatter(
        x=plot_df[x_feature],
        y=plot_df[y_feature],
        mode="markers",
        name="테스트 데이터",
        marker={
            "size": 8,
            "opacity": 0.75,
            "color": plot_df["실제 결과"].map({
                "뇌졸중 아님": "#2878B5",
                "뇌졸중": "#D62728"
            }),
            "line": {
                "width": 0.5,
                "color": "white"
            }
        },
        text=plot_df["실제 결과"],
        hovertemplate=(
            "가로축: %{x}<br>"
            "세로축: %{y}<br>"
            "실제 결과: %{text}"
            "<extra></extra>"
        ),
    )
)

# 실제 결과 범례를 명확하게 표시
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        name="실제 뇌졸중 아님",
        marker={"color": "#2878B5", "size": 9},
    )
)

fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        name="실제 뇌졸중",
        marker={"color": "#D62728", "size": 9},
    )
)

fig.update_layout(
    title="테스트 데이터와 모델의 판단 영역",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    height=650,
    legend_title="범례",
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# 17. 확률 경계 안내
# ==========================================

st.subheader("로지스틱 회귀의 확률 0.5 경계")

st.write(
    f"그림에서 계산한 뇌졸중 확률의 범위는 "
    f"약 {prob_min:.3f} ~ {prob_max:.3f}입니다."
)

if boundary_visible:

    st.success(
        "확률 0.5 경계가 현재 그림의 범위 안에 있습니다. "
        "검은 점선이 경계입니다."
    )

else:

    st.warning(
        "확률 0.5 경계가 현재 그림 범위 안에 나타나지 않습니다. "
        "따라서 이 그림에서는 경계선을 확인할 수 없습니다."
    )

if fixed_features:

    fixed_text = ", ".join(
        f"{FEATURE_NAMES[feature]} = "
        f"{fixed_values[feature]:.3f}"
        for feature in fixed_features
    )

    st.caption(
        "그림을 계산할 때 고정한 나머지 속성: "
        + fixed_text
    )


# ==========================================
# 18. 의사결정트리의 질문 그림
# ==========================================

st.divider()
st.header("⑨ 의사결정트리가 던지는 질문")

st.write("""
의사결정트리는 질문을 최대 세 번까지 던집니다.

각 마디에는 그곳에 도달한 훈련 데이터의 사람 수,
실제 뇌졸중 환자 수와 그 비율을 표시합니다.

마지막 마디는 모델이 내리는 답입니다.
""")

tree_structure = tree_model.tree_

left_children = tree_structure.children_left
right_children = tree_structure.children_right
tree_features = tree_structure.feature
tree_thresholds = tree_structure.threshold

# 훈련 데이터가 각 마디에 도달했는지 계산
decision_path = tree_model.decision_path(
    X_train_scaled
)

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white", nodesep=0.35, ranksep=0.55];',
    'node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=10, margin=0.15];',
    'edge [fontname="sans-serif", fontsize=10];',
]

leaf_count = 0
negative_leaf_count = 0
used_tree_features = set()

for node_id in range(tree_structure.node_count):

    # 해당 마디에 도달한 훈련 데이터
    node_mask = (
        decision_path[:, node_id]
        .toarray()
        .ravel()
        .astype(bool)
    )

    node_people = int(node_mask.sum())

    node_positive = int(
        y_train[node_mask].sum()
    )

    node_rate = (
        node_positive / node_people * 100
        if node_people > 0
        else 0
    )

    is_leaf = (
        left_children[node_id] == right_children[node_id]
    )

    if is_leaf:

        leaf_count += 1

        # 이 마디에서 모델이 내리는 답
        prediction = int(
            tree_model.predict(
                X_train_scaled[node_mask][:1]
            )[0]
        )

        if prediction == 0:
            negative_leaf_count += 1
            answer = "뇌졸중 아님"
            fill_color = "#D8EAF8"
        else:
            answer = "뇌졸중"
            fill_color = "#F8DCDC"

        label = (
            f"답: {answer}\\n"
            f"훈련 인원: {node_people}명\\n"
            f"실제 뇌졸중: {node_positive}명\\n"
            f"실제 비율: {node_rate:.1f}%"
        )

    else:

        feature_index = tree_features[node_id]

        feature = selected_features[feature_index]

        used_tree_features.add(feature)

        # StandardScaler 기준의 경계를 원래 단위로 환산
        original_threshold = (
            tree_thresholds[node_id]
            * scaler.scale_[feature_index]
            + scaler.mean_[feature_index]
        )

        label = (
            f"{FEATURE_NAMES[feature]} ≤ "
            f"{original_threshold:.2f} ?\\n"
            f"훈련 인원: {node_people}명\\n"
            f"실제 뇌졸중: {node_positive}명\\n"
            f"실제 비율: {node_rate:.1f}%"
        )

        fill_color = "#FFF2CC"

    # DOT 문자열에서 따옴표 등 특수문자 처리
    label = label.replace('"', '\\"')

    dot_lines.append(
        f'{node_id} [label="{label}", fillcolor="{fill_color}"];'
    )

    if not is_leaf:

        left_id = left_children[node_id]
        right_id = right_children[node_id]

        # 왼쪽: 질문에 예 (조건 만족)
        dot_lines.append(
            f'{node_id} -> {left_id} [label="예"];'
        )

        # 오른쪽: 질문에 아니요 (조건 불만족)
        dot_lines.append(
            f'{node_id} -> {right_id} [label="아니요"];'
        )


dot_lines.append("}")

dot_string = "\n".join(dot_lines)

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# ==========================================
# 19. 의사결정트리 요약
# ==========================================

st.subheader("의사결정트리 요약")

st.write(
    f"- 답을 내리는 마지막 마디: **{leaf_count}칸**"
)

st.write(
    f"- 그중 '뇌졸중 아님'이라고 답하는 마디: "
    f"**{negative_leaf_count}칸**"
)

if used_tree_features:

    used_names = [
        FEATURE_NAMES[feature]
        for feature in selected_features
        if feature in used_tree_features
    ]

    st.write(
        "- 실제로 질문에 사용한 속성: **"
        + ", ".join(used_names)
        + "**"
    )

else:

    st.write(
        "- 실제로 질문에 사용한 속성: **없음** "
        "(첫 마디에서 바로 답을 냈습니다.)"
    )


st.divider()

st.caption("""
이 페이지는 학습용 분류 실습입니다.
모델의 예측 결과는 의료 진단이나 실제 치료 판단에 사용할 수 없습니다.
""")

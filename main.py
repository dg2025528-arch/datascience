import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="태양광 설치 군집 분석", page_icon="☀️", layout="wide")
st.title("☀️ 지역별 전력 사용량 군집 분석")
st.caption("전력 사용 패턴이 비슷한 지역을 묶어 태양광 설치 우선순위를 해석합니다.")


@st.cache_data
def load_csv(file):
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSV를 읽을 수 없습니다. 인코딩을 확인하세요.")


uploaded = st.sidebar.file_uploader("서울 열린데이터광장 CSV 업로드", type="csv")
k = st.sidebar.slider("군집 수 (k)", 2, 8, 3)

if uploaded is None:
    st.info("왼쪽에서 CSV 파일을 업로드하세요.")
    st.stop()

df = load_csv(uploaded)
st.subheader("1. 원본 데이터")
st.dataframe(df.head(20), use_container_width=True)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
all_cols = df.columns.tolist()

st.sidebar.markdown("### 컬럼 선택")
region_col = st.sidebar.selectbox("지역 컬럼", all_cols)
feature_cols = st.sidebar.multiselect(
    "군집에 쓸 숫자 컬럼", num_cols, default=num_cols[:3]
)

if len(feature_cols) < 2:
    st.warning("숫자 컬럼을 2개 이상 선택하세요.")
    st.stop()

# 지역별 평균으로 집계
data = (
    df.groupby(region_col)[feature_cols].mean().dropna().reset_index()
)
if len(data) <= k:
    st.warning("지역 수가 군집 수보다 많아야 합니다. k를 줄이세요.")
    st.stop()

X = StandardScaler().fit_transform(data[feature_cols])
model = KMeans(n_clusters=k, n_init=10, random_state=42)
data["군집"] = model.fit_predict(X).astype(str)

st.subheader("2. 군집 결과")
score = silhouette_score(X, model.labels_)
st.metric("실루엣 점수 (1에 가까울수록 좋음)", f"{score:.3f}")

# 2차원 시각화 (PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit_transform(X)
data["PC1"], data["PC2"] = pca[:, 0], pca[:, 1]
fig = px.scatter(
    data, x="PC1", y="PC2", color="군집",
    hover_name=region_col, hover_data=feature_cols,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("3. 군집별 평균")
summary = data.groupby("군집")[feature_cols].mean().round(2)
summary["지역 수"] = data.groupby("군집").size()
st.dataframe(summary, use_container_width=True)

st.subheader("4. 설치 우선순위 해석")
first = feature_cols[0]
order = summary[first].sort_values(ascending=False).index.tolist()
st.write(f"**`{first}` 평균이 높은 순서**로 우선순위를 매겼습니다.")
for rank, c in enumerate(order, 1):
    regions = ", ".join(data.loc[data["군집"] == c, region_col].astype(str))
    st.markdown(f"**{rank}순위 · 군집 {c}**: {regions}")

st.download_button(
    "군집 결과 CSV 다운로드",
    data.to_csv(index=False).encode("utf-8-sig"),
    "cluster_result.csv",
    "text/csv",
)

from pathlib import Path
from io import BytesIO
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="2025 서울 전력 사용량 분석",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ 2025년 서울 25개 구 전력 사용량 분석")
st.caption(
    "연간 사용 규모 · 월별 집중도 · 비슷한 사용 패턴을 가진 지역을 살펴봅니다."
)

BASE_DIR = Path(__file__).resolve().parent
MONTHS = [f"{i}월" for i in range(1, 13)]

SEOUL_GU = {
    "강남구", "강동구", "강북구", "강서구", "관악구",
    "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구",
    "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구",
}


# =========================================================
# 파일 탐색 및 읽기
# =========================================================
def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value)).lstrip("\ufeff")


def to_number(series):
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def find_2025_files():
    result = []

    for folder in [BASE_DIR, BASE_DIR / "data"]:
        if not folder.is_dir():
            continue

        for path in folder.iterdir():
            if not path.is_file():
                continue

            if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
                continue

            if path.name.startswith("~$"):
                continue

            name = path.stem.lower()
            is_power = (
                "energyuse" in name
                or "전력사용량" in name
            )
            is_2025 = bool(
                re.search(r"2025|(?<!\d)25(?!\d)", name)
            )
            is_2026 = bool(
                re.search(r"2026|26\.8", name)
            )

            if is_power and is_2025 and not is_2026:
                result.append(path)

    return sorted(
        result,
        key=lambda p: (
            0 if "energyuse" in p.stem.lower() else 1,
            str(p),
        ),
    )


@st.cache_data
def read_raw(content, suffix):
    if suffix != ".csv":
        # 제공된 구조 기준: 첫 번째 시트 사용
        return pd.read_excel(BytesIO(content), header=None)

    for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(
                BytesIO(content),
                header=None,
                encoding=encoding,
            )
        except UnicodeError:
            continue

    raise ValueError("CSV 인코딩을 읽지 못했습니다.")


def find_header(raw):
    for row in range(min(20, len(raw))):
        values = {normalize(v) for v in raw.iloc[row]}

        if "1월" in values and (
            "자치구명" in values or "시군구" in values
        ):
            return row

    raise ValueError(
        "제목 행을 찾지 못했습니다. "
        "자치구명 또는 시군구, 1월~12월 열이 있는지 확인하세요."
    )


def build_table(raw, header_row):
    names = []
    used = set()

    for i, value in enumerate(raw.iloc[header_row]):
        base = normalize(value) or f"빈열{i + 1}"
        name = base
        duplicate = 2

        while name in used:
            name = f"{base}_{duplicate}"
            duplicate += 1

        names.append(name)
        used.add(name)

    df = raw.iloc[header_row + 1:].copy()
    df.columns = names

    return (
        df.dropna(how="all")
        .dropna(axis=1, how="all")
        .reset_index(drop=True)
    )


def prepare_data(path):
    raw = read_raw(path.read_bytes(), path.suffix.lower())
    header_row = find_header(raw)
    df = build_table(raw, header_row)

    logs = []

    def record(stage, before, after, note=""):
        logs.append({
            "처리 단계": stage,
            "처리 전 행 수": before,
            "처리 후 행 수": after,
            "설명": note,
        })

    record(
        "제목 행 적용",
        len(df),
        len(df),
        f"첫 줄을 0으로 센 제목 행 번호: {header_row}",
    )

    is_kepco = {
        "연도", "시도", "시군구", "업종별"
    }.issubset(df.columns)

    if is_kepco:
        source = "한전 시군구별 용도업종별 전력사용량"
        region_col = "시군구"

        before = len(df)
        df = df[to_number(df["연도"]).eq(2025)].copy()
        record("2025년만 선택", before, len(df))

        before = len(df)
        df = df[
            df["시도"].map(normalize).eq("서울특별시")
        ].copy()
        record(
            "서울특별시만 선택",
            before,
            len(df),
            "다른 시도의 같은 이름 구 제외",
        )

        before = len(df)

        # 구별 전체 사용량은 이미 합계 행에 들어 있음.
        # 합계와 세부 업종을 함께 더하지 않음.
        df = df[
            df["업종별"].map(normalize).eq("합계")
        ].copy()

        record(
            "업종별 합계 행만 사용",
            before,
            len(df),
            "세부 업종 행 제외하여 이중 집계 방지",
        )

        conversion = 1 / 1000

    else:
        if "자치구명" not in df.columns:
            raise ValueError("자치구명 열을 찾지 못했습니다.")

        source = "서울시 에너지정보"
        region_col = "자치구명"

        before = len(df)
        df = df[
            ~df[region_col].map(normalize).isin(
                ["총합", "합계", "총계", "소계", "계"]
            )
        ].copy()

        record("총합 행 제거", before, len(df))
        conversion = 1

    missing_columns = [m for m in MONTHS if m not in df.columns]
    if missing_columns:
        raise ValueError(
            "월별 열이 없습니다: " + ", ".join(missing_columns)
        )

    df[region_col] = df[region_col].map(normalize)

    before = len(df)
    df = df[df[region_col].isin(SEOUL_GU)].copy()
    record("서울 25개 구 이름 확인", before, len(df))

    if df.empty:
        raise ValueError("전처리 후 서울 구 데이터가 없습니다.")

    if df[region_col].duplicated().any():
        duplicate_gu = df.loc[
            df[region_col].duplicated(keep=False), region_col
        ].tolist()

        raise ValueError(
            f"구별 행이 중복되어 자동 집계하지 않았습니다: {duplicate_gu}"
        )

    matrix = df.set_index(region_col)[MONTHS].copy()
    matrix.index.name = "자치구"

    for month in MONTHS:
        matrix[month] = to_number(matrix[month]) * conversion

    record(
        "숫자 변환 및 단위 통일",
        len(matrix),
        len(matrix),
        "kWh ÷ 1000 → MWh" if is_kepco else "원본 MWh 유지",
    )

    # 일부 월이 없는데 연간 합계로 해석하는 것을 방지
    before = len(matrix)
    matrix = matrix.dropna(how="any")

    record(
        "12개월 완전한 구만 사용",
        before,
        len(matrix),
        "결측값을 0으로 채우지 않음",
    )

    if (matrix < 0).any().any():
        raise ValueError(
            "구별 합계에 음수가 있습니다. 원본 확인이 필요합니다."
        )

    before = len(matrix)
    matrix = matrix.loc[matrix.sum(axis=1) > 0]

    record(
        "연간 합계가 양수인 구 확인",
        before,
        len(matrix),
    )

    return (
        matrix.sort_index(),
        pd.DataFrame(logs),
        source,
        df,
    )


# =========================================================
# 데이터 선택 및 전처리
# =========================================================
files = find_2025_files()

if not files:
    st.error(
        "2025년 전력 사용량 파일을 찾지 못했습니다. "
        "energyUse_data 25.csv 또는 "
        "시군구별전력사용량 2025.csv를 main.py 옆이나 data 폴더에 올려 주세요."
    )
    st.stop()

st.sidebar.header("📂 데이터 설정")

path = st.sidebar.selectbox(
    "2025년 자료 선택",
    files,
    format_func=lambda p: str(p.relative_to(BASE_DIR)),
)

st.sidebar.info(
    "2026년 및 태양광 파일은 이번 분석에서 제외합니다. "
    "선택한 한 출처로 세 그래프를 모두 계산합니다."
)

try:
    matrix, logs, source, original = prepare_data(path)
except Exception as exc:
    st.error(f"데이터 처리 오류: {exc}")
    st.stop()

if matrix.empty:
    st.error("12개월 값이 모두 있는 구가 없습니다.")
    st.stop()

annual = matrix.sum(axis=1)

ranking = (
    annual.rename("연간 사용량(MWh)")
    .sort_values(ascending=False)
    .rename_axis("자치구")
    .reset_index()
)

shares = matrix.div(annual, axis=0) * 100

st.write(
    f"**분석 연도:** 2025년 · **출처:** {source} · "
    "**단위:** MWh · **기간:** 1~12월"
)

col1, col2, col3 = st.columns(3)
col1.metric("분석 지역", f"{len(matrix)}개 구")
col2.metric("최대 사용 지역", ranking.iloc[0]["자치구"])
col3.metric("최소 사용 지역", ranking.iloc[-1]["자치구"])

missing_gu = sorted(SEOUL_GU - set(matrix.index))
if missing_gu:
    st.warning(
        "분석에서 빠진 서울 구: " + ", ".join(missing_gu)
    )

with st.expander("📋 원본 표·전처리 log 확인"):
    st.markdown("#### 제목을 적용한 원본 표")
    st.dataframe(
        original.fillna("").astype(str),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 전처리 log")
    st.dataframe(logs, use_container_width=True, hide_index=True)

    st.markdown("#### 분석에 사용하는 월별 값(MWh)")
    st.dataframe(matrix, use_container_width=True)

    st.download_button(
        "전처리 log 다운로드",
        to_csv_bytes(logs),
        "preprocess_2025.csv",
        "text/csv",
    )

tab1, tab2, tab3 = st.tabs([
    "📊 연간 사용량 막대",
    "🔥 구 × 월 히트맵",
    "🧩 군집 산점도(PCA)",
])


# =========================================================
# 1. 연간 사용량 정렬 막대
# =========================================================
with tab1:
    st.subheader("2025년 구별 연간 전력 사용량")
    st.caption(
        "분석 질문: 어느 구가 많이 쓰며, 사용량 상·하위 구는 어디인가?"
    )

    fig = px.bar(
        ranking,
        x="자치구",
        y="연간 사용량(MWh)",
        color="연간 사용량(MWh)",
        color_continuous_scale="YlOrRd",
    )

    fig.update_layout(
        height=560,
        xaxis={
            "categoryorder": "array",
            "categoryarray": ranking["자치구"].tolist(),
        },
        yaxis_title="연간 전력 사용량 (MWh)",
    )

    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown("#### 🔺 사용량 상위 3개 구")
        st.dataframe(
            ranking.head(3),
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.markdown("#### 🔻 사용량 하위 3개 구")
        st.dataframe(
            ranking.tail(3).sort_values("연간 사용량(MWh)"),
            hide_index=True,
            use_container_width=True,
        )

    top = ranking.iloc[0]
    bottom = ranking.iloc[-1]
    ratio = top["연간 사용량(MWh)"] / bottom["연간 사용량(MWh)"]

    st.success(
        f"발견: {top['자치구']}의 연간 사용량은 "
        f"{top['연간 사용량(MWh)']:,.0f} MWh로, "
        f"{bottom['자치구']}의 약 {ratio:.2f}배입니다."
    )

    st.caption(
        "해석의 한계: 총사용량에는 인구·건물 규모·업종 구성 등의 영향이 "
        "포함됩니다. 사용량이 많다는 이유만으로 에너지 효율이 낮다고 "
        "판단할 수 없습니다."
    )


# =========================================================
# 2. 구 × 월 히트맵
# =========================================================
with tab2:
    st.subheader("구별 월별 사용 패턴 비교")
    st.caption(
        "분석 질문: 총사용량 규모를 제외해도 특정 월에 집중되는 구가 있는가?"
    )

    heat_mode = st.radio(
        "히트맵 표시 기준",
        ["연간 대비 월별 비율(%)", "원본 사용량(MWh)"],
        horizontal=True,
    )

    if heat_mode.startswith("연간"):
        heat = shares.copy()
        color_label = "연간 대비 비율 (%)"
    else:
        heat = matrix.copy()
        color_label = "사용량 (MWh)"

    # 막대그래프와 같은 순서로 표시
    heat = heat.reindex(ranking["자치구"].tolist())

    fig = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels={
            "x": "월",
            "y": "자치구",
            "color": color_label,
        },
    )
    fig.update_layout(height=max(550, len(heat) * 25))

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "비율 모드는 각 구의 연간 사용량을 100%로 봅니다. "
        "12개월 사용량이 같다면 한 달의 비중은 약 8.33%입니다."
    )

    peak_table = pd.DataFrame({
        "최대 사용 월": shares.idxmax(axis=1),
        "최대 월 비중(%)": shares.max(axis=1),
    }).sort_values("최대 월 비중(%)", ascending=False)

    st.markdown("#### 특정 한 달에 상대적으로 집중된 구")
    st.dataframe(
        peak_table.head(5).round(2),
        use_container_width=True,
    )

    counts = peak_table["최대 사용 월"].value_counts()
    common_peak = counts.index[0]
    common_count = int(counts.iloc[0])

    st.success(
        f"발견: {len(matrix)}개 구 중 {common_count}개 구의 "
        f"최대 사용 월은 {common_peak}입니다."
    )

    most_concentrated = peak_table.iloc[0]
    st.write(
        f"**최대 월 비중이 가장 높은 구:** {peak_table.index[0]} "
        f"— {most_concentrated['최대 사용 월']}에 "
        f"연간 사용량의 {most_concentrated['최대 월 비중(%)']:.2f}% 사용"
    )

    st.caption(
        "월별 증가의 원인은 이 그래프만으로 확정할 수 없습니다. "
        "냉난방 수요 등을 설명하려면 기온·업종 자료로 추가 확인해야 합니다. "
        "월마다 날짜 수가 다른 영향도 포함됩니다."
    )


# =========================================================
# 3. 군집 산점도
# =========================================================
with tab3:
    st.subheader("월별 사용 패턴이 비슷한 구 묶기")
    st.caption(
        "연간 총량이 아니라 월별 비율로 군집을 만들어 "
        "큰 구와 작은 구의 패턴을 비교합니다."
    )

    features = shares / 100
    varying_cols = features.columns[
        features.nunique() > 1
    ].tolist()

    if len(features) < 3 or len(varying_cols) < 2:
        st.warning(
            "군집 분석에는 구 3개 이상과 값이 달라지는 월 2개 이상이 필요합니다."
        )
    else:
        X = StandardScaler().fit_transform(features[varying_cols])

        unique_count = len(np.unique(X, axis=0))
        max_k = min(8, len(features) - 1, unique_count)

        if max_k < 2:
            st.warning("서로 다른 사용 패턴이 충분하지 않습니다.")
        else:
            k = int(st.number_input(
                "군집 수",
                min_value=2,
                max_value=max_k,
                value=min(3, max_k),
            ))

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=20,
            )
            labels = model.fit_predict(X)

            pca = PCA(n_components=2)
            coordinates = pca.fit_transform(X)

            result = pd.DataFrame({
                "자치구": features.index,
                "군집": (labels + 1).astype(str),
                "PC1": coordinates[:, 0],
                "PC2": coordinates[:, 1],
                "연간 사용량(MWh)": annual.reindex(
                    features.index
                ).to_numpy(),
            })

            fig = px.scatter(
                result,
                x="PC1",
                y="PC2",
                color="군집",
                text="자치구",
                hover_name="자치구",
                hover_data=["연간 사용량(MWh)"],
                category_orders={
                    "군집": [str(i) for i in range(1, k + 1)]
                },
            )

            fig.update_traces(textposition="top center")
            fig.update_layout(height=650)

            st.plotly_chart(fig, use_container_width=True)

            explained = pca.explained_variance_ratio_ * 100
            st.caption(
                f"PC1 설명 분산: {explained[0]:.1f}% / "
                f"PC2: {explained[1]:.1f}% / "
                f"합계: {explained.sum():.1f}%"
            )
            st.caption(
                "군집은 전체 월별 비율을 표준화한 특성으로 계산하고, "
                "PCA는 결과를 2차원으로 보여주는 용도로만 사용합니다. "
                "군집 번호는 순위가 아닙니다."
            )

            actual_k = len(np.unique(labels))
            if 1 < actual_k < len(features):
                score = silhouette_score(X, labels)
                st.metric("실루엣 점수", f"{score:.3f}")
                st.caption(
                    "높을수록 선택한 특성에서 군집 구분이 뚜렷합니다. "
                    "태양광 설치 적합성을 나타내는 점수는 아닙니다."
                )

            profiles = shares.copy()
            profiles["군집"] = (labels + 1).astype(str)
            cluster_means = profiles.groupby("군집")[MONTHS].mean()

            st.markdown("#### 군집별 구성 지역")

            for cluster_id in sorted(
                result["군집"].unique(), key=int
            ):
                regions = result.loc[
                    result["군집"] == cluster_id, "자치구"
                ].tolist()

                peak_month = cluster_means.loc[cluster_id].idxmax()

                st.write(
                    f"**군집 {cluster_id}:** "
                    + ", ".join(regions)
                )
                st.caption(
                    f"군집 평균 월별 비중이 가장 높은 달: {peak_month}"
                )

            with st.expander("군집별 월평균 비중(%) 확인"):
                st.dataframe(
                    cluster_means.round(2),
                    use_container_width=True,
                )

            # 같은 군집 안에서 규모 차이가 큰 사례 찾기
            examples = []

            for cluster_id, group in result.groupby("군집"):
                if len(group) < 2:
                    continue

                ordered = group.sort_values("연간 사용량(MWh)")
                low = ordered.iloc[0]
                high = ordered.iloc[-1]

                scale_ratio = (
                    high["연간 사용량(MWh)"]
                    / low["연간 사용량(MWh)"]
                )

                examples.append({
                    "군집": cluster_id,
                    "사용량 큰 구": high["자치구"],
                    "사용량 작은 구": low["자치구"],
                    "사용량 배수": scale_ratio,
                })

            if examples:
                example = max(
                    examples, key=lambda item: item["사용량 배수"]
                )

                st.success(
                    f"살펴볼 사례: {example['사용량 큰 구']}와 "
                    f"{example['사용량 작은 구']}는 연간 사용량이 "
                    f"약 {example['사용량 배수']:.2f}배 차이 나지만 "
                    f"군집 {example['군집']}에 함께 묶였습니다."
                )

                st.write(
                    "**해석:** 총사용량의 크기와 월별 사용 패턴은 "
                    "서로 다른 특성입니다. 같은 군집에 속한 구가 "
                    "반드시 비슷한 양의 전기를 사용하는 것은 아닙니다."
                )

            st.download_button(
                "군집 결과 CSV 다운로드",
                to_csv_bytes(result),
                "cluster_2025.csv",
                "text/csv",
            )


# =========================================================
# 결과 다운로드 및 해석 안내
# =========================================================
st.divider()

st.download_button(
    "분석에 사용한 2025년 월별 데이터 다운로드",
    to_csv_bytes(matrix.reset_index()),
    "seoul_2025_monthly_MWh.csv",
    "text/csv",
)

st.subheader("📝 발표용 인사이트 정리")
st.markdown("""
- **막대그래프:** 어느 구의 사용량이 크며, 상·하위 구의 차이는 얼마인가?
- **히트맵:** 어느 달에 사용이 집중되며, 그 집중도가 큰 구는 어디인가?
- **군집 산점도:** 사용 규모는 다른데 비슷한 패턴으로 묶인 구는 어디인가?
""")

st.warning(
    "이 분석은 전력 소비의 규모와 월별 패턴을 보여줍니다. "
    "태양광 설치 최적 입지를 확정하려면 일사량, 설치 가능 면적, "
    "시간대별 수요 등의 추가 자료가 필요합니다."
)

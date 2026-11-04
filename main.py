from pathlib import Path
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
    page_title="서울 지역별 전력 사용량 분석",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ 서울 지역별 전력 사용량 분석")
st.caption("저장소에 등록한 데이터로 사용 규모와 월별 패턴을 비교합니다.")

BASE_DIR = Path(__file__).resolve().parent
SUPPORTED = {".xlsx", ".xls", ".csv"}

SEOUL_GU = {
    "강남구", "강동구", "강북구", "강서구", "관악구",
    "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구",
    "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구",
}


# =========================================================
# 파일 처리
# =========================================================
def find_files():
    """저장소 루트 및 data 폴더의 데이터 파일 탐색."""
    files = []

    for folder in (BASE_DIR, BASE_DIR / "data"):
        if not folder.is_dir():
            continue

        for path in folder.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED
                and not path.name.startswith("~$")
            ):
                files.append(path)

    return sorted(set(files), key=lambda p: str(p.relative_to(BASE_DIR)))


@st.cache_data
def get_sheets(content, suffix):
    import io

    if suffix == ".csv":
        return ["CSV"]

    with pd.ExcelFile(io.BytesIO(content)) as book:
        return book.sheet_names


@st.cache_data
def read_raw(content, suffix, sheet):
    import io

    if suffix != ".csv":
        return pd.read_excel(
            io.BytesIO(content),
            sheet_name=sheet,
            header=None,
        )

    last_error = None

    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(
                io.BytesIO(content),
                encoding=encoding,
                header=None,
            )
        except (UnicodeError, pd.errors.ParserError) as exc:
            last_error = exc

    raise ValueError(f"CSV를 읽지 못했습니다: {last_error}")


def make_table(raw, start, count):
    if start + count > len(raw):
        raise ValueError("제목 행 범위가 데이터 길이를 초과합니다.")

    header = raw.iloc[start:start + count].copy()

    # 여러 줄 제목에서 병합된 상위 제목 보완
    if count > 1:
        header = header.ffill(axis=1)

    columns = []
    used = set()

    for i in range(raw.shape[1]):
        parts = []

        for value in header.iloc[:, i]:
            if pd.notna(value):
                text = str(value).strip()
                if text and text not in parts:
                    parts.append(text)

        base = " / ".join(parts) if parts else f"빈 제목 {i + 1}"
        name = base
        n = 2

        while name in used:
            name = f"{base} ({n})"
            n += 1

        used.add(name)
        columns.append(name)

    df = raw.iloc[start + count:].copy()
    df.columns = columns

    return (
        df.dropna(how="all")
        .dropna(axis=1, how="all")
        .reset_index(drop=True)
    )


def as_number(series):
    text = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({
            "": pd.NA,
            "-": pd.NA,
            "--": pd.NA,
            "…": pd.NA,
        })
    )

    return pd.to_numeric(text, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def normalize_text(series):
    return (
        series.astype("string")
        .str.replace(r"\s+", "", regex=True)
        .str.strip()
    )


def clean_gu(series):
    return (
        normalize_text(series)
        .str.replace(r"^서울특별시", "", regex=True)
        .str.replace(r"^서울시", "", regex=True)
    )


def guess_column(columns, words):
    for word in words:
        for i, column in enumerate(columns):
            if word in str(column):
                return i
    return 0


def month_number(name):
    match = re.search(r"(?<!\d)(1[0-2]|0?[1-9])\s*월", str(name))
    return int(match.group(1)) if match else 99


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def show_table(df):
    # 혼합형 셀을 안전하게 표시
    st.dataframe(
        df.fillna("").astype(str),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 데이터 선택
# =========================================================
files = find_files()

if not files:
    st.error(
        "데이터 파일이 없습니다. GitHub에서 main.py와 같은 위치 또는 "
        "data 폴더에 엑셀·CSV 파일을 올려 주세요."
    )
    st.stop()

st.sidebar.header("📂 데이터 선택")

path = st.sidebar.selectbox(
    "파일",
    files,
    format_func=lambda p: str(p.relative_to(BASE_DIR)),
)

file_key = str(path.relative_to(BASE_DIR))

try:
    content = path.read_bytes()
    suffix = path.suffix.lower()
    sheets = get_sheets(content, suffix)

    sheet = st.sidebar.selectbox(
        "시트",
        sheets,
        key=f"sheet_{file_key}",
    )

    raw = read_raw(content, suffix, sheet)

except Exception as exc:
    st.error(f"파일 읽기 오류: {exc}")
    st.stop()

if raw.empty:
    st.warning("선택한 시트에 데이터가 없습니다.")
    st.stop()

key = f"{file_key}_{sheet}"

with st.expander("🔍 원본 처음 20줄 — 제목 행 확인", expanded=False):
    st.dataframe(
        raw.head(20).fillna("").astype(str),
        use_container_width=True,
    )
    st.caption(
        "월 이름이나 자치구 등 실제 제목이 있는 줄의 맨 왼쪽 번호를 "
        "사이드바에 입력하세요. 엑셀의 서식·색상은 재현하지 않습니다."
    )

header_start = st.sidebar.number_input(
    "제목 시작 행 번호 (0=첫 줄)",
    min_value=0,
    max_value=len(raw) - 1,
    value=0,
    key=f"header_{key}",
)

header_count = st.sidebar.selectbox(
    "제목이 차지하는 행 수",
    [1, 2, 3],
    key=f"header_count_{key}",
)

try:
    df = make_table(raw, int(header_start), int(header_count))
except Exception as exc:
    st.error(str(exc))
    st.stop()

if df.empty or not len(df.columns):
    st.warning("제목 행 설정을 확인하세요. 읽을 데이터가 없습니다.")
    st.stop()

st.subheader("1. 선택한 데이터")
st.write(f"**파일:** {path.name} · **시트:** {sheet}")
show_table(df.head(100))
st.caption(f"전체 {len(df):,}행 · {len(df.columns)}열 / 위 표는 최대 100행 표시")

st.info(
    "전력 사용량 파일에서 지역 열과 월별 사용량 열을 선택하세요. "
    "태양광 사업 정보처럼 월별 열이 없는 파일은 원본 표를 확인할 수 있지만, "
    "아래 월별 패턴 분석에는 별도 가공이 필요합니다."
)


# =========================================================
# 컬럼 설정
# =========================================================
columns = df.columns.tolist()

st.sidebar.header("⚙️ 분석 설정")

region_col = st.sidebar.selectbox(
    "지역 열",
    columns,
    index=guess_column(
        columns, ["자치구", "시군구", "구군", "시군", "지역"]
    ),
    key=f"region_{key}",
)

sido_options = ["선택 안 함"] + [
    c for c in columns if c != region_col
]

sido_default = next(
    (
        i for i, col in enumerate(sido_options)
        if col != "선택 안 함"
        and any(word in col for word in ["시도", "시·도", "시ㆍ도"])
    ),
    0,
)

sido_col = st.sidebar.selectbox(
    "시도 열 — 전국 자료라면 반드시 선택",
    sido_options,
    index=sido_default,
    key=f"sido_{key}",
)

month_options = [
    c for c in columns
    if c != region_col and c != sido_col
]

month_defaults = sorted(
    [c for c in month_options if month_number(c) != 99],
    key=month_number,
)

month_cols = st.sidebar.multiselect(
    "월별 사용량 열",
    month_options,
    default=month_defaults,
    key=f"months_{key}",
)

# 선택한 월은 가능한 경우 달 순서로 정렬
month_cols = sorted(month_cols, key=month_number)

unit = st.sidebar.text_input(
    "원본 사용량 단위",
    value="단위 확인 필요",
    key=f"unit_{key}",
)

aggregation = st.sidebar.selectbox(
    "같은 구가 여러 행일 때 집계",
    ["평균", "합계"],
    key=f"aggregation_{key}",
)

st.sidebar.caption(
    "세부 항목을 더해야 할 때만 합계를 선택하세요. "
    "총계와 세부 항목을 함께 더하면 중복 집계됩니다."
)

is_2026 = bool(re.search(r"2026|26\.8", path.stem))

zero_missing = st.sidebar.checkbox(
    "0을 미공개·결측값으로 처리",
    value=is_2026,
    key=f"zero_{key}",
)

st.sidebar.caption(
    "미공개 달의 0에만 적용하세요. 실제 사용량 0도 일괄적으로 결측이 됩니다."
)

only_seoul_gu = st.sidebar.checkbox(
    "서울 25개 구 이름만 남기기",
    value=True,
    key=f"gu_filter_{key}",
)

if not month_cols:
    st.warning("사이드바에서 월별 사용량 열을 선택하세요.")
    st.stop()

if sido_col == "선택 안 함":
    st.warning(
        "전국 자료라면 시도 열을 반드시 선택하세요. 구 이름만으로는 "
        "다른 시도의 '중구' 등을 구별할 수 없습니다."
    )


# =========================================================
# 전처리 및 기록
# =========================================================
logs = []


def record(stage, before, after, note=""):
    logs.append({
        "단계": stage,
        "처리 전 행 수": before,
        "처리 후 행 수": after,
        "제외 행 수": before - after,
        "설명": note,
    })


work = df.copy()
record("원본 표", len(work), len(work))

# 1. 합계 제거
before = len(work)
region_normalized = normalize_text(work[region_col])
total_names = {
    "합계", "총계", "소계", "계", "전체", "전국",
    "서울시", "서울특별시",
}
work = work[~region_normalized.isin(total_names)].copy()
record("합계·총괄 행 제거", before, len(work))

# 2. 서울 필터: 이름이 같은 다른 시도의 구 혼입 방지
if sido_col != "선택 안 함":
    before = len(work)
    sido = normalize_text(work[sido_col])
    work = work[sido.isin(["서울", "서울시", "서울특별시"])].copy()
    record("시도=서울 필터", before, len(work))
else:
    record(
        "시도 필터 미적용",
        len(work),
        len(work),
        "원본이 서울 전용인지 확인 필요",
    )

# 3. 지역명 정리 및 서울 구 확인
work[region_col] = clean_gu(work[region_col])

before = len(work)
work = work[
    work[region_col].notna() & work[region_col].ne("")
].copy()
record("지역명 결측 제거", before, len(work))

if only_seoul_gu:
    before = len(work)
    work = work[work[region_col].isin(SEOUL_GU)].copy()
    record("서울 25개 구 이름 확인", before, len(work))

# 4. 숫자 변환
for col in month_cols:
    work[col] = as_number(work[col])

if zero_missing:
    zeros = int(work[month_cols].eq(0).sum().sum())
    work[month_cols] = work[month_cols].replace(0, np.nan)
    record(
        "0을 결측으로 변환",
        len(work),
        len(work),
        f"{zeros:,}개 셀 변환",
    )

before = len(work)
work = work.dropna(subset=month_cols, how="all")
record("모든 선택 월이 결측인 행 제거", before, len(work))

# 5. 구별 집계
before = len(work)

if aggregation == "합계":
    data = (
        work.groupby(region_col)[month_cols]
        .sum(min_count=1)
        .reset_index()
    )
else:
    data = (
        work.groupby(region_col)[month_cols]
        .mean()
        .reset_index()
    )

record(
    "구별 집계",
    before,
    len(data),
    f"{aggregation} 적용; 이 단계의 행 감소는 구별 집계 결과",
)

st.subheader("2. 전처리 결과")
c1, c2, c3 = st.columns(3)
c1.metric("지역 수", len(data))
c2.metric("선택 월 열 수", len(month_cols))
c3.metric(
    "월별 결측 셀 수",
    int(data[month_cols].isna().sum().sum()),
)

with st.expander("전처리 log 확인"):
    log_df = pd.DataFrame(logs)
    st.dataframe(log_df, use_container_width=True, hide_index=True)

    st.download_button(
        "전처리 log 다운로드",
        csv_bytes(log_df),
        "preprocess_log.csv",
        "text/csv",
    )

if data.empty:
    st.error(
        "전처리 후 데이터가 없습니다. 제목 행, 지역 열, 시도 열을 확인하세요."
    )
    st.stop()

missing_gu = sorted(SEOUL_GU - set(data[region_col]))

if only_seoul_gu and missing_gu:
    st.caption("현재 표에 없는 서울 구: " + ", ".join(missing_gu))

matrix = data.set_index(region_col)[month_cols].astype(float)
available_months = [
    c for c in month_cols if matrix[c].notna().any()
]

if not available_months:
    st.warning("선택한 월에 유효한 숫자가 없습니다.")
    st.stop()

unpublished = [c for c in month_cols if c not in available_months]

if unpublished:
    st.warning(
        "모든 구에서 결측인 월은 그래프·군집에서 제외합니다: "
        + ", ".join(unpublished)
    )

matrix = matrix[available_months]

tab_bar, tab_heat, tab_cluster = st.tabs([
    "📊 막대그래프",
    "🔥 히트맵",
    "🧩 군집 산점도",
])


# =========================================================
# 막대그래프
# =========================================================
with tab_bar:
    st.subheader("구별 전력 사용 규모")
    st.caption("질문: 어느 구의 사용량이 크고, 지역 간 차이는 얼마나 클까?")

    bar_mode = st.radio(
        "표시 기준",
        ["선택 기간 합계", "특정 월"],
        horizontal=True,
    )

    if bar_mode == "특정 월":
        month = st.selectbox("표시할 월", available_months)
        values = matrix[month].dropna()
        period_label = month
    else:
        # 서로 다른 기간 합계를 비교하지 않도록 완전한 구만 사용
        complete = matrix.dropna(how="any")
        values = complete.sum(axis=1, min_count=1)
        period_label = f"선택 {len(available_months)}개월 합계"

        excluded = len(matrix) - len(complete)
        if excluded:
            st.warning(
                f"선택 기간에 일부 월이 결측인 {excluded}개 구는 "
                "동일 기간 비교를 위해 막대그래프에서 제외했습니다."
            )

        st.caption(
            "12개월 전체가 아니라면 연간 사용량이 아닌 "
            "'선택 기간 합계'입니다."
        )

    if values.empty:
        st.info("비교 가능한 값이 없습니다. 월 선택을 조정하세요.")
    else:
        bar = (
            values.rename("사용량")
            .rename_axis(region_col)
            .reset_index()
            .sort_values("사용량", ascending=False)
        )

        count = int(st.number_input(
            "표시할 지역 수",
            min_value=1,
            max_value=len(bar),
            value=len(bar),
        ))

        fig = px.bar(
            bar.head(count),
            x=region_col,
            y="사용량",
            color="사용량",
            color_continuous_scale="YlOrRd",
            title=period_label,
            labels={"사용량": f"사용량 ({unit})"},
        )
        st.plotly_chart(fig, use_container_width=True)

        largest = bar.iloc[0]
        st.write(
            f"**관찰:** 현재 비교에서 가장 큰 값은 "
            f"**{largest[region_col]}**, "
            f"**{largest['사용량']:,.2f} {unit}**입니다."
        )
        st.caption(
            "총사용량이 크다는 사실만으로 설치 적합성을 판단할 수는 없습니다. "
            "건물 규모·인구·산업 구성 등의 영향도 고려해야 합니다."
        )

        st.download_button(
            "막대그래프 데이터 다운로드",
            csv_bytes(bar),
            "bar_data.csv",
            "text/csv",
        )


# =========================================================
# 히트맵
# =========================================================
with tab_heat:
    st.subheader("지역 × 월 사용 패턴")
    st.caption("질문: 사용 규모를 제외해도 지역별 계절 패턴이 다를까?")

    heat_mode = st.radio(
        "히트맵 값",
        [
            "원본 사용량",
            "선택 기간 대비 월별 비율(%)",
            "구별 표준화",
        ],
    )

    hm = matrix.copy()
    color_scale = "YlOrRd"
    color_label = unit
    midpoint = None

    if heat_mode != "원본 사용량":
        before_count = len(hm)
        hm = hm.dropna(how="any")

        if len(hm) < before_count:
            st.caption(
                f"같은 기간 비교를 위해 결측이 있는 "
                f"{before_count - len(hm)}개 구를 제외했습니다."
            )

        if heat_mode == "선택 기간 대비 월별 비율(%)":
            totals = hm.sum(axis=1)
            hm = hm.loc[totals > 0]
            hm = hm.div(hm.sum(axis=1), axis=0) * 100
            color_label = "비율 (%)"
        else:
            means = hm.mean(axis=1)
            stds = hm.std(axis=1, ddof=0)
            hm = (
                hm.sub(means, axis=0)
                .div(stds.replace(0, np.nan), axis=0)
            )
            color_scale = "RdBu_r"
            color_label = "표준점수"
            midpoint = 0

            st.caption(
                "양수는 해당 구의 기간 평균보다 사용량이 높은 달, "
                "음수는 낮은 달입니다. 모든 달이 같은 구는 빈칸으로 표시됩니다."
            )

    if hm.empty:
        st.info("유효한 데이터가 없습니다. 월 선택을 조정하세요.")
    else:
        fig = px.imshow(
            hm,
            aspect="auto",
            color_continuous_scale=color_scale,
            color_continuous_midpoint=midpoint,
            labels={
                "x": "월",
                "y": "지역",
                "color": color_label,
            },
        )
        fig.update_layout(height=max(450, len(hm) * 25))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "빈칸은 결측값입니다. 원본 값은 사용 규모를, "
            "비율·표준화는 지역 내부의 월별 패턴을 보여줍니다."
        )


# =========================================================
# 군집 산점도
# =========================================================
with tab_cluster:
    st.subheader("월별 전력 사용 패턴이 비슷한 구 묶기")

    basis = st.radio(
        "군집 기준",
        ["월별 비율 — 패턴 중심", "월별 원본 값 — 규모 포함"],
    )

    features = matrix.dropna(how="any").copy()
    excluded_count = len(matrix) - len(features)

    if excluded_count:
        st.caption(
            f"선택한 월 중 결측이 있는 {excluded_count}개 구는 군집에서 제외됩니다."
        )

    if basis.startswith("월별 비율"):
        features = features.loc[features.sum(axis=1) > 0]
        features = features.div(features.sum(axis=1), axis=0)

    # 일정한 특성은 거리 계산에 도움이 되지 않아 제거
    varying_cols = features.columns[
        features.nunique(dropna=True) > 1
    ].tolist()

    if len(features) < 3 or len(varying_cols) < 2:
        st.warning(
            "군집 산점도에는 유효한 구가 3개 이상이고, "
            "값이 서로 다른 월 열이 2개 이상 필요합니다."
        )
    else:
        X = StandardScaler().fit_transform(features[varying_cols])
        unique_count = len(np.unique(X, axis=0))
        max_k = min(8, len(features) - 1, unique_count)

        if max_k < 2:
            st.warning("서로 다른 패턴이 충분하지 않아 군집을 만들 수 없습니다.")
        else:
            k = int(st.number_input(
                "군집 수",
                min_value=2,
                max_value=max_k,
                value=min(3, max_k),
            ))

            model = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=42,
            )
            labels = model.fit_predict(X)
            label_count = len(np.unique(labels))

            if 1 < label_count < len(features):
                score = silhouette_score(X, labels)
                st.metric("실루엣 점수", f"{score:.3f}")
                st.caption(
                    "-1~1 범위이며 높을수록 군집 구분이 뚜렷합니다. "
                    "좋은 설치 입지를 찾았다는 점수는 아닙니다."
                )

            pca = PCA(n_components=2)
            coordinates = pca.fit_transform(X)

            result = pd.DataFrame({
                "지역": features.index,
                "군집": (labels + 1).astype(str),
                "PC1": coordinates[:, 0],
                "PC2": coordinates[:, 1],
            })

            fig = px.scatter(
                result,
                x="PC1",
                y="PC2",
                color="군집",
                text="지역",
                hover_name="지역",
                category_orders={
                    "군집": [str(i) for i in range(1, k + 1)]
                },
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            explained = pca.explained_variance_ratio_.sum() * 100
            st.caption(
                f"PC1·PC2는 월별 특성을 요약한 축입니다. "
                f"두 축의 설명 분산 비율 합계는 {explained:.1f}%입니다. "
                "군집은 이 그림이 아니라 전체 입력 특성으로 계산했습니다."
            )

            st.subheader("군집별 특징 확인")

            profile_source = features.copy()
            profile_source["군집"] = (labels + 1).astype(str)
            profile = profile_source.groupby("군집").mean()

            profile_display = profile.copy()
            if basis.startswith("월별 비율"):
                profile_display *= 100

            st.dataframe(
                profile_display,
                use_container_width=True,
            )

            for cluster_id in sorted(result["군집"].unique(), key=int):
                regions = result.loc[
                    result["군집"] == cluster_id, "지역"
                ].tolist()

                peak = profile.loc[cluster_id].idxmax()
                st.write(
                    f"**군집 {cluster_id}** — {', '.join(regions)}"
                )
                st.caption(
                    f"선택 기간 내 군집 평균이 가장 높은 월: {peak}"
                )

            st.download_button(
                "군집 결과 다운로드",
                csv_bytes(result),
                "cluster_result.csv",
                "text/csv",
            )


# =========================================================
# 인사이트 안내
# =========================================================
st.divider()
st.subheader("📝 그래프에서 발견할 내용")

st.markdown("""
- **막대그래프:** 사용량이 큰 구와 작은 구의 차이는 어느 정도인가?
- **히트맵:** 사용량이 큰 구와 여름에 집중해서 사용하는 구는 같은가?
- **군집:** 어떤 구들이 비슷하게 묶이며, 군집별 월별 특징은 무엇인가?
""")

st.warning(
    "월별 사용량만으로 낮 시간 수요나 태양광 설치 최적 입지를 "
    "확정할 수 없습니다. 실제 설치 제안에는 일사량, 설치 가능 면적, "
    "시간대별 수요 등의 추가 자료가 필요합니다."
)

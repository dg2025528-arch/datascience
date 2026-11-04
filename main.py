from pathlib import Path
from io import BytesIO
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="2026 서울 전력·태양광 분석",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ 2026 서울 25개 구 전력 사용량과 태양광 설비 분석")
st.caption("전력 사용량과 설치된 태양광 설비로 구를 묶고, 추가 설치 우선 검토 지역을 살펴봅니다.")

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
# 공통 도우미
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


def overview(df):
    """자료형, 빈 값 개수 표"""
    return pd.DataFrame({
        "자료형": df.dtypes.astype(str),
        "빈 값 개수": df.isna().sum(),
        "빈 값 비율(%)": (df.isna().mean() * 100).round(1),
    })


def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def finding(text, evidence):
    st.success(f"**찾은 것:** {text}  \n**근거:** {evidence}")


# =========================================================
# 전력 사용량 파일 탐색 및 읽기
# =========================================================
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
            is_power = "energyuse" in name or "전력사용량" in name
            is_2025 = bool(re.search(r"2025|(?<!\d)25(?!\d)", name))
            is_2026 = bool(re.search(r"2026|26\.8", name))

            if is_power and is_2025 and not is_2026:
                result.append(path)

    return sorted(
        result,
        key=lambda p: (0 if "energyuse" in p.stem.lower() else 1, str(p)),
    )


@st.cache_data
def read_raw(content, suffix):
    if suffix != ".csv":
        return pd.read_excel(BytesIO(content), header=None)

    for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(BytesIO(content), header=None, encoding=encoding)
        except UnicodeError:
            continue
    raise ValueError("CSV 인코딩을 읽지 못했습니다.")


def find_header(raw):
    for row in range(min(20, len(raw))):
        values = {normalize(v) for v in raw.iloc[row]}
        if "1월" in values and ("자치구명" in values or "시군구" in values):
            return row
    raise ValueError(
        "제목 행을 찾지 못했습니다. 자치구명 또는 시군구, 1월~12월 열이 있는지 확인하세요."
    )


def build_table(raw, header_row):
    names, used = [], set()
    for i, value in enumerate(raw.iloc[header_row]):
        base = normalize(value) or f"빈열{i + 1}"
        name, duplicate = base, 2
        while name in used:
            name = f"{base}_{duplicate}"
            duplicate += 1
        names.append(name)
        used.add(name)

    df = raw.iloc[header_row + 1:].copy()
    df.columns = names
    return (
        df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
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

    record("제목 행 적용", len(df), len(df),
           f"첫 줄을 0으로 센 제목 행 번호: {header_row}")

    is_kepco = {"연도", "시도", "시군구", "업종별"}.issubset(df.columns)

    if is_kepco:
        source = "한전 시군구별 용도업종별 전력사용량"
        region_col = "시군구"

        before = len(df)
        df = df[to_number(df["연도"]).eq(2025)].copy()
        record("2025년만 선택", before, len(df))

        before = len(df)
        df = df[df["시도"].map(normalize).eq("서울특별시")].copy()
        record("서울특별시만 선택", before, len(df),
               "다른 시도의 같은 이름 구 제외")

        before = len(df)
        df = df[df["업종별"].map(normalize).eq("합계")].copy()
        record("업종별 합계 행만 사용", before, len(df),
               "세부 업종 행 제외하여 이중 집계 방지")

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
        raise ValueError("월별 열이 없습니다: " + ", ".join(missing_columns))

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

    record("숫자 변환 및 단위 통일", len(matrix), len(matrix),
           "kWh ÷ 1000 → MWh" if is_kepco else "원본 MWh 유지")

    before = len(matrix)
    matrix = matrix.dropna(how="any")
    record("12개월 완전한 구만 사용", before, len(matrix),
           "결측값을 0으로 채우지 않음")

    if (matrix < 0).any().any():
        raise ValueError("구별 합계에 음수가 있습니다. 원본 확인이 필요합니다.")

    before = len(matrix)
    matrix = matrix.loc[matrix.sum(axis=1) > 0]
    record("연간 합계가 양수인 구 확인", before, len(matrix))

    return matrix.sort_index(), pd.DataFrame(logs), source, df


# =========================================================
# 전력 데이터 선택 및 전처리
# =========================================================
files = find_2025_files()

if not files:
    st.error(
        "2025년 전력 사용량 파일을 찾지 못했습니다. "
        "energyUse_data 25.xlsx 또는 시군구별전력사용량 2025.xlsx를 "
        "main.py 옆이나 data 폴더에 올려 주세요."
    )
    st.stop()

st.sidebar.header("📂 데이터 설정")
path = st.sidebar.selectbox(
    "2025년 전력 자료 선택",
    files,
    format_func=lambda p: str(p.relative_to(BASE_DIR)),
)

try:
    matrix, logs, source, original = prepare_data(path)
except Exception as exc:
    st.error(f"데이터 처리 오류: {exc}")
    st.stop()

if matrix.empty:
    st.error("12개월 값이 모두 있는 구가 없습니다.")
    st.stop()


# =========================================================
# 태양광 데이터 읽기 (컬럼은 사이드바에서 선택)
# =========================================================
SOLAR_FILE = BASE_DIR / "서울시 태양광발전사업 정보.csv"


@st.cache_data
def read_solar(path_str):
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path_str, encoding=enc)
        except UnicodeError:
            continue
    raise ValueError("태양광 CSV 인코딩을 읽지 못했습니다.")


if not SOLAR_FILE.exists():
    st.error(f"태양광 파일이 없습니다: {SOLAR_FILE.name}")
    st.stop()

solar_raw = read_solar(str(SOLAR_FILE))


def guess(cols, keywords):
    for i, c in enumerate(cols):
        if any(kw in str(c) for kw in keywords):
            return i
    return 0


st.sidebar.header("☀️ 태양광 데이터 설정")
cols = solar_raw.columns.tolist()
addr_col = st.sidebar.selectbox(
    "구 이름이 들어 있는 컬럼", cols,
    index=guess(cols, ["주소", "소재지", "위치"]),
)
cap_col = st.sidebar.selectbox(
    "설비용량 컬럼", cols, index=guess(cols, ["용량"]),
)
cap_unit = st.sidebar.radio("용량 단위", ["kW", "MW", "W"], horizontal=True)
drop_extreme = st.sidebar.checkbox("사업장 용량 극단값 제외 (IQR×3)", value=False)

st.sidebar.header("⚙️ 분석 가정")
yield_mwh_per_kw = st.sidebar.slider(
    "kW당 연간 발전량 가정 (MWh/kW)", 0.9, 1.5, 1.2, 0.05
)
k = int(st.sidebar.number_input("군집 수 (k)", 2, 6, 3))
w_demand = st.sidebar.slider("우선순위 점수: 수요 가중치", 0.0, 1.0, 0.5, 0.1)


# =========================================================
# 태양광 전처리 + 로그
# =========================================================
def prepare_solar(raw):
    logs = []

    def rec(stage, before, after, note=""):
        logs.append({"데이터": "태양광", "처리 단계": stage,
                     "처리 전 행 수": before, "처리 후 행 수": after,
                     "설명": note})

    df = raw.copy()
    rec("원본", len(df), len(df))

    unit_to_kw = {"kW": 1, "MW": 1000, "W": 0.001}[cap_unit]
    df["용량_kW"] = to_number(df[cap_col]) * unit_to_kw

    before = len(df)
    df = df.dropna(subset=["용량_kW"])
    rec("용량 결측·숫자 변환 실패 제거", before, len(df), "빈 값이나 숫자가 아닌 값")

    before = len(df)
    df = df[df["용량_kW"] > 0]
    rec("용량 0 이하 제거", before, len(df), "0 또는 음수 용량")

    pattern = "(" + "|".join(sorted(SEOUL_GU)) + ")"
    df["자치구"] = df[addr_col].astype(str).str.extract(pattern)[0]
    before = len(df)
    df = df.dropna(subset=["자치구"])
    rec("서울 구 추출 실패 제거", before, len(df),
        "주소에서 25개 구 이름을 찾지 못한 행")

    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame(logs)

    lo, hi = iqr_bounds(df["용량_kW"], k=3)
    n_extreme = int((df["용량_kW"] > hi).sum())
    before = len(df)
    if drop_extreme:
        df = df[df["용량_kW"] <= hi]
        note = f"상한 {hi:,.0f} kW 초과 {n_extreme}건 제외"
    else:
        note = f"상한 {hi:,.0f} kW 초과 {n_extreme}건 발견, 제외하지 않고 유지"
    rec("사업장 용량 극단값(IQR×3)", before, len(df), note)

    agg = df.groupby("자치구").agg(
        설비건수=("용량_kW", "size"),
        설비용량_kW=("용량_kW", "sum"),
    )
    return df, agg, pd.DataFrame(logs)


solar_clean, solar_agg, solar_logs = prepare_solar(solar_raw)

if solar_clean.empty:
    st.error("태양광 전처리 후 남은 행이 없습니다. 사이드바에서 컬럼·단위를 확인하세요.")
    st.dataframe(solar_logs, hide_index=True)
    st.stop()


# =========================================================
# 수요 + 공급 결합
# =========================================================
annual = matrix.sum(axis=1)
shares = matrix.div(annual, axis=0) * 100

lo_d, hi_d = iqr_bounds(annual)
demand_outliers = annual[(annual < lo_d) | (annual > hi_d)].index.tolist()
logs = logs.copy()
logs.insert(0, "데이터", "전력 수요")
logs = pd.concat([logs, pd.DataFrame([{
    "데이터": "전력 수요",
    "처리 단계": "구별 연간 사용량 이상치(IQR×1.5)",
    "처리 전 행 수": len(matrix),
    "처리 후 행 수": len(matrix),
    "설명": (f"이상치 후보: {', '.join(demand_outliers)} (유지)"
             if demand_outliers else "이상치 없음"),
}])], ignore_index=True)

feat = pd.DataFrame(index=matrix.index)
feat["연간소비량_MWh"] = annual
feat["설비건수"] = solar_agg["설비건수"].reindex(feat.index).fillna(0)
feat["설비용량_kW"] = solar_agg["설비용량_kW"].reindex(feat.index).fillna(0)
feat["추정발전량_MWh"] = feat["설비용량_kW"] * yield_mwh_per_kw
feat["자급률_%"] = feat["추정발전량_MWh"] / feat["연간소비량_MWh"] * 100
feat["여름비중_%"] = shares[["7월", "8월"]].sum(axis=1)
feat["겨울비중_%"] = shares[["1월", "2월", "12월"]].sum(axis=1)

ranking = (
    annual.rename("연간 사용량(MWh)")
    .sort_values(ascending=False)
    .rename_axis("자치구").reset_index()
)

st.write(
    f"**분석 연도:** 2025년 · **전력 출처:** {source} · "
    f"**공급 데이터:** {SOLAR_FILE.name} · **분석 지역:** {len(matrix)}개 구"
)
missing_gu = sorted(SEOUL_GU - set(matrix.index))
if missing_gu:
    st.warning("분석에서 빠진 서울 구: " + ", ".join(missing_gu))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 데이터 개요·전처리",
    "📊 수요 현황",
    "☀️ 공급·자급률",
    "🧩 군집 분석",
    "🎯 우선순위",
    "⚠️ 방법·한계",
])


# =========================================================
# 탭 1. 데이터 개요 + 처리 로그
# =========================================================
with tab1:
    st.subheader("원본 데이터")

    # 전력 사용량: 제목 행 적용 전의 진짜 원본
    power_raw = read_raw(path.read_bytes(), path.suffix.lower())

    st.markdown(f"#### 전력 사용량 원본 ({path.name})")
    st.write(f"**행 {power_raw.shape[0]}개 × 열 {power_raw.shape[1]}개** · "
             f"빈 값 {int(power_raw.isna().sum().sum())}개")
    st.dataframe(power_raw.fillna("").astype(str),
                 use_container_width=True, hide_index=True)

    st.markdown(f"#### 태양광 발전사업 원본 ({SOLAR_FILE.name})")
    st.write(f"**행 {solar_raw.shape[0]}개 × 열 {solar_raw.shape[1]}개** · "
             f"빈 값 {int(solar_raw.isna().sum().sum())}개")
    st.dataframe(solar_raw, use_container_width=True, hide_index=True)

    st.subheader("자료형·빈 값 개수")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**전력 사용량 (제목 행 적용 후)**")
        st.caption("제목 행을 찾으려고 문자열로 읽어서 자료형이 object로 나옵니다.")
        st.dataframe(overview(original), use_container_width=True)
    with c2:
        st.markdown("**태양광 발전사업**")
        st.dataframe(overview(solar_raw), use_container_width=True)

    st.markdown("**분석에 쓰는 구 × 월 표 (숫자 변환 후, MWh)**")
    st.write(f"행 {matrix.shape[0]}개 × 열 {matrix.shape[1]}개")
    st.dataframe(matrix, use_container_width=True)
    st.dataframe(overview(matrix), use_container_width=True)

    st.subheader("결측치·이상치 처리와 처리 전후 행 수")
    all_logs = pd.concat([logs, solar_logs], ignore_index=True)
    st.dataframe(all_logs, use_container_width=True, hide_index=True)
    st.caption(
        "이상치 원칙: 구 단위(25개)는 값을 지우면 정보 손실이 커서 "
        "IQR로 표시만 하고 유지합니다. 태양광 사업장 단위는 극단값 제외를 "
        "사이드바에서 선택할 수 있습니다."
    )
    st.download_button("전처리 로그 다운로드", to_csv_bytes(all_logs),
                       "preprocess_log.csv", "text/csv")
# =========================================================
# 탭 2. 수요 현황
# =========================================================
with tab2:
    # ---------------- 막대그래프 (원본 복구) ----------------
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

    st.divider()

    # ---------------- 히트맵 (원본 복구) ----------------
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
# 탭 3. 공급·자급률
# =========================================================
with tab3:
    st.subheader("질문 3. 구별 태양광 설비는 얼마나 있는가?")
    sup = feat.reset_index().sort_values("설비용량_kW", ascending=False)
    fig = px.bar(
        sup, x="자치구", y="설비용량_kW",
        title="자치구별 태양광 발전사업 설비용량 합계",
        color="설비용량_kW", color_continuous_scale="Blues",
    )
    fig.update_layout(height=480, xaxis_title="자치구",
                      yaxis_title="설비용량 합계 (kW)")
    st.plotly_chart(fig, use_container_width=True)

    s_top, s_bot = sup.iloc[0], sup.iloc[-1]
    zero_gu = int((feat["설비용량_kW"] == 0).sum())
    finding(
        f"설비용량은 {s_top['자치구']}가 가장 크고 {s_bot['자치구']}가 가장 작습니다.",
        f"{s_top['자치구']} {s_top['설비용량_kW']:,.0f} kW, "
        f"{s_bot['자치구']} {s_bot['설비용량_kW']:,.0f} kW, "
        f"설비가 0인 구 {zero_gu}개, 전체 {len(solar_clean):,}건 집계",
    )

    st.divider()
    st.subheader("질문 4. 소비량 대비 공급이 부족한 구는 어디인가?")
    ss = feat.reset_index().sort_values("자급률_%")
    fig = px.bar(
        ss, x="자치구", y="자급률_%",
        title=f"자치구별 태양광 자급률 (kW당 {yield_mwh_per_kw} MWh/년 가정)",
        color="자급률_%", color_continuous_scale="Greens",
    )
    fig.update_layout(height=480, xaxis_title="자치구",
                      yaxis_title="자급률 = 추정 발전량 ÷ 연간 소비량 (%)")
    st.plotly_chart(fig, use_container_width=True)

    low, high = ss.iloc[0], ss.iloc[-1]
    ratio_txt = (f"{high['자급률_%'] / low['자급률_%']:.1f}배"
                 if low["자급률_%"] > 0 else "하위 구는 0%")
    finding(
        f"{low['자치구']}의 자급률이 가장 낮습니다.",
        f"{low['자치구']} {low['자급률_%']:.3f}% vs "
        f"{high['자치구']} {high['자급률_%']:.3f}% ({ratio_txt}), "
        f"전체 평균 {feat['자급률_%'].mean():.3f}%",
    )
    st.caption("등록 사업자 설비 기준이라 소규모·자가용 설비는 빠졌을 수 있어 "
               "절대값이 아니라 구 간 상대 비교용으로 봐야 합니다.")


# =========================================================
# 탭 4. 군집 분석 (전력 사용량 × 태양광 설비)
# =========================================================
with tab4:
    st.subheader("전력 사용량과 태양광 설비로 구 묶기")

    cluster_input = pd.DataFrame({
        "log_소비량": np.log10(feat["연간소비량_MWh"]),
        "log_설비용량": np.log10(feat["설비용량_kW"] + 1),  # 0kW 구 대비 +1
    })
    Xs = StandardScaler().fit_transform(cluster_input)

    r = cluster_input["log_소비량"].corr(cluster_input["log_설비용량"])
    st.caption(f"두 변수(log10)의 상관계수: {r:.2f} "
               "(수요 큰 구에 설비도 많은 정도를 보여줍니다)")

  
    # ---- 질문 6: 군집 산점도
    st.divider()
    st.markdown("#### 질문 6. 전력 사용량 대비 태양광 설비가 적은 군집은 어디인가?")

    plot_df = feat.reset_index().copy()
    plot_df["설비용량_표시"] = plot_df["설비용량_kW"].clip(lower=1)  # 로그 축용

    fig = px.scatter(
        plot_df,
        x="연간소비량_MWh", y="설비용량_표시",
        color="군집", text="자치구",
        log_x=True, log_y=True,
        hover_data={"설비용량_kW": ":,.0f", "설비건수": True,
                    "연간소비량_MWh": ":,.0f", "설비용량_표시": False},
        title="자치구별 전력 사용량과 태양광 설비용량 (군집별 색)",
        category_orders={"군집": [str(i) for i in range(1, k + 1)]},
    )
    fig.update_traces(textposition="top center", marker={"size": 11})
    fig.add_vline(x=feat["연간소비량_MWh"].median(), line_dash="dot",
                  line_color="gray")
    fig.add_hline(y=max(feat["설비용량_kW"].median(), 1), line_dash="dot",
                  line_color="gray")
    fig.update_layout(
        height=620,
        xaxis_title="연간 전력 사용량 (MWh, 로그 축)",
        yaxis_title="설치된 태양광 설비용량 합계 (kW, 로그 축)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("점선은 25개 구의 중앙값입니다. 오른쪽 아래(사용량 많음·설비 적음)가 "
               "추가 설치 검토 영역입니다. 설비 0kW인 구는 로그 축 표시를 위해 "
               "1kW 위치에 찍습니다.")

    profile = feat.groupby("군집")[
        ["연간소비량_MWh", "설비용량_kW", "설비건수"]
    ].mean().round(1)
    profile["구 수"] = feat.groupby("군집").size()
    profile["kW_per_GWh"] = (profile["설비용량_kW"]
                             / (profile["연간소비량_MWh"] / 1000)).round(2)
    st.dataframe(profile, use_container_width=True)

    prio = profile["kW_per_GWh"].idxmin()
    members = feat.index[feat["군집"] == prio].tolist()
    finding(
        f"군집 {prio}가 전력 사용량에 비해 태양광 설비가 가장 적어 설치 우선 검토 대상입니다.",
        f"군집 평균 사용량 {profile.loc[prio, '연간소비량_MWh']:,.0f} MWh, "
        f"평균 설비 {profile.loc[prio, '설비용량_kW']:,.0f} kW "
        f"(사용량 1GWh당 {profile.loc[prio, 'kW_per_GWh']} kW), "
        f"구성: {', '.join(members)}",
    )

    # ---- 질문 7: 덴드로그램
    st.divider()
    st.markdown("#### 질문 7. 어떤 구끼리 가까운가? (계층 구조)")
    names = cluster_input.index.tolist()
    fig = ff.create_dendrogram(
        Xs, labels=names,
        linkagefun=lambda x: linkage(x, "ward"),
    )
    fig.update_layout(
        title="자치구 계층적 군집 덴드로그램 (ward)", height=500,
        xaxis_title="자치구", yaxis_title="군집 간 거리",
    )
    st.plotly_chart(fig, use_container_width=True)

    d = linkage(Xs, "ward")
    first_pair = int(np.argmin(d[:, 2]))
    a, b = int(d[first_pair, 0]), int(d[first_pair, 1])
    if a < len(names) and b < len(names):
        finding(
            f"{names[a]}와 {names[b]}가 가장 먼저 합쳐지는, 가장 비슷한 구입니다.",
            f"병합 거리 {d[first_pair, 2]:.2f} (전체 최대 병합 거리 {d[:, 2].max():.2f})",
        )

    st.download_button("군집 결과 다운로드", to_csv_bytes(feat.reset_index()),
                       "cluster_result.csv", "text/csv")


# =========================================================
# 탭 5. 우선순위
# =========================================================
with tab5:
    st.subheader("질문 8. 태양광을 더 지을 우선 검토 구는 어디인가?")
    st.caption(
        f"점수 = 수요 순위 백분위 × {w_demand:.1f} + "
        f"공급 부족 순위 백분위 × {1 - w_demand:.1f} (0~100점). "
        "가중치는 사이드바에서 조절합니다."
    )

    score = pd.DataFrame(index=feat.index)
    score["수요점수"] = feat["연간소비량_MWh"].rank(pct=True) * 100
    score["공급부족점수"] = (-feat["자급률_%"]).rank(pct=True) * 100
    score["우선순위점수"] = (
        w_demand * score["수요점수"] + (1 - w_demand) * score["공급부족점수"]
    )
    score["군집"] = feat["군집"]
    score = score.sort_values("우선순위점수", ascending=False).round(1)

    fig = px.bar(
        score.reset_index(), x="자치구", y="우선순위점수", color="군집",
        title="자치구별 태양광 추가 설치 우선순위 점수",
    )
    fig.update_layout(height=500, xaxis_title="자치구",
                      yaxis_title="우선순위 점수 (0~100)",
                      xaxis={"categoryorder": "array",
                             "categoryarray": score.index.tolist()})
    st.plotly_chart(fig, use_container_width=True)

    top5 = score.head(5)
    finding(
        f"우선 검토 상위 5개 구는 {', '.join(top5.index)}입니다.",
        f"1위 {top5.index[0]} {top5['우선순위점수'].iloc[0]:.1f}점, "
        f"5위 {top5.index[-1]} {top5['우선순위점수'].iloc[-1]:.1f}점, "
        f"수요 가중치 {w_demand:.1f}",
    )
    st.dataframe(score, use_container_width=True)
    st.download_button("우선순위 다운로드", to_csv_bytes(score.reset_index()),
                       "priority.csv", "text/csv")


# =========================================================
# 탭 6. 방법·한계
# =========================================================
with tab6:
    st.markdown(f"""
**방법**
- 수요: 2025년 구별 월별 전력 사용량 → 연간 합계(MWh)
- 공급: 태양광 발전사업 CSV를 구별로 집계(건수, 설비용량)
- 자급률 = 설비용량(kW) × {yield_mwh_per_kw} MWh/kW ÷ 연간 소비량(MWh)
- 군집: 연간 소비량과 설비용량(log10)을 표준화해 KMeans로 묶음 (k={k})
- 우선순위: 수요와 공급 부족의 순위 백분위 가중합

**한계**
- 구 25개, 1년 자료라 군집은 확정이 아닌 **탐색적 분류**입니다.
- 두 변수 모두 규모 기반이라 군집이 "규모 등급"으로 나뉘기 쉽습니다.
- 발전량 계수 {yield_mwh_per_kw} MWh/kW는 가정값입니다. 일사량·설치 조건은 반영하지 않았습니다.
- 발전사업 등록 설비 위주라 소규모·자가용 설비가 누락될 수 있습니다.
- 설치 가능 면적, 일사량, 계통 여건은 포함하지 않아 입지를 **확정하지 않고 우선 검토 대상**만 제시합니다.
""")

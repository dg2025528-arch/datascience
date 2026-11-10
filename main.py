
# 실행: python -m streamlit run main.py
# 필수 패키지: streamlit pandas numpy plotly scikit-learn
# Excel 사용 시 추가: openpyxl (.xlsx), xlrd (.xls)
# CSV/Excel 파일은 main.py 옆 또는 data 폴더에 놓습니다.

from pathlib import Path
from io import BytesIO
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. 기본 설정
# =========================================================
st.set_page_config(
    page_title="서울 전력·태양광 분석",
    page_icon="⚡",
    layout="wide",
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

st.title("⚡ 서울 전력 사용량과 태양광 설비 분석")
st.caption(
    "2025년 전력 수요와 태양광 사업 자료를 비교하고, "
    "구별 특성을 군집으로 비교하고 추가 설치 검토 후보를 확인합니다."
)


# =========================================================
# 2. 공통 함수
# =========================================================
def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value)).lstrip("\ufeff")


def to_number(series):
    text = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(text, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def describe_columns(df):
    return pd.DataFrame({
        "컬럼명": [str(c) for c in df.columns],
        "자료형": [str(t) for t in df.dtypes],
        "결측치 수": df.isna().sum().to_numpy(),
        "결측치 비율(%)": (
            df.isna().mean().to_numpy() * 100
        ).round(2),
    })


def show_table(df, **kwargs):
    # 혼합 자료형으로 인한 표 렌더링 문제 방지
    display = df.copy().astype("string").fillna("")
    st.dataframe(display, use_container_width=True, **kwargs)


def finding(text, evidence):
    st.success(
        f"**찾은 것:** {text}\n\n"
        f"**근거 수치:** {evidence}"
    )


def iqr_limits(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    width = q3 - q1
    return q1 - 1.5 * width, q3 + 1.5 * width


def add_log(logs, dataset, stage, before, after, note=""):
    logs.append({
        "데이터": dataset,
        "처리 단계": stage,
        "처리 전 행 수": before,
        "처리 후 행 수": after,
        "설명": note,
    })


def discover_files():
    result = []
    for folder in [BASE_DIR, BASE_DIR / "data"]:
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() in {".csv", ".xlsx", ".xls"}
                and not path.name.startswith("~$")
            ):
                result.append(path)
    return sorted(result, key=lambda p: p.name)


@st.cache_data
def read_file(content, suffix):
    """CSV/Excel을 헤더 지정 없이 읽음."""
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(content), header=None)

    for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(
                BytesIO(content),
                header=None,
                encoding=encoding,
                low_memory=False,
            )
        except UnicodeError:
            continue

    raise ValueError("CSV 인코딩을 읽지 못했습니다.")


def make_table(raw, header_row):
    names = []
    used = set()

    for i, value in enumerate(raw.iloc[header_row]):
        base = normalize(value) or f"빈열_{i + 1}"
        name = base
        count = 2

        while name in used:
            name = f"{base}_{count}"
            count += 1

        names.append(name)
        used.add(name)

    table = raw.iloc[header_row + 1:].copy()
    table.columns = names

    # 빈 문자열도 결측치로 인식
    table = table.replace(r"^\s*$", np.nan, regex=True)
    return table.reset_index(drop=True)


def guess_column(columns, words):
    for i, column in enumerate(columns):
        if any(word in str(column) for word in words):
            return i
    return 0


def extract_gu(value):
    """구 이름을 주소의 독립된 단어로 추출."""
    text = str(value).strip()

    if normalize(text) in SEOUL_GU:
        return normalize(text)

    # 다른 시도의 중구·강서구 등을 서울로 오인하지 않도록 방지
    other_city = re.match(
        r"^(부산|대구|인천|광주|대전|울산|세종|경기|강원|충청|전라|경상|제주)",
        text,
    )
    if other_city:
        return None

    pattern = (
        r"(?<![가-힣])("
        + "|".join(sorted(SEOUL_GU, key=len, reverse=True))
        + r")(?![가-힣])"
    )
    match = re.search(pattern, text)
    return match.group(1) if match else None



PRIORITY_MODES = [
    "기존 조건: 수요 높음·설비용량 적음",
    "수요 대비 설비가 적은 순",
    "수요 중앙값 초과 구 중 설비비율이 낮은 순",
]

FEATURE_LABELS = {
    "연간소비량_MWh": "연간 전력 사용량",
    "설비용량_kW": "태양광 설비용량",
    "여름사용비중_pct": "여름(6~8월) 전력 비중",
    "가정용비중_pct": "가정용부문 전력 비중",
    "서비스업비중_pct": "순수써비스 업종 전력 비중",
}


def make_sector_features(sector_source, annual, factor):
    """선택한 전력 파일 안에서만 업종 비중을 계산합니다."""
    result = pd.DataFrame(index=annual.index)
    result.index.name = "자치구"
    notes = []
    if sector_source is None:
        return result, notes
    work = sector_source.copy()
    work["시군구"] = work["시군구"].map(normalize)
    work["업종별"] = work["업종별"].map(normalize)
    for sector, column in [
        ("가정용부문", "가정용비중_pct"),
        ("순수써비스", "서비스업비중_pct"),
    ]:
        rows = work[work["업종별"].eq(sector) & work["시군구"].isin(annual.index)]
        if rows.empty:
            continue
        if rows["시군구"].duplicated().any():
            notes.append(f"{sector}: 구별 중복 행이 있어 이 특성을 사용하지 않습니다.")
            continue
        values = rows.set_index("시군구")[MONTHS].apply(to_number) * factor
        invalid = values.isna().any(axis=1) | (values < 0).any(axis=1)
        totals = values.sum(axis=1, min_count=12).mask(invalid)
        ratio = totals.reindex(annual.index).div(annual).mul(100)
        # '-'를 0으로 간주하지 않고, 합계보다 큰 세부값도 확인 전에는 제외한다.
        valid = ratio.between(0, 100 + 1e-6)
        result[column] = ratio.where(valid)
        if not valid.all():
            notes.append(
                f"{sector}: 미집계·결측·음수·합계 불일치로 "
                f"{int((~valid).sum())}개 구의 비중을 계산하지 못했습니다."
            )
    return result, notes


def combine_demand_solar(annual, solar_agg):
    """자료에 없는 설비는 NaN으로 보존하고 실제 0과 구분합니다."""
    demand = annual.rename("연간소비량_MWh").rename_axis("자치구").reset_index()
    result = demand.merge(solar_agg, on="자치구", how="left", validate="one_to_one")
    result["자료내등록없음"] = result["설비용량_kW"].isna()
    result["설비건수"] = result["설비건수"].astype("Int64")
    result["수요대비설비_kW_GWh"] = (
        result["설비용량_kW"] / (result["연간소비량_MWh"] / 1000)
    )
    return result.sort_values("연간소비량_MWh", ascending=False).reset_index(drop=True)


def build_priority(frame, mode, limit=5):
    """군집/산점도와 독립된 공개 조건으로 추가 조사 후보를 선정합니다."""
    if mode not in PRIORITY_MODES:
        raise ValueError("지원하지 않는 추천 기준입니다.")
    valid = frame.loc[
        ~frame["자료내등록없음"]
        & frame["연간소비량_MWh"].gt(0)
        & frame["설비용량_kW"].ge(0)
    ].copy()
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["연간소비량_MWh", "설비용량_kW", "수요대비설비_kW_GWh"]
    )
    demand_median = float(valid["연간소비량_MWh"].median())
    solar_median = float(valid["설비용량_kW"].median())
    valid["수요중앙값차이_MWh"] = valid["연간소비량_MWh"] - demand_median
    valid["용량중앙값차이_kW"] = valid["설비용량_kW"] - solar_median
    low_capacity = valid["설비용량_kW"].lt(solar_median)
    zero_median = bool(not valid.empty and solar_median == 0)
    if zero_median:
        # 검증된 0이 입력된 경우에 한함. 미등록 구는 이미 제외되어 있다.
        low_capacity = valid["설비용량_kW"].eq(0)
    valid["기존조건후보"] = valid["연간소비량_MWh"].gt(demand_median) & low_capacity
    valid = valid.sort_values(
        ["수요대비설비_kW_GWh", "연간소비량_MWh", "자치구"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    valid.insert(0, "비율순서", np.arange(1, len(valid) + 1))
    valid["선정"] = False
    if mode == PRIORITY_MODES[0]:
        valid["선정"] = valid["기존조건후보"]
        rule = "전력 사용량 > 수요 중앙값, 설비용량 < 용량 중앙값"
        if zero_median:
            rule = "전력 사용량 > 수요 중앙값, 확인된 설비용량 = 0"
    elif mode == PRIORITY_MODES[1]:
        valid.loc[valid.head(int(limit)).index, "선정"] = True
        rule = f"수요 대비 설비용량(kW/GWh)이 낮은 순서로 최대 {int(limit)}개 구"
    else:
        pool = valid[valid["연간소비량_MWh"].gt(demand_median)]
        valid.loc[pool.head(int(limit)).index, "선정"] = True
        rule = (
            f"수요 중앙값을 초과한 구 중 설비용량/전력수요가 낮은 순서로 "
            f"최대 {int(limit)}개 구"
        )
    valid["선정근거"] = np.where(valid["선정"], rule, "선택한 조건 밖")
    return valid, {
        "demand_median": demand_median, "solar_median": solar_median,
        "n": len(valid), "zero_median": zero_median, "rule": rule,
    }


def prepare_power(raw, unit=None):
    """원래 전처리를 재사용해 2025년 서울 구별 전력 표를 만듭니다."""
    power_logs = []
    sector_source = None
    header_row = None

    for row in range(min(30, len(raw))):
        values = {normalize(v) for v in raw.iloc[row]}
        if "1월" in values and (
            "자치구명" in values or "시군구" in values
        ):
            header_row = row
            break

    if header_row is None:
        raise ValueError(
            "'자치구명' 또는 '시군구', '1월'이 포함된 제목 행이 필요합니다."
        )

    power_original = make_table(raw, header_row)
    work = power_original.copy()

    before = len(work)
    work = work.dropna(how="all")
    add_log(
        power_logs, "전력", "완전히 빈 행 제거",
        before, len(work),
    )

    is_kepco = {
        "연도", "시도", "시군구", "업종별"
    }.issubset(work.columns)

    if is_kepco:
        region_col = "시군구"
        default_unit = "kWh"

        before = len(work)
        work = work[to_number(work["연도"]).eq(2025)].copy()
        add_log(
            power_logs, "전력", "2025년 선택",
            before, len(work),
        )

        before = len(work)
        work = work[
            work["시도"].map(normalize).isin(["서울특별시", "서울"])
        ].copy()
        add_log(
            power_logs, "전력", "서울 선택",
            before, len(work),
        )

        sector_source = work.copy()

        before = len(work)
        work = work[
            work["업종별"].map(normalize).isin(["합계", "계", "총계"])
        ].copy()
        add_log(
            power_logs, "전력", "업종 합계 행 선택",
            before, len(work), "세부 업종과 합계의 이중 집계 방지",
        )
    else:
        region_col = "자치구명"
        default_unit = "MWh"

        if region_col not in work.columns:
            raise ValueError("자치구명 컬럼이 없습니다.")

        if "연도" in work.columns:
            before = len(work)
            work = work[to_number(work["연도"]).eq(2025)].copy()
            add_log(
                power_logs, "전력", "2025년 선택",
                before, len(work),
            )

    missing_columns = [m for m in MONTHS if m not in work.columns]
    if missing_columns:
        raise ValueError(
            "월별 컬럼이 없습니다: " + ", ".join(missing_columns)
        )

    power_unit = unit or default_unit
    if power_unit not in {"kWh", "MWh", "GWh"}:
        raise ValueError("전력 단위는 kWh, MWh, GWh 중 하나여야 합니다.")

    work[region_col] = work[region_col].map(normalize)

    before = len(work)
    work = work[work[region_col].isin(SEOUL_GU)].copy()
    add_log(
        power_logs, "전력", "서울 25개 구 선택",
        before, len(work),
    )

    if work[region_col].duplicated().any():
        duplicates = work.loc[
            work[region_col].duplicated(keep=False), region_col
        ].unique()
        raise ValueError(
            "구별 중복 행이 있습니다. 자동 합산하지 않았습니다: "
            + ", ".join(duplicates)
        )

    matrix = work.set_index(region_col)[MONTHS].copy()
    matrix.index.name = "자치구"

    factor = {"kWh": 0.001, "MWh": 1, "GWh": 1000}[power_unit]

    for month in MONTHS:
        matrix[month] = to_number(matrix[month]) * factor

    invalid_cells = int(matrix.isna().sum().sum())
    missing_details = [
        f"{gu}: " + ", ".join(row.index[row.isna()].tolist())
        for gu, row in matrix.iterrows() if row.isna().any()
    ]
    before = len(matrix)
    matrix = matrix.dropna(how="any")
    add_log(
        power_logs, "전력", "월별 결측·숫자 변환 실패 처리",
        before, len(matrix),
        f"결측 셀 {invalid_cells}개. 12개월이 완전한 구만 사용. "
        + "; ".join(missing_details),
    )

    before = len(matrix)
    matrix = matrix.loc[~(matrix < 0).any(axis=1)]
    add_log(
        power_logs, "전력", "음수가 포함된 구 제외",
        before, len(matrix), "원본 확인 필요",
    )

    before = len(matrix)
    matrix = matrix.loc[matrix.sum(axis=1) > 0].sort_index()
    add_log(
        power_logs, "전력", "연간 사용량 0 이하 제외",
        before, len(matrix),
    )

    if matrix.empty:
        raise ValueError("전처리 후 남은 전력 데이터가 없습니다.")

    annual = matrix.sum(axis=1)
    shares = matrix.div(annual, axis=0) * 100

    lo, hi = iqr_limits(annual)
    outliers = annual[(annual < lo) | (annual > hi)].index.tolist()

    add_log(
        power_logs, "전력", "연간 사용량 이상치 점검",
        len(matrix), len(matrix),
        "IQR×1.5 기준. 실제 지역 규모 차이일 수 있어 유지. 후보: "
        + (", ".join(outliers) if outliers else "없음"),
    )

    sector_features, sector_notes = make_sector_features(sector_source, annual, factor)
    return {
        "original": power_original, "matrix": matrix, "annual": annual,
        "shares": shares, "logs": power_logs, "unit": power_unit,
        "default_unit": default_unit, "sector_features": sector_features,
        "sector_notes": sector_notes,
    }


@st.cache_data(show_spinner=False)
def fit_cluster_models(frame, feature_cols, log_cols, k):
    """같은 전처리로 세 군집 모델을 비교하고 구별 결과와 진단 표를 반환합니다.

    군집 번호는 평균 전력 수요, 평균 설비용량 순서이며 설치 순위가 아닙니다.
    GMM의 군집소속확률은 최대 사후확률이며 설치 성공확률이 아닙니다.
    입력 오류는 즉시 알리고, 개별 모델 실패는 진단 표에 남깁니다.
    """
    feature_cols = list(feature_cols)
    log_cols = set(log_cols)
    if not feature_cols or len(set(feature_cols)) != len(feature_cols):
        raise ValueError("서로 다른 군집 입력 특성을 하나 이상 선택하세요.")
    required = set(feature_cols) | {"연간소비량_MWh", "설비용량_kW"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("군집 입력 컬럼이 없습니다: " + ", ".join(missing))
    if not log_cols.issubset(feature_cols):
        raise ValueError("로그 변환 대상은 군집 입력 특성에 포함되어야 합니다.")
    if not isinstance(k, (int, np.integer)) or not 2 <= k < len(frame):
        raise ValueError("군집 수는 2 이상이고 분석 대상 구 수보다 작아야 합니다.")

    base = frame.reset_index(drop=True).copy()
    inputs = base[feature_cols].to_numpy(dtype=float, copy=True)
    if not np.isfinite(inputs).all():
        raise ValueError("군집 입력에 결측값 또는 무한대가 있습니다.")
    ordering = base[["연간소비량_MWh", "설비용량_kW"]].to_numpy(dtype=float)
    if not np.isfinite(ordering).all():
        raise ValueError("군집 번호 정렬에 사용하는 수요·용량이 유효하지 않습니다.")
    for index, column in enumerate(feature_cols):
        if column in log_cols:
            if (inputs[:, index] < 0).any():
                raise ValueError(f"{column}: 로그 변환 대상에는 음수가 없어야 합니다.")
            inputs[:, index] = np.log1p(inputs[:, index])
    constants = [
        column for index, column in enumerate(feature_cols)
        if np.all(inputs[:, index] == inputs[0, index])
    ]
    if len(np.unique(inputs, axis=0)) < k:
        raise ValueError("서로 다른 구별 입력값 수보다 군집 수가 많습니다.")
    scaled = StandardScaler().fit_transform(inputs)
    if not np.isfinite(scaled).all():
        raise ValueError("표준화 결과가 유효하지 않습니다. 입력값 범위를 확인하세요.")
    common_notes = ["상수 특성: " + ", ".join(constants)] if constants else []

    factories = {
        "KMeans": lambda: KMeans(n_clusters=k, random_state=42, n_init=20),
        "계층적 군집 (Ward)": lambda: AgglomerativeClustering(
            n_clusters=k, linkage="ward"
        ),
        "GMM": lambda: GaussianMixture(
            n_components=k, covariance_type="diag", reg_covar=1e-3,
            n_init=10, max_iter=500, random_state=42,
        ),
    }
    labeled_frames = {}
    metrics = []
    for name, factory in factories.items():
        row = {
            "모델": name, "실루엣 점수": np.nan, "군집 수": np.nan,
            "최소 군집 구 수": np.nan, "군집 크기": "", "BIC": np.nan,
            "상태": "",
        }
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = factory()
                labels = model.fit_predict(scaled)
                actual_labels, counts = np.unique(labels, return_counts=True)
                row["군집 수"] = len(actual_labels)
                row["최소 군집 구 수"] = int(counts.min())
                if len(actual_labels) != k:
                    raise ValueError(
                        f"요청한 {k}개 중 {len(actual_labels)}개 군집만 생성되어 제외"
                    )
                if name == "GMM" and not model.converged_:
                    raise ValueError("GMM이 수렴하지 않아 결과에서 제외했습니다.")

                ordered_labels = sorted(
                    actual_labels,
                    key=lambda label: (
                        float(ordering[labels == label, 0].mean()),
                        float(ordering[labels == label, 1].mean()), int(label),
                    ),
                )
                label_map = {
                    label: str(index + 1) for index, label in enumerate(ordered_labels)
                }
                result = base.drop(columns=["군집소속확률"], errors="ignore").copy()
                result["군집"] = [label_map[label] for label in labels]
                if name == "GMM":
                    probabilities = model.predict_proba(scaled).max(axis=1)
                    if not np.isfinite(probabilities).all():
                        raise ValueError("GMM 군집소속확률이 유효하지 않습니다.")
                    result["군집소속확률"] = probabilities
                    row["BIC"] = float(model.bic(scaled))
                if 2 <= len(actual_labels) < len(base):
                    row["실루엣 점수"] = float(silhouette_score(scaled, labels))
                sizes = result["군집"].value_counts()
                row["군집 크기"] = ", ".join(
                    f"{index}군집 {int(sizes[str(index)])}개"
                    for index in range(1, k + 1)
                )
                labeled_frames[name] = result
                row["상태"] = "정상"
        except Exception as exc:
            row["상태"] = f"분석 제외: {exc}"
        # 학습 결과와 관련된 경고만 표시하고 실행환경의 코어 감지 안내 등은 제외한다.
        notes = common_notes + list(dict.fromkeys(
            str(w.message) for w in caught
            if issubclass(w.category, (ConvergenceWarning, RuntimeWarning))
        ))
        if notes:
            row["상태"] += " | " + " | ".join(notes)
        metrics.append(row)

    if not labeled_frames:
        details = "; ".join(f"{row['모델']}: {row['상태']}" for row in metrics)
        raise ValueError("모든 군집 모델을 적용하지 못했습니다. " + details)
    return labeled_frames, pd.DataFrame(metrics)


# =========================================================
# 3. 파일 선택
# =========================================================
all_files = discover_files()

power_files = [
    p for p in all_files
    if (
        "energyuse" in p.stem.lower()
        or "전력" in p.stem
    )
    and re.search(r"2025|(?<!\d)25(?!\d)", p.stem)
]

solar_files = [p for p in all_files if "태양광" in p.stem]

st.sidebar.header("📂 데이터 설정")

if not power_files:
    st.error(
        "2025년 전력 사용량 파일을 찾지 못했습니다. "
        "파일을 main.py 옆 또는 data 폴더에 넣어주세요."
    )
    st.stop()

power_path = st.sidebar.selectbox(
    "전력 사용량 파일",
    power_files,
    format_func=lambda p: p.name,
)

try:
    power_raw = read_file(
        power_path.read_bytes(), power_path.suffix.lower()
    )
except Exception as exc:
    st.error(f"전력 파일 읽기 실패: {exc}")
    st.stop()


# =========================================================
# 4. 전력 데이터 전처리
# =========================================================
try:
    power_data = prepare_power(power_raw)
    saved_power_units = st.session_state.setdefault("saved_power_units", {})
    power_identity = str(power_path.resolve())
    remembered_unit = saved_power_units.get(power_identity, power_data["default_unit"])
    power_unit = st.sidebar.selectbox(
        "전력 원본 단위", ["kWh", "MWh", "GWh"],
        index=["kWh", "MWh", "GWh"].index(remembered_unit),
        key=f"power_unit::{power_path.resolve()}",
    )
    saved_power_units[power_identity] = power_unit
    st.sidebar.caption("단위는 원본 자료 설명에서 확인하세요. 파일별로 따로 기억합니다.")
    if power_unit != power_data["unit"]:
        power_data = prepare_power(power_raw, power_unit)
    power_original = power_data["original"]
    matrix = power_data["matrix"]
    annual = power_data["annual"]
    shares = power_data["shares"]
    power_logs = power_data["logs"]
except Exception as exc:
    st.error(f"전력 전처리 오류: {exc}")
    st.markdown("#### 읽어온 원본 데이터")
    show_table(power_raw)
    st.stop()

# 선택 파일과 비교 파일은 합산하거나 평균내지 않고 각각 전처리한다.
power_sources = {power_path: power_data}
source_errors = []
for other_path in power_files:
    if other_path == power_path:
        continue
    try:
        other_raw = read_file(other_path.read_bytes(), other_path.suffix.lower())
        other_unit = saved_power_units.get(str(other_path.resolve()))
        power_sources[other_path] = prepare_power(other_raw, other_unit)
    except Exception as exc:
        source_errors.append(f"{other_path.name}: {exc}")


# =========================================================
# 5. 태양광 데이터 읽기·전처리
# =========================================================
solar_raw = None
solar_original = None
solar_clean = None
solar_agg = None
solar_logs = []
solar_error = None

if not solar_files:
    solar_error = "태양광 파일을 찾지 못했습니다."
else:
    solar_path = st.sidebar.selectbox(
        "태양광 파일",
        solar_files,
        format_func=lambda p: p.name,
    )

    try:
        solar_raw = read_file(
            solar_path.read_bytes(), solar_path.suffix.lower()
        )

        solar_header = int(st.sidebar.number_input(
            "태양광 제목 행 번호 (첫 행 = 1)",
            min_value=1,
            max_value=max(1, min(30, len(solar_raw))),
            value=1,
        ))

        solar_original = make_table(solar_raw, solar_header - 1)
        columns = solar_original.columns.tolist()

        st.sidebar.subheader("☀️ 태양광 컬럼 확인")

        addr_col = st.sidebar.selectbox(
            "주소 또는 자치구 컬럼",
            columns,
            index=guess_column(
                columns, ["소재지", "주소", "자치구", "시군구", "위치"]
            ),
        )
        cap_col = st.sidebar.selectbox(
            "설비용량 컬럼",
            columns,
            index=guess_column(columns, ["용량"]),
        )
        cap_unit = st.sidebar.selectbox(
            "설비용량 원본 단위",
            ["kW", "MW", "W"],
        )

        status_options = ["사용 안 함"] + columns
        status_default = next(
            (c for c in columns if "가동상태" in c or "운영상태" in c), "사용 안 함"
        )
        status_col = st.sidebar.selectbox(
            "운영 상태 필터 컬럼 (선택)",
            status_options, index=status_options.index(status_default),
        )

        solar_work = solar_original.copy()

        add_log(
            solar_logs, "태양광", "제목 행 적용 후 원본",
            len(solar_work), len(solar_work),
        )

        if status_col != "사용 안 함":
            states = sorted(
                solar_work[status_col].dropna().astype(str).unique().tolist()
            )
            active_states = [
                value for value in states
                if normalize(value) in {"가동중", "운영중", "정상가동", "정상운영"}
            ]
            selected_states = st.sidebar.multiselect(
                "집계할 상태", states, default=active_states or states,
                key=f"solar_states::{solar_path.resolve()}::{status_col}",
            )
            if not active_states:
                st.sidebar.caption("운영 상태의 뜻을 확인하고 집계할 값을 선택하세요.")
            before = len(solar_work)
            solar_work = solar_work[
                solar_work[status_col].astype(str).isin(selected_states)
            ].copy()
            add_log(
                solar_logs, "태양광", "운영 상태 필터",
                before, len(solar_work),
                ", ".join(selected_states),
            )

        st.sidebar.caption(
            "허가·계획 단계가 포함된 자료라면 운영 중인 상태만 선택하세요."
        )

        solar_work["자치구"] = solar_work[addr_col].map(extract_gu)
        solar_work["용량_kW"] = (
            to_number(solar_work[cap_col])
            * {"kW": 1, "MW": 1000, "W": 0.001}[cap_unit]
        )

        before = len(solar_work)
        solar_work = solar_work.dropna(subset=["자치구"])
        add_log(
            solar_logs, "태양광", "서울 자치구 추출 실패 제외",
            before, len(solar_work),
            "주소·자치구 컬럼 확인 필요",
        )

        before = len(solar_work)
        solar_work = solar_work.dropna(subset=["용량_kW"])
        add_log(
            solar_logs, "태양광", "용량 결측·숫자 변환 실패 제외",
            before, len(solar_work),
        )

        before = len(solar_work)
        solar_work = solar_work[solar_work["용량_kW"] > 0].copy()
        add_log(
            solar_logs, "태양광", "용량 0 이하 제외",
            before, len(solar_work),
        )

        if solar_work.empty:
            raise ValueError(
                "유효한 태양광 행이 없습니다. 제목 행·주소·용량·단위를 확인하세요."
            )

        lo, hi = iqr_limits(solar_work["용량_kW"])
        extreme_count = int(
            (
                (solar_work["용량_kW"] < lo)
                | (solar_work["용량_kW"] > hi)
            ).sum()
        )
        add_log(
            solar_logs, "태양광", "용량 이상치 점검",
            len(solar_work), len(solar_work),
            f"IQR×1.5 후보 {extreme_count}건. 대규모 설비일 수 있어 유지",
        )

        duplicate_count = int(solar_work.duplicated().sum())
        add_log(
            solar_logs, "태양광", "완전 동일 행 점검",
            len(solar_work), len(solar_work),
            f"완전 동일 행 {duplicate_count}건. 자동 삭제하지 않음; "
            "주소가 같아도 별도 시설·설치 단계일 수 있음",
        )
        if duplicate_count:
            st.warning(
                f"태양광에 완전 동일 행 {duplicate_count}건이 있습니다. "
                "중복 여부를 원본에서 확인하세요."
            )
        solar_clean = solar_work
        solar_agg = (
            solar_clean.groupby("자치구", as_index=False)
            .agg(
                설비건수=("용량_kW", "size"),
                설비용량_kW=("용량_kW", "sum"),
            )
        )

    except Exception as exc:
        solar_error = str(exc)


# =========================================================
# 6. 분석 테이블 및 공개된 추천 기준
# =========================================================
demand_df = (
    annual.rename("연간소비량_MWh").rename_axis("자치구").reset_index()
    .sort_values("연간소비량_MWh", ascending=False).reset_index(drop=True)
)
combined = None
priority_df = None
priority_meta = None
candidates = pd.DataFrame()
priority_mode = PRIORITY_MODES[0]
recommend_n = 5

if solar_error is None and solar_agg is not None:
    combined = combine_demand_solar(annual, solar_agg)
    summer = shares[["6월", "7월", "8월"]].sum(axis=1)
    combined["여름사용비중_pct"] = combined["자치구"].map(summer)
    sector_features = power_data["sector_features"]
    if len(sector_features.columns):
        combined = combined.merge(
            sector_features.reset_index(), on="자치구", how="left", validate="one_to_one"
        )
    known_count = int((~combined["자료내등록없음"]).sum())
    if known_count:
        st.sidebar.header("📍 추가 설치 검토 기준")
        priority_mode = st.sidebar.selectbox("후보 선정 기준", PRIORITY_MODES)
        if priority_mode != PRIORITY_MODES[0]:
            recommend_n = int(st.sidebar.number_input(
                "최대 후보 구 수", min_value=1, max_value=known_count,
                value=min(5, known_count), step=1,
            ))
        st.sidebar.caption(
            "후보 선정은 설비용량 기준입니다. 군집 모델·산점도 축을 바꿔도 "
            "추천 조건은 바뀌지 않습니다."
        )
        priority_df, priority_meta = build_priority(combined, priority_mode, recommend_n)
        candidates = priority_df[priority_df["선정"]].copy()


# =========================================================
# 7. 군집 계산: 기존 KMeans 유지, Ward·GMM 비교 / PCA 없음
# =========================================================
cluster_result = None
cluster_error = None
cluster_metrics = None
cluster_results = {}
cluster_excluded = []
selected_model = None
feature_cols = ["연간소비량_MWh", "설비용량_kW"]
k = None
cluster_y = "설비용량_kW"

if combined is not None:
    st.sidebar.header("🧩 군집 설정")
    cluster_y = st.sidebar.selectbox(
        "군집 산점도 y축 (표시만 변경)",
        ["설비용량_kW", "설비건수"],
        format_func=lambda x: {
            "설비용량_kW": "태양광 설비용량 (kW)",
            "설비건수": "태양광 사업 건수 (건)",
        }[x],
    )
    st.sidebar.caption(
        "군집 기본 입력은 전력량과 설비용량입니다. 건수는 CSV의 유효 행 수이며 "
        "발전소 수와 다를 수 있습니다."
    )
    if st.sidebar.checkbox("군집에 여름(6~8월) 전력 비중 추가", value=False):
        feature_cols.append("여름사용비중_pct")
    sector_options = ["사용 안 함"] + [
        col for col in ["가정용비중_pct", "서비스업비중_pct"]
        if col in combined.columns and combined[col].notna().any()
    ]
    if len(sector_options) > 1:
        sector_choice = st.sidebar.selectbox(
            "군집에 업종 비중 1개 추가", sector_options,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
            key=f"sector_feature::{power_path.resolve()}",
        )
        if sector_choice != "사용 안 함":
            feature_cols.append(sector_choice)
    else:
        st.sidebar.caption("현재 전력 파일에는 사용할 업종별 비중이 없습니다.")

    try:
        cluster_frame = combined.loc[~combined["자료내등록없음"]].copy()
        cluster_frame = cluster_frame.replace([np.inf, -np.inf], np.nan)
        cluster_frame = cluster_frame.dropna(subset=feature_cols)
        cluster_excluded = sorted(set(combined["자치구"]) - set(cluster_frame["자치구"]))
        unique_count = len(np.unique(cluster_frame[feature_cols].to_numpy(dtype=float), axis=0))
        max_k = min(6, len(cluster_frame) - 1, unique_count)
        if max_k < 2:
            raise ValueError("군집 분석에 필요한 서로 다른 구별 값이 부족합니다.")
        k = int(st.sidebar.selectbox(
            "군집 수", list(range(2, max_k + 1)), index=1 if max_k >= 3 else 0,
        ))
        if k > 3:
            st.sidebar.caption("25개 구처럼 작은 표본은 2~3개 군집부터 비교하는 것이 좋습니다.")
        cluster_results, cluster_metrics = fit_cluster_models(
            cluster_frame, feature_cols, ["연간소비량_MWh", "설비용량_kW"], k
        )
        selected_model = st.sidebar.selectbox("표시할 군집 모델", list(cluster_results))
        cluster_result = cluster_results[selected_model]
    except Exception as exc:
        cluster_error = str(exc)


# =========================================================
# 8. 요약
# =========================================================
c1, c2, c3 = st.columns(3)
c1.metric("전력 분석 지역", f"{len(matrix)}개 구")
c2.metric("연간 전력 사용량 합계", f"{annual.sum():,.0f} MWh")
c3.metric("최대 사용 지역", demand_df.iloc[0]["자치구"])

missing_gu = sorted(SEOUL_GU - set(matrix.index))
if missing_gu:
    st.warning("전력 분석에서 제외된 구: " + ", ".join(missing_gu))

if solar_error:
    st.warning(
        f"태양광 분석 준비 필요: {solar_error} "
        "전력 막대그래프와 히트맵은 계속 확인할 수 있습니다."
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 원본·전처리",
    "📊 연간 사용량",
    "🔥 월별 히트맵",
    "🧩 군집 비교·산점도",
    "📝 추천·자료 비교",
])


# =========================================================
# 9. 원본·전처리
# =========================================================
with tab1:
    st.subheader("전력 사용량 원본")
    st.write(f"파일: **{power_path.name}**")
    st.write(
        f"파일 원본: {power_raw.shape[0]}행 × {power_raw.shape[1]}열"
    )
    show_table(power_raw, hide_index=True)

    st.markdown("#### 제목 행 적용 후 자료형·결측치")
    st.caption(
        "아래는 필터링 전 표 기준입니다. Excel의 혼합 자료형 때문에 "
        "숫자 컬럼도 object로 읽힐 수 있으며 분석 전 숫자로 변환합니다."
    )
    st.write(
        f"{power_original.shape[0]}행 × {power_original.shape[1]}열"
    )
    st.dataframe(
        describe_columns(power_original),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("태양광 원본")

    if solar_raw is not None:
        st.write(f"파일: **{solar_path.name}**")
        st.write(
            f"파일 원본: {solar_raw.shape[0]}행 × {solar_raw.shape[1]}열"
        )
        show_table(solar_raw, hide_index=True)

    if solar_original is not None:
        st.markdown("#### 제목 행 적용 후 자료형·결측치")
        st.dataframe(
            describe_columns(solar_original),
            use_container_width=True,
            hide_index=True,
        )

    if solar_raw is None:
        st.info("태양광 파일이 없어 원본을 표시하지 못했습니다.")

    st.divider()
    st.subheader("결측치·이상치 처리 전후")

    log_df = pd.DataFrame(power_logs + solar_logs)
    show_table(log_df, hide_index=True)

    st.caption(
        "전력 결측치는 0으로 대체하지 않고 해당 구를 제외합니다. "
        "태양광 용량 결측·주소 추출 실패 행은 제외합니다. "
        "통계적 이상치는 실제 규모 차이일 수 있어 자동 삭제하지 않습니다."
    )

    st.download_button(
        "전처리 로그 다운로드",
        csv_bytes(log_df),
        "preprocessing_log.csv",
        "text/csv",
        key="download_logs",
    )

    if power_data["sector_notes"]:
        for note in power_data["sector_notes"]:
            st.warning(note)
    st.markdown("#### 전처리 후 전력 사용량 (MWh)")
    st.dataframe(matrix, use_container_width=True)

    if combined is not None:
        st.markdown("#### 구별 전력·태양광 결합 데이터")
        show_table(combined, hide_index=True)
        st.caption(
            "자료내등록없음=True인 구는 설비용량을 빈값으로 유지합니다. "
            "실제 0으로 바꾸지 않고 군집과 후보 선정에서 제외합니다."
        )


# =========================================================
# 10. 연간 사용량 막대그래프
# =========================================================
with tab2:
    st.subheader("질문 1. 어느 구가 전기를 많이 사용하는가?")

    try:
        fig = px.bar(
            demand_df,
            x="자치구",
            y="연간소비량_MWh",
            color="연간소비량_MWh",
            color_continuous_scale="YlOrRd",
            title="2025년 서울 자치구별 연간 전력 사용량",
            labels={
                "자치구": "자치구",
                "연간소비량_MWh": "연간 전력 사용량 (MWh)",
            },
        )
        fig.update_layout(
            height=560,
            xaxis={
                "categoryorder": "array",
                "categoryarray": demand_df["자치구"].tolist(),
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("#### 사용량 상위 3개 구")
            st.dataframe(
                demand_df.head(3),
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.markdown("#### 사용량 하위 3개 구")
            st.dataframe(
                demand_df.tail(3).sort_values("연간소비량_MWh"),
                hide_index=True,
                use_container_width=True,
            )

        top = demand_df.iloc[0]
        bottom = demand_df.iloc[-1]
        ratio = top["연간소비량_MWh"] / bottom["연간소비량_MWh"]

        finding(
            f"{top['자치구']}의 사용량이 가장 많습니다.",
            f"{top['연간소비량_MWh']:,.0f} MWh이며, "
            f"{bottom['자치구']} "
            f"{bottom['연간소비량_MWh']:,.0f} MWh의 "
            f"{ratio:.2f}배입니다.",
        )

        st.caption(
            "총사용량에는 인구·건물 규모·업종 구성이 영향을 줍니다. "
            "사용량이 많다고 에너지 효율이 낮다고 판단할 수 없습니다."
        )

    except Exception as exc:
        st.error(f"막대그래프 생성 오류: {exc}")


# =========================================================
# 11. 월별 히트맵
# =========================================================
with tab3:
    st.subheader("질문 2. 어느 달에 전력 사용이 집중되는가?")

    heat_mode = st.radio(
        "히트맵 표시 기준",
        ["연간 대비 월별 비율(%)", "원본 사용량(MWh)"],
        horizontal=True,
    )

    try:
        use_share = heat_mode.startswith("연간")
        heat = shares if use_share else matrix
        heat = heat.reindex(demand_df["자치구"].tolist())
        color_label = "연간 대비 비율 (%)" if use_share else "사용량 (MWh)"

        # px.imshow 대신 Heatmap을 사용해 버전 호환 문제 최소화
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Heatmap(
            z=heat.to_numpy(dtype=float),
            x=MONTHS,
            y=heat.index.tolist(),
            colorscale="YlOrRd",
            colorbar={"title": color_label},
            hovertemplate=(
                "자치구: %{y}<br>월: %{x}<br>"
                + color_label
                + ": %{z:,.2f}<extra></extra>"
            ),
        ))
        fig.update_layout(
            title=f"2025년 자치구별 월별 전력 사용 패턴 — {color_label}",
            xaxis_title="월",
            yaxis_title="자치구",
            height=max(550, len(heat) * 25),
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "비율 모드는 각 구의 연간 사용량을 100%로 봅니다. "
            "매달 같은 양을 쓰면 월별 비중은 약 8.33%입니다."
        )

        peak_table = pd.DataFrame({
            "자치구": shares.index,
            "최대 사용 월": shares.idxmax(axis=1).to_numpy(),
            "최대 월 비중(%)": shares.max(axis=1).to_numpy(),
        }).sort_values("최대 월 비중(%)", ascending=False)

        st.markdown("#### 특정 한 달에 사용이 집중된 상위 5개 구")
        st.dataframe(
            peak_table.head(5).round(2),
            hide_index=True,
            use_container_width=True,
        )

        counts = shares.idxmax(axis=1).value_counts()
        highest_count = int(counts.max())
        common_months = counts[counts == highest_count].index.tolist()
        concentrated = peak_table.iloc[0]

        finding(
            f"최대 사용 월로 가장 자주 나타난 달은 "
            f"{', '.join(common_months)}입니다.",
            f"각 {highest_count}개 구에서 해당 월이 최대이며, "
            f"{concentrated['자치구']}는 "
            f"{concentrated['최대 사용 월']}에 연간 사용량의 "
            f"{concentrated['최대 월 비중(%)']:.2f}%를 사용했습니다.",
        )

        st.caption(
            "월별 차이의 원인은 이 그래프만으로 확정할 수 없습니다. "
            "월별 날짜 수, 기온, 업종 등의 영향이 포함됩니다."
        )

    except Exception as exc:
        st.error(f"히트맵 생성 오류: {exc}")


# =========================================================
# 12. 군집 산점도
# =========================================================
with tab4:
    st.subheader(
        "질문 3. 전력 사용량과 태양광 설비가 비슷한 구는 어디인가?"
    )

    if solar_error:
        st.info(f"태양광 데이터 설정을 확인하세요: {solar_error}")

    elif cluster_error:
        st.info(f"군집 분석 안내: {cluster_error}")

    elif cluster_result is not None:
        try:
            y_title = (
                "태양광 설비용량 (kW)"
                if cluster_y == "설비용량_kW"
                else "태양광 사업 건수 (건)"
            )

            st.markdown("#### 같은 입력으로 비교한 세 모델")
            st.dataframe(cluster_metrics.round(4), hide_index=True, use_container_width=True)
            st.caption(
                "실루엣 점수는 군집 분리도입니다. BIC는 같은 자료·입력 특성에서 GMM 설정끼리 비교하는 지표이며 "
                "다른 모델의 실루엣 점수와 직접 비교하지 않습니다. 설치 예측 정확도가 아닙니다."
            )
            if (cluster_metrics["최소 군집 구 수"].dropna() < 3).any():
                st.info("3개 구 미만의 작은 군집이 있습니다. 군집 수·입력 변경에 민감할 수 있습니다.")
            if cluster_excluded:
                st.warning("설비 또는 선택 특성 미확인으로 군집에서 제외: " + ", ".join(cluster_excluded))
            st.caption(
                f"표시 모델: {selected_model}, 군집 수: {k}. 입력: "
                + ", ".join(FEATURE_LABELS[c] for c in feature_cols)
                + ". 전력량·설비용량만 log1p 변환하고 모든 입력을 표준화합니다. PCA는 사용하지 않습니다."
            )
            if len(feature_cols) > 2:
                st.caption("점의 색은 추가 특성까지 반영한 결과입니다. 화면의 두 축만으로 만든 군집이 아닙니다.")
            if selected_model == "GMM":
                st.caption("군집소속확률은 선택한 GMM 안에서의 소속 정도이며 설치 성공확률이 아닙니다.")
            hover_fields = {
                "연간소비량_MWh": ":,.0f", "설비용량_kW": ":,.2f",
                "설비건수": True, "자료내등록없음": True,
            }
            if "군집소속확률" in cluster_result.columns:
                hover_fields["군집소속확률"] = ":.1%"

            # 원래 단위를 그대로 표시해 0도 정확히 표현
            fig = px.scatter(
                cluster_result,
                x="연간소비량_MWh",
                y=cluster_y,
                color="군집",
                text="자치구",
                hover_name="자치구",
                hover_data=hover_fields,
                title=f"{selected_model}: 서울 자치구별 전력 사용량 × {y_title}",
                labels={
                    "연간소비량_MWh": "연간 전력 사용량 (MWh)",
                    "설비용량_kW": "태양광 설비용량 (kW)",
                    "설비건수": "태양광 사업 건수 (건)",
                },
                category_orders={
                    "군집": [str(i) for i in range(1, k + 1)]
                },
            )
            fig.update_traces(
                marker={"size": 12},
                textposition="top center",
            )

            demand_median = float(
                cluster_result["연간소비량_MWh"].median()
            )
            solar_median = float(cluster_result[cluster_y].median())

            fig.add_vline(
                x=demand_median,
                line_dash="dot",
                line_color="gray",
            )
            fig.add_hline(
                y=solar_median,
                line_dash="dot",
                line_color="gray",
            )
            fig.update_layout(height=680)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"점선은 분석 대상 {len(cluster_result)}개 구의 중앙값입니다. "
                "축은 원래 단위이고, 군집 계산에만 log1p·표준화를 사용했습니다. "
                "군집 번호는 평균 전력 수요 순으로 정렬한 표시이며 설치 순위가 아닙니다. "
                "모델이 다르면 같은 번호라도 구성원이 다를 수 있습니다. 건수 축은 표시용으로만 사용합니다."
            )

            profile = (
                cluster_result.groupby("군집", as_index=False)
                .agg(
                    구수=("자치구", "size"),
                    평균전력사용량_MWh=("연간소비량_MWh", "mean"),
                    평균설비용량_kW=("설비용량_kW", "mean"),
                    평균사업건수=("설비건수", "mean"),
                    총전력사용량_MWh=("연간소비량_MWh", "sum"),
                    총설비용량_kW=("설비용량_kW", "sum"),
                )
            )
            profile["수요대비설비_kW_GWh"] = (
                profile["총설비용량_kW"]
                / (profile["총전력사용량_MWh"] / 1000)
            )

            st.markdown("#### 군집별 평균 특성")
            st.dataframe(
                profile.round(2),
                hide_index=True,
                use_container_width=True,
            )

            for cluster_id, group in cluster_result.groupby("군집"):
                st.write(
                    f"**군집 {cluster_id}:** "
                    + ", ".join(group["자치구"].tolist())
                )

            st.info(
                "군집은 비슷한 구를 묶는 분석입니다. 추가 설치 검토 후보는 "
                "추천·자료 비교 탭에서 별도의 공개된 기준으로 선정합니다."
            )
            st.download_button(
                "구별 군집 결과 다운로드",
                csv_bytes(cluster_result),
                f"seoul_{dict(KMeans='kmeans', GMM='gmm').get(selected_model, 'ward')}_cluster_result.csv",
                "text/csv",
                key="download_clusters",
            )

        except Exception as exc:
            st.error(f"군집 산점도 생성 오류: {exc}")


# =========================================================
# 13. 추천 기준·전력 자료 비교·한계
# =========================================================
with tab5:
    st.subheader("질문 4. 어느 구를 태양광 추가 설치 검토 후보로 볼 것인가?")
    if priority_df is not None and not priority_df.empty:
        st.write(f"**선택한 기준:** {priority_mode}")
        st.caption(
            f"선정 조건: {priority_meta['rule']}. 중앙값은 전력·태양광 정보가 "
            f"모두 확인된 {priority_meta['n']}개 구 기준입니다. "
            "비율 동률은 전력 수요가 큰 순, 그다음 자치구 이름순으로 표시합니다."
        )
        a, b, c = st.columns(3)
        a.metric("조건에 해당하는 후보", f"{len(candidates)}개 구")
        b.metric("수요 중앙값", f"{priority_meta['demand_median']:,.2f} MWh")
        c.metric("설비용량 중앙값", f"{priority_meta['solar_median']:,.3f} kW")
        if priority_meta["zero_median"] and priority_mode == PRIORITY_MODES[0]:
            st.info(
                "확인된 설비용량의 중앙값이 0이므로 기존 조건의 용량 기준을 "
                "'용량=0'으로 적용했습니다. 미등록 구는 여전히 제외합니다."
            )
        result_cols = [
            "자치구", "연간소비량_MWh", "설비용량_kW", "설비건수",
            "수요대비설비_kW_GWh", "수요중앙값차이_MWh", "용량중앙값차이_kW",
        ]
        if not candidates.empty:
            finding(
                "추가 조사 후보: " + ", ".join(candidates["자치구"]),
                priority_meta["rule"],
            )
            st.dataframe(candidates[result_cols].round(3), hide_index=True, use_container_width=True)
        else:
            st.info("선택한 조건을 만족하는 구가 없습니다. 이것이 설치 가능한 곳이 없다는 뜻은 아닙니다.")
        st.caption(
            "수요 대비 설비 = 설비용량(kW) / 연간 사용량(GWh). 작을수록 제공 자료에서 "
            "수요 대비 설비가 적게 집계됩니다."
        )
        if priority_mode == PRIORITY_MODES[0] and not priority_meta["zero_median"]:
            st.caption("기존 조건은 엄격한 <, > 비교이므로 중앙값과 같은 구는 해당 조건에서 제외됩니다.")
        with st.expander("전체 구의 비교값과 후보 선정 여부"):
            st.dataframe(
                priority_df[["비율순서"] + result_cols + ["선정", "선정근거"]].round(3),
                hide_index=True, use_container_width=True,
            )
        unknown_gu = combined.loc[combined["자료내등록없음"], "자치구"].tolist()
        if unknown_gu:
            st.warning("태양광 현황 미확인으로 추천에서 제외: " + ", ".join(unknown_gu))
        export_priority = priority_df.assign(
            전력자료=power_path.name, 전력원본단위=power_unit,
            태양광자료=solar_path.name, 추천기준=priority_mode,
        )
        st.download_button(
            "후보 선정 근거 다운로드", csv_bytes(export_priority),
            "seoul_solar_priority.csv", "text/csv", key="download_priority",
        )
        st.info(
            "이 결과는 제공 자료와 선택한 기준에 따른 추가 조사 우선순위입니다. "
            "실제 발전량·설치 가능 면적·비용을 평가한 최적 입지나 설치 성공확률은 아닙니다."
        )
    elif combined is not None:
        st.info("전력·태양광 정보가 함께 확인된 구가 없어 후보를 계산하지 않았습니다.")
    else:
        st.info("태양광 파일을 설정하면 구별 추가 조사 후보를 확인할 수 있습니다.")

    st.divider()
    st.subheader("전력 파일을 바꿔도 결과가 유지되는가?")
    source_rows = []
    source_candidate_sets = []
    for source_path, data in sorted(power_sources.items(), key=lambda item: item[0].name):
        common = annual.index.intersection(data["annual"].index)
        base_sum = float(annual.reindex(common).sum())
        compared_sum = float(data["annual"].reindex(common).sum())
        difference = compared_sum - base_sum if len(common) else np.nan
        row = {
            "전력파일": source_path.name, "현재선택": source_path == power_path,
            "원본단위": data["unit"], "분석구수": len(data["annual"]),
            "연간합계_MWh": float(data["annual"].sum()),
            "공통비교구수": len(common),
            "선택자료대비차이_MWh": difference,
            "선택자료대비차이_pct": difference / base_sum * 100 if base_sum > 0 else np.nan,
        }
        if solar_agg is not None and solar_error is None:
            other_combined = combine_demand_solar(data["annual"], solar_agg)
            other_priority, _ = build_priority(other_combined, priority_mode, recommend_n)
            selected_gu = other_priority.loc[other_priority["선정"], "자치구"].tolist()
            row["선택기준후보"] = ", ".join(selected_gu) or "없음"
            source_candidate_sets.append(set(selected_gu))
        source_rows.append(row)
    source_comparison = pd.DataFrame(source_rows)
    st.dataframe(source_comparison.round(3), hide_index=True, use_container_width=True)
    st.caption(
        "각 파일을 독립적으로 집계하며 서로 합산·평균하지 않습니다. 비교 차이는 공통 구만 "
        "사용하며 현재 선택한 전력 파일이 기준입니다. 단위는 파일별 선택값을 사용하고, "
        "아직 선택하지 않은 파일은 형식별 기본 단위를 사용합니다."
    )
    if len(power_sources) > 1:
        st.info(
            "전력 파일 간 차이는 집계 대상·작성 기준 차이일 수 있습니다. "
            "파일 내용만으로 어느 자료가 더 정확한지는 단정하지 않습니다."
        )
        with st.expander("전력 파일별 구별 연간 사용량 비교 (MWh)"):
            source_detail = pd.concat(
                {path.name: data["annual"] for path, data in
                 sorted(power_sources.items(), key=lambda item: item[0].name)}, axis=1
            )
            source_detail.index.name = "자치구"
            st.dataframe(source_detail.round(3), use_container_width=True)
        if len(source_candidate_sets) >= 2:
            common_candidates = sorted(set.intersection(*source_candidate_sets))
            st.write(
                f"**정상 처리된 {len(source_candidate_sets)}개 전력 파일의 공통 후보:** "
                + (", ".join(common_candidates) if common_candidates else "없음")
            )
            st.caption("같은 추천 기준·태양광 자료를 적용한 결과이며, 공통 후보도 설치 적합성이 검증된 것은 아닙니다.")
    else:
        st.caption("전력 파일이 두 개 이상 있으면 출처별 수요와 후보를 비교할 수 있습니다.")
    for error in source_errors:
        st.warning("비교에서 제외한 전력 파일: " + error)
    st.download_button(
        "전력 자료 비교 다운로드", csv_bytes(source_comparison),
        "seoul_power_source_comparison.csv", "text/csv", key="download_sources",
    )

    st.markdown("""
### 분석 방법
- 전력은 2025년 1~12월을 합산하고 MWh로 통일합니다.
- 월별 값이 불완전한 구는 연간 비교에서 제외하고 처리 로그를 남깁니다.
- 태양광은 선택한 파일·운영 상태의 유효 기록을 구별로 합산합니다.
- 자료기준일·설치연도를 이유로 태양광 기록을 자동 제외하지 않습니다.
- 태양광 미등록 구의 값은 빈값으로 유지하고 군집·추천에서 제외합니다.
- 큰 수요·대형 설비는 실제 특성일 수 있어 IQR 이상치라는 이유로 삭제하지 않습니다.
- 군집은 같은 입력·전처리로 KMeans, Ward, GMM을 비교합니다. PCA는 사용하지 않습니다.
- 기본 입력은 전력량·설비용량이며 계절·업종 비중은 선택적으로 추가합니다.
- 업종 비중은 선택한 전력 파일에서만 계산하며 '-'를 임의로 0으로 바꾸지 않습니다.
- 추천은 군집과 별개의 공개 조건입니다. 산점도 축·군집 모델을 바꿔도 추천 기준은 유지됩니다.

### 해석의 한계
- 최종 분석 단위는 서울 25개 구입니다. 시설 557건이나 월별 300셀이
  독립적인 설치 적합성 정답 557개·300개가 되는 것은 아닙니다.
- 세 모델은 지역 유형을 나누는 비지도 군집 모델입니다.
  실루엣·BIC·군집소속확률은 설치 성공률이나 발전량 예측 정확도가 아닙니다.
- 제공된 태양광 목록의 범위 밖에 있는 설비까지 조사한 결과는 아닙니다.
- 설비 건수는 유효 행 수이며 한 행이 한 발전소인지는 자료 정의를 확인해야 합니다.
- 설비용량(kW)은 발전량(kWh)이 아닙니다. 수요 대비 설비 지표는 전력 자급률이 아닙니다.
- 월별 사용량은 낮 시간 수요나 순간 최대전력을 직접 나타내지 않습니다.
- 구별 설비가 적다고 그 구에 전력 공급 부족이 발생한다는 의미는 아닙니다.
- 실제 건물·옥상 선정에는 설치 가능 면적, 일사량·그늘, 건물 구조, 계통 접속, 비용 등이 필요합니다.
""")
    st.download_button(
        "전처리된 월별 전력 데이터 다운로드",
        csv_bytes(matrix.reset_index()), "seoul_2025_monthly_MWh.csv",
        "text/csv", key="download_monthly",
    )




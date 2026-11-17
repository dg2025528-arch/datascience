# 서울 자치구 태양광 추가 설치 검토 시급도 — 회귀 앱
# 실행: python -m streamlit run main.py
# 설치: python -m pip install "streamlit>=1.67" "pandas>=2.0" "numpy>=1.24" "plotly>=5.20" "scikit-learn>=1.4"
# Excel 선택 시: python -m pip install openpyxl xlrd
# 기존 CSV/Excel 파일을 main.py 옆 또는 data/ 폴더에 두거나 화면에서 업로드합니다.
# 실제 시급도 정답이 없는 기본 모드는 공개된 기준 점수를 재현하는 모의 학습입니다.
# 별도 평가점수 CSV(자치구,시급도점수)를 제공하면 해당 점수로 학습합니다.
# 점수, 순위, 설비비율 및 기준 점수의 구성요소는 입력 특성에 넣지 않습니다.
# 검증 전처리는 학습 폴드 안에서만 학습합니다. 제공된 태양광 자료를 연도로 자동 제외하지 않습니다.

from pathlib import Path
from io import BytesIO
import re
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

MONTHS = [f"{i}월" for i in range(1, 13)]

SEOUL_GU = {
    "강남구", "강동구", "강북구", "강서구", "관악구",
    "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구",
    "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구",
}

FEATURE_LABELS = {
    "연간소비량_MWh": "연간 전력 사용량",
    "설비용량_kW": "태양광 설비용량",
    "여름사용비중_pct": "여름(6~8월) 전력 비중",
    "가정용비중_pct": "가정용부문 전력 비중",
    "서비스업비중_pct": "서비스업 전력 비중",
}

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
    st.dataframe(display, width="stretch", **kwargs)

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

"""서울 자치구의 연속형 설치 검토 시급도 점수·회귀 분석 엔진.

기준 점수는 연구자가 정한 정책 지표이며 관측된 설치 성과가 아니다.
이를 목표값으로 학습한 교차검증은 기준 재현 실험이다. 별도 평가점수를
사용해도 25개 안팎 표본의 탐색적 검증으로 해석한다. 모든 전처리 통계는
각 학습 폴드 안에서 추정하며, 동일한 5개 분할로 한 번만 비교한다.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR


MODEL_NAMES = ["다중 선형 회귀", "Ridge 회귀", "Lasso 회귀", "그래디언트 부스팅 회귀", "SVR"]
URGENCY_LOG_COLUMNS = {"연간소비량_MWh", "설비용량_kW"}
URGENCY_DEFAULT_PARAMS = {
    "ridge_alpha": 1.0,
    "lasso_alpha": 0.1,
    "gb_estimators": 100,
    "gb_learning_rate": 0.05,
    "gb_depth": 2,
    "gb_leaf": 3,
    "svr_c": 10.0,
    "svr_epsilon": 1.0,
    "svr_gamma": "scale",
    "svr_kernel": "rbf",
}


def make_policy_scores(frame, demand_ref_gwh=2000.0, intensity_ref=1.0, demand_weight=0.5):
    """0~100의 기준 시급도 점수를 고정 기준값으로 계산한다.

    D = 연간 사용량(GWh), I = 설비용량(kW) / D(kW/GWh)
    점수 = 100 * {w * D/(D+D_ref) + (1-w) * I_ref/(I+I_ref)}.
    기준값과 가중치는 조정 가능한 정책 가정이며 발전량/공급률이 아니다.
    수요는 양수여야 한다. 미확인·음수 용량은 0으로 채우지 않는다.
    다른 구를 추가해도 기존 행의 점수가 바뀌지 않는다.
    """
    if not (np.isfinite(demand_ref_gwh) and demand_ref_gwh > 0):
        raise ValueError("수요 기준값(GWh)은 양수여야 합니다.")
    if not (np.isfinite(intensity_ref) and intensity_ref > 0):
        raise ValueError("수요 대비 설비 기준값(kW/GWh)은 양수여야 합니다.")
    if not (np.isfinite(demand_weight) and 0 <= demand_weight <= 1):
        raise ValueError("수요 가중치는 0~1이어야 합니다.")
    required = ["연간소비량_MWh", "설비용량_kW"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("시급도 입력 컬럼이 없습니다: " + ", ".join(missing))

    result = frame.copy()
    demand = pd.to_numeric(frame[required[0]], errors="coerce").astype(float)
    capacity = pd.to_numeric(frame[required[1]], errors="coerce").astype(float)
    demand = demand.where(np.isfinite(demand) & (demand > 0))
    capacity = capacity.where(np.isfinite(capacity) & (capacity >= 0))
    result["연간소비량_GWh"] = demand / 1000.0
    result["설비용량_MW"] = capacity / 1000.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        intensity = capacity / result["연간소비량_GWh"]
    intensity = intensity.where(np.isfinite(intensity))
    result["수요대비설비_kW_GWh"] = intensity
    result["수요지수"] = result["연간소비량_GWh"] / (result["연간소비량_GWh"] + demand_ref_gwh)
    result["설비희소지수"] = intensity_ref / (intensity + intensity_ref)
    score = 100.0 * (
        demand_weight * result["수요지수"]
        + (1.0 - demand_weight) * result["설비희소지수"]
    )
    eligible = demand.notna() & capacity.notna() & intensity.notna()
    # 가중치가 0/1이어도 필수 자료 미확인 행에는 점수를 만들지 않는다.
    result["기준시급도점수"] = score.where(eligible).clip(0.0, 100.0)
    return result


def _urgency_feature_transform(values, log_indices=(), n_features=None):
    """기준 단위·열 순서를 유지하는, 학습 통계가 없는 로그 변환."""
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 2 or (n_features is not None and result.shape[1] != n_features):
        raise ValueError("전처리 입력의 특성 개수 또는 배열 형태가 올바르지 않습니다.")
    if not np.isfinite(result).all():
        raise ValueError("전처리 입력에는 결측값 또는 무한대가 없어야 합니다.")
    if log_indices:
        selected = result[:, list(log_indices)]
        if (selected < 0).any():
            raise ValueError("로그 변환 대상에는 음수를 사용할 수 없습니다.")
        result[:, list(log_indices)] = np.log1p(selected)
    return result


def build_preprocessor(feature_cols, use_log=True):
    """log → scale Pipeline. 출력 열 순서는 feature_cols와 같다."""
    names = list(feature_cols)
    if not names or len(names) != len(set(names)):
        raise ValueError("중복되지 않은 입력 특성을 하나 이상 선택하세요.")
    log_indices = tuple(index for index, name in enumerate(names) if use_log and name in URGENCY_LOG_COLUMNS)
    return Pipeline([
        ("log", FunctionTransformer(
            _urgency_feature_transform,
            kw_args={"log_indices": log_indices, "n_features": len(names)},
            validate=False,
        )),
        ("scale", StandardScaler()),
    ])


def _empty_urgency_predictions(index):
    return pd.DataFrame({"score": np.nan, "raw_score": np.nan}, index=index)


def _new_urgency_result(frame, feature_cols):
    return {
        "task": "regression", "status": "unavailable", "message": "", "model": None,
        "cv_name": "평가 없음", "n_labeled": 0, "feature_names": list(feature_cols),
        "metrics": {}, "raw_metrics": {}, "baseline_metrics": {},
        "metrics_scope": "OOF 점수를 0~100으로 제한한 후 계산; raw_metrics는 제한 전",
        "oof": _empty_urgency_predictions(frame.index),
        "full": _empty_urgency_predictions(frame.index),
        "clipping": {"oof_count": 0, "full_count": 0},
        "diagnostics": {},
    }


def _urgency_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rho = np.nan
    if len(y_true) >= 2 and np.unique(y_true).size > 1 and np.unique(y_pred).size > 1:
        try:
            # pandas의 Spearman 구현을 사용하며 추가 패키지를 요구하지 않는다.
            rho = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
        except (ImportError, ValueError, FloatingPointError):
            rho = np.nan
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if np.var(y_true) > 0 else np.nan,
        "Spearman": rho,
    }


def _fit_urgency_pipeline(pipeline, features, targets):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(features, targets)
    notices = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    return pipeline, notices


def _urgency_estimator_factories(params):
    return {
        "다중 선형 회귀": lambda: LinearRegression(),
        "Ridge 회귀": lambda: Ridge(alpha=params["ridge_alpha"]),
        "Lasso 회귀": lambda: Lasso(alpha=params["lasso_alpha"], max_iter=20000, selection="cyclic"),
        "그래디언트 부스팅 회귀": lambda: GradientBoostingRegressor(
            n_estimators=params["gb_estimators"], learning_rate=params["gb_learning_rate"],
            max_depth=params["gb_depth"], min_samples_leaf=params["gb_leaf"],
            random_state=42, loss="squared_error", n_iter_no_change=None,
        ),
        "SVR": lambda: SVR(
            C=params["svr_c"], epsilon=params["svr_epsilon"],
            gamma=params["svr_gamma"], kernel=params["svr_kernel"],
        ),
    }


def fit_urgency_models(frame, feature_cols, targets, use_log=True, params=None):
    """선택된 5개 회귀 모델을 같은 5개 폴드로 비교하고 최종 모델을 학습한다.

    targets는 frame.index에 대응하는 0~100 점수 Series 또는 같은 길이 배열이다.
    미평가 점수는 보간하지 않는다. 미평가 행은 학습·OOF에서 제외하지만 특성이
    유효하면 full에 예측한다. 표준화는 매 폴드의 학습 행으로만 수행한다.

    oof: 해당 행을 학습하지 않은 모델의 예측. full: 모든 평가자료를 학습한
    최종 모델의 예측. 성능은 oof로만 계산한다. 두 표는 score와 raw_score를
    제공하며 score=clip(raw_score, 0, 100)이다. metrics는 제한 후 OOF 성능,
    raw_metrics는 제한 전 OOF 성능이다. 순위 상관은 Spearman 계수다.

    최소 8개 평가점수 조건은 이 앱의 실행 기준이며 통계적 충분성을 뜻하지
    않는다. 자동 튜닝·반복 검증·가장 좋은 결과만 고르는 절차를 수행하지 않는다.
    """
    names = list(feature_cols)
    results = {name: _new_urgency_result(frame, names) for name in MODEL_NAMES}
    try:
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError("분석 행 인덱스 또는 컬럼이 중복되어 있습니다. 구별로 한 행씩 정리하세요.")
        if not names or len(names) != len(set(names)):
            raise ValueError("중복되지 않은 입력 특성을 하나 이상 선택하세요.")
        forbidden = ("시급", "점수", "순위", "정답", "목표", "평가", "수요지수", "설비희소", "수요대비설비", "score", "target", "rank", "label")
        if any(any(token in str(name).lower().replace(" ", "") for token in forbidden) for name in names):
            raise ValueError("목표 점수·그 구성 지수·순위·평가 결과는 모델 입력에 포함할 수 없습니다.")
        if any(name not in frame.columns for name in names):
            raise ValueError("선택한 입력 특성 중 자료에 없는 컬럼이 있습니다.")
        features = frame[names].apply(pd.to_numeric, errors="coerce")
        finite_features = pd.Series(
            np.isfinite(features.to_numpy(dtype=float, na_value=np.nan)).all(axis=1), index=frame.index,
        )
        if "연간소비량_MWh" in names:
            finite_features &= features["연간소비량_MWh"] > 0
        if "설비용량_kW" in names:
            finite_features &= features["설비용량_kW"] >= 0
        if isinstance(targets, pd.Series):
            if not targets.index.is_unique:
                raise ValueError("평가점수의 인덱스가 중복되어 있습니다.")
            raw_targets = targets.reindex(frame.index)
        else:
            if len(targets) != len(frame):
                raise ValueError("평가점수의 개수가 분석 행 수와 다릅니다.")
            raw_targets = pd.Series(targets, index=frame.index)
        y = pd.to_numeric(raw_targets, errors="coerce").astype(float)
        malformed = raw_targets.notna() & y.isna()
        outside = y.notna() & (~np.isfinite(y) | ~y.between(0, 100))
        if malformed.any() or outside.any():
            raise ValueError("평가점수는 0~100의 유한한 숫자 또는 미평가 빈칸이어야 합니다.")
        config = URGENCY_DEFAULT_PARAMS.copy()
        if params is not None:
            unknown = set(params) - set(config)
            if unknown:
                raise ValueError("지원하지 않는 모델 설정: " + ", ".join(sorted(unknown)))
            config.update(params)
        labeled = finite_features & y.notna()
        labeled_index = frame.index[labeled]
        prediction_index = frame.index[finite_features]
        X_train = features.loc[labeled_index]
        y_train = y.loc[labeled_index]
        n_labeled = len(labeled_index)
        diagnostics = {
            "n_rows": len(frame), "n_valid_features": int(finite_features.sum()),
            "n_invalid_features": int((~finite_features).sum()),
            "n_unlabeled_valid_features": int((finite_features & y.isna()).sum()),
        }
        for result in results.values():
            result["n_labeled"] = n_labeled
            result["diagnostics"] = diagnostics.copy()
        if n_labeled < 8:
            for result in results.values():
                result["message"] = (
                    f"유효 입력과 평가점수가 함께 있는 구가 {n_labeled}개입니다. "
                    "이 앱은 최소 8개부터 학습합니다. 이 조건이 통계적 충분성을 보장하지는 않습니다."
                )
            return results
        splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
        baseline_pred = np.full(n_labeled, np.nan)
        for train_positions, test_positions in splits:
            baseline = DummyRegressor(strategy="mean").fit(
                X_train.iloc[train_positions], y_train.iloc[train_positions],
            )
            baseline_pred[test_positions] = baseline.predict(X_train.iloc[test_positions])
        baseline_metrics = _urgency_regression_metrics(y_train, baseline_pred)
    except Exception as exc:
        for result in results.values():
            result["status"] = "error"
            result["message"] = f"입력 확인 필요: {exc}"
        return results

    factories = _urgency_estimator_factories(config)
    for name in MODEL_NAMES:
        result = results[name]
        notices = []
        try:
            if name == "Ridge 회귀" and not (np.isfinite(config["ridge_alpha"]) and config["ridge_alpha"] > 0):
                raise ValueError("Ridge alpha는 양수여야 합니다.")
            if name == "Lasso 회귀" and not (np.isfinite(config["lasso_alpha"]) and config["lasso_alpha"] > 0):
                raise ValueError("Lasso alpha는 양수여야 합니다.")
            prototype = Pipeline([
                ("preprocess", build_preprocessor(names, use_log=use_log)),
                ("estimator", factories[name]()),
            ])
            raw_oof = np.full(n_labeled, np.nan)
            for train_positions, test_positions in splits:
                fold_model, fold_notices = _fit_urgency_pipeline(
                    clone(prototype), X_train.iloc[train_positions], y_train.iloc[train_positions],
                )
                notices.extend(fold_notices)
                prediction = np.asarray(fold_model.predict(X_train.iloc[test_positions]), dtype=float)
                if not np.isfinite(prediction).all():
                    raise ValueError("교차검증에서 유효하지 않은 점수를 반환했습니다.")
                raw_oof[test_positions] = prediction
            oof_pred = np.clip(raw_oof, 0.0, 100.0)
            result["oof"].loc[labeled_index, "raw_score"] = raw_oof
            result["oof"].loc[labeled_index, "score"] = oof_pred
            result["metrics"] = _urgency_regression_metrics(y_train, oof_pred)
            result["raw_metrics"] = _urgency_regression_metrics(y_train, raw_oof)
            result["baseline_metrics"] = baseline_metrics.copy()
            result["clipping"]["oof_count"] = int(((raw_oof < 0) | (raw_oof > 100)).sum())
            result["cv_name"] = "동일한 5겹 교차검증(KFold, seed=42)"

            final_model, final_notices = _fit_urgency_pipeline(clone(prototype), X_train, y_train)
            notices.extend(final_notices)
            raw_full = np.asarray(final_model.predict(features.loc[prediction_index]), dtype=float)
            if not np.isfinite(raw_full).all():
                raise ValueError("최종 모델에서 유효하지 않은 점수를 반환했습니다.")
            result["full"].loc[prediction_index, "raw_score"] = raw_full
            result["full"].loc[prediction_index, "score"] = np.clip(raw_full, 0.0, 100.0)
            result["clipping"]["full_count"] = int(((raw_full < 0) | (raw_full > 100)).sum())
            result["model"] = final_model
            result["status"] = "limited" if notices else "ok"
            result["message"] = (
                "같은 5개 분할에서 학습 폴드의 평균 점수를 내는 기준선과 비교했습니다. "
                "MAE·RMSE·R²·Spearman은 0~100 제한 후 OOF 예측으로 계산합니다. "
                "원 예측은 raw_score, 제한 전 지표는 raw_metrics에서 확인할 수 있습니다."
            )
            if np.var(y_train.to_numpy()) == 0:
                result["message"] += " 평가점수가 모두 같아 R²와 순위 상관은 정의하지 않았습니다."
            if notices:
                result["message"] += " 수렴 경고: " + " / ".join(dict.fromkeys(notices))
        except Exception as exc:
            result["status"] = "error"
            result["message"] = f"모델 계산을 완료하지 못했습니다: {exc}"
            result["model"] = None
            result["metrics"] = {}
            result["raw_metrics"] = {}
            result["baseline_metrics"] = {}
            result["oof"] = _empty_urgency_predictions(frame.index)
            result["full"] = _empty_urgency_predictions(frame.index)
            result["clipping"] = {"oof_count": 0, "full_count": 0}
    return results


"""각 회귀 모델 탭에 표시할 설명. 최종 main.py에 병합되는 순수 데이터입니다."""

MODEL_EXPLANATIONS = {
    "다중 선형 회귀": {
        "easy": (
            "가상의 예시로, 전력 수요가 커질수록 점수를 더하고 기존 설비가 많을수록 "
            "점수를 빼는 계산표를 생각하면 됩니다. 다만 더하고 빼는 양은 사람이 미리 "
            "정하지 않고 학습 자료에서 찾습니다. 여러 특성에 각각 가중치를 곱한 뒤 "
            "기본값을 더해 한 구의 시급도 점수를 계산합니다. 실제 가중치와 방향은 "
            "선택한 입력·학습 대상에 따라 달라집니다."
        ),
        "principle": (
            "여러 입력 특성의 선형 결합과 절편으로 점수를 표현하고, 학습 대상 점수와 "
            "예측 점수 사이의 오차 제곱합이 작아지는 계수를 구합니다. 이것을 최소제곱법이라고 "
            "합니다. 선형이라는 말은 전처리한 특성과 점수의 관계를 가중합으로 표현한다는 뜻입니다. "
            "계수와 절편은 학습 결과인 모델 파라미터입니다."
        ),
        "why": (
            "25개 구와 소수의 수치 특성으로 시작하므로, 계산 구조를 설명하기 쉬운 기준 모델이 "
            "필요합니다. 더 복잡한 모델이 이 단순한 관계보다 검증 오차를 줄이는지 비교할 수 "
            "있습니다. 선정 이유는 해석 가능성과 비교 기준으로서의 역할이며, 가장 우수하다고 "
            "미리 판단한 것은 아닙니다."
        ),
        "predict_steps": [
            "새 구의 입력을 학습 때 사용한 변수와 기준 단위로 준비하고, 동일한 전처리를 적용합니다.",
            "변환된 각 특성에 학습한 계수를 곱하고, 모든 값을 합한 뒤 절편을 더합니다.",
            "계산한 연속 점수를 출력합니다. 이 점수는 선택한 학습 대상을 추정한 값입니다.",
        ],
        "limits": [
            "특성 사이의 복잡한 상호작용이나 굽은 관계를 자동으로 충분히 표현하지 못합니다.",
            "표본이 적거나 비슷한 특성을 함께 쓰면 계수가 흔들릴 수 있으며, 계수는 인과효과가 아닙니다.",
            "기준 점수를 학습한 결과만으로 실제 설치의 필요성이나 사업 성과가 검증되지는 않습니다.",
        ],
        "parameters": [
            ("fit_intercept", True, "절편을 함께 학습할지 정합니다. True이면 입력의 가중합에 더할 기본값을 학습합니다."),
        ],
        "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
    },
    "Ridge 회귀": {
        "easy": (
            "가상의 예시로, 몇 개 구에 점수를 잘 맞추려다 어떤 특성의 가중치를 지나치게 크게 "
            "잡으면 다른 구에서는 예측이 크게 흔들릴 수 있습니다. Ridge는 선형 계산표를 "
            "만들면서 가중치가 과도하게 커지는 데 제약을 둡니다. 작은 차이를 모두 따라가기보다 "
            "조금 더 완만한 계산식을 찾도록 조절하는 방법입니다."
        ),
        "principle": (
            "선형 회귀의 오차 제곱합에 계수 제곱합에 대한 벌점을 더해 최소화합니다. "
            "이 벌점을 L2 규제라고 하며, alpha가 규제 강도를 정합니다. alpha가 커질수록 "
            "계수를 작게 만드는 압력이 커집니다. 계수를 대체로 남겨 두면서 크기를 줄이는 "
            "방식이며, 변수별 스케일을 맞추면 규제를 더 일관되게 적용할 수 있습니다."
        ),
        "why": (
            "서울 25개 구처럼 표본이 적을 때 단순 선형 회귀의 계수가 과도하게 커지는 현상을 "
            "완화하는 비교 모델로 사용합니다. 계절·업종 비중처럼 관련된 특성을 추가했을 때도 "
            "규제의 효과를 살펴볼 수 있습니다. 현재 alpha는 시작 설정이며 검증으로 찾은 "
            "최적값이나 성능 보장을 뜻하지 않습니다."
        ),
        "predict_steps": [
            "새 입력에 학습 당시의 변수 선택·변환·표준화 기준을 그대로 적용합니다.",
            "규제를 적용해 학습한 계수와 절편으로 특성의 가중합을 계산합니다.",
            "연속 시급도 점수를 출력하고, 직접 계산한 기준 점수 또는 별도 평가점수와 구분해 해석합니다.",
        ],
        "limits": [
            "alpha를 너무 크게 두면 필요한 차이까지 줄여 과소적합할 수 있습니다.",
            "기본 관계는 여전히 선형이며, 규제만으로 부족한 표본이나 누락된 설치 조건을 해결할 수 없습니다.",
            "계수 크기는 전처리와 함께 해석해야 하며 실제 정책적 중요도와 같지 않습니다.",
        ],
        "parameters": [
            ("alpha", "ridge_alpha", "계수 제곱합에 부과하는 벌점의 강도입니다. 커질수록 계수를 더 작게 만드는 방향으로 학습합니다."),
            ("fit_intercept", True, "각 특성의 계수와 함께 절편을 학습합니다."),
        ],
        "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html",
    },
    "Lasso 회귀": {
        "easy": (
            "가상의 예시로, 전력 수요·기존 설비·여름 사용 비중을 넣었는데 마지막 특성을 "
            "추가해도 점수 설명이 별로 나아지지 않는다고 생각해 봅시다. Lasso는 어떤 "
            "특성의 가중치를 정확히 0으로 만들 수 있습니다. 여러 특성을 넣은 계산표에서 "
            "일부 항목을 빼는 효과가 있지만, 실제로 어떤 항목이 빠질지는 학습 자료와 설정에 달려 있습니다."
        ),
        "principle": (
            "선형 예측의 오차와 계수 절댓값 합에 대한 벌점을 함께 줄입니다. "
            "이 방식을 L1 규제라고 합니다. 규제를 통해 계수를 줄이며 일부 계수는 0이 될 수 "
            "있습니다. alpha가 클수록 규제가 강해지고, max_iter는 계수를 찾는 반복 계산의 "
            "최대 횟수를 정합니다. 계수 자체는 하이퍼파라미터가 아니라 학습 결과입니다."
        ),
        "why": (
            "계절·업종 특성을 선택적으로 추가했을 때 더 단순한 입력 조합으로 점수를 설명할 수 "
            "있는지 비교하려고 사용합니다. 기본 특성이 두 개뿐이면 변수 선택의 이점은 제한적입니다. "
            "적은 표본에서 항상 안정적인 변수를 골라 준다고 가정하지 않고, Ridge와 일반 선형 회귀를 "
            "함께 비교하는 용도로 둡니다."
        ),
        "predict_steps": [
            "새 구의 자료를 학습 때의 입력 순서와 동일한 전처리 기준으로 변환합니다.",
            "학습한 계수와 절편으로 점수를 계산합니다. 계수가 0인 특성은 해당 계산에 기여하지 않습니다.",
            "연속 점수를 출력하고, 어떤 계수가 남았는지와 검증 오차를 함께 확인합니다.",
        ],
        "limits": [
            "서로 비슷한 특성이 있으면 어느 계수를 0으로 만드는지가 자료 변화에 민감할 수 있습니다.",
            "alpha가 크면 모든 계수가 거의 사라져 구별 차이를 충분히 표현하지 못할 수 있습니다.",
            "선택된 특성이 실제 시급도의 원인이라는 뜻은 아니며, 기준 점수 재현 결과를 현장 검증으로 해석할 수 없습니다.",
        ],
        "parameters": [
            ("alpha", "lasso_alpha", "계수 절댓값 합에 대한 규제 강도입니다. 커질수록 더 많은 계수가 작아지거나 0이 될 수 있습니다."),
            ("max_iter", 20000, "계수를 찾는 반복 계산의 최대 횟수입니다. 수렴하면 그전에 멈춥니다."),
            ("fit_intercept", True, "입력의 가중합에 더할 절편을 함께 학습합니다."),
        ],
        "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html",
    },
    "그래디언트 부스팅 회귀": {
        "easy": (
            "가상의 예시로, 첫 계산에서 한 구를 50점으로 예상했는데 학습 대상은 65점이었다고 "
            "해 봅시다. 다음 작은 나무는 이런 부족분을 보완하는 규칙을 배웁니다. 이후 나무도 "
            "앞선 예측에서 남은 오차를 조금씩 보완합니다. 여러 나무의 보완값을 순서대로 더해 "
            "최종 점수를 만드는 방식입니다. 예시의 숫자는 실제 분석 결과가 아닙니다."
        ),
        "principle": (
            "기본 예측에서 시작해 손실함수를 줄이는 방향으로 회귀나무를 순차적으로 추가합니다. "
            "현재 제곱오차 손실에서는 다음 나무가 앞선 예측의 잔차를 학습하는 것으로 이해할 수 "
            "있습니다. 각 나무의 출력을 learning_rate만큼 반영해 더하며, 특성의 비선형 관계와 "
            "상호작용을 표현할 수 있습니다."
        ),
        "why": (
            "수요와 설비가 점수에 미치는 관계를 하나의 직선으로 설명하기 어려울 가능성을 "
            "살펴보는 비교 모델입니다. 표본은 25개 구이므로 나무 깊이와 말단 노드의 최소 구 수를 "
            "제한해 복잡도를 낮춥니다. 비선형 모델이라는 이유만으로 선형 모델보다 우수하다고 "
            "판단하지 않고 동일한 검증 구에서 오차를 비교합니다."
        ),
        "predict_steps": [
            "새 구의 특성을 학습 당시와 동일한 방식으로 준비하고 전처리합니다.",
            "학습된 각 나무의 조건을 따라가서 보완값을 얻습니다.",
            "기본 예측에 나무별 보완값을 학습률만큼 반영해 더하고 연속 점수를 출력합니다.",
        ],
        "limits": [
            "나무 수·깊이가 커지면 적은 구의 특성을 외워 새 자료에서 오차가 커질 수 있습니다.",
            "단일 선형 계수처럼 결과를 간단하게 설명하기 어렵고, 학습 범위를 벗어난 상황을 추정하는 데 한계가 있습니다.",
            "기준 점수 재현 모드에서 낮은 오차는 지정한 계산 규칙을 잘 따라갔다는 의미입니다.",
        ],
        "parameters": [
            ("n_estimators", "gb_estimators", "순차적으로 추가할 회귀나무의 수입니다."),
            ("learning_rate", "gb_learning_rate", "각 나무의 보완값을 최종 예측에 얼마나 반영할지 정합니다."),
            ("max_depth", "gb_depth", "각 나무가 조건을 나눌 수 있는 최대 깊이입니다."),
            ("min_samples_leaf", "gb_leaf", "각 말단 노드에 남아야 하는 최소 학습 구 수입니다."),
            ("loss", "squared_error", "학습 중 줄일 오차의 형태입니다. 현재는 제곱오차를 사용합니다."),
            ("random_state", 42, "나무 학습의 무작위 과정을 재현하기 위한 값이며, 성능 점수가 아닙니다."),
        ],
        "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html",
    },
    "SVR": {
        "easy": (
            "가상의 예시로, 학습 대상이 60점일 때 59.5점이나 60.5점 정도의 작은 오차는 "
            "허용하고 전체 관계를 매끄럽게 설명하는 곡선을 찾는다고 생각하면 됩니다. "
            "허용 폭을 벗어난 오차에는 벌점을 줍니다. RBF 커널을 선택하면 특성이 가까운 "
            "구들 사이의 유사성을 이용해 굽은 관계를 표현합니다. 예시 숫자는 실제 구의 점수가 아닙니다."
        ),
        "principle": (
            "SVR은 함수의 복잡도를 제한하면서 epsilon 허용 폭 바깥의 오차를 줄이는 회귀 방법입니다. "
            "C가 복잡도와 오차 벌점의 균형을 정합니다. 커널은 특성 사이의 관계를 계산하는 "
            "방식이며, RBF 커널에서는 gamma가 각 학습점의 영향 범위를 조절합니다. "
            "예측 함수는 학습에서 선택된 서포트 벡터와 새 입력 사이의 커널 값을 이용합니다."
        ),
        "why": (
            "작은 수치형 자료에서 규제를 적용하면서 매끄러운 비선형 관계를 비교할 수 있어 "
            "선정했습니다. 변수별 숫자 규모의 영향을 줄이도록 학습 구 기준의 전처리를 적용합니다. "
            "25개 구에서 설정에 따른 변화가 클 수 있으므로, 복잡한 모델의 우수성을 전제하지 "
            "않고 선형 모델과 같은 검증 방식으로 확인합니다."
        ),
        "predict_steps": [
            "새 입력을 학습 때의 변수·단위·전처리 기준에 맞춥니다.",
            "새 입력과 학습한 서포트 벡터 사이의 선택한 커널 값을 계산합니다.",
            "학습된 가중치와 절편으로 커널 값을 합쳐 연속 시급도 점수를 출력합니다.",
        ],
        "limits": [
            "C·epsilon·gamma 설정에 민감하며, 현재 기본값이 최적이라고 검증된 것은 아닙니다.",
            "예측 근거를 특성별 단일 계수로 설명하기 어렵고, 학습 범위를 벗어난 상황의 추정에는 주의가 필요합니다.",
            "출력은 학습 대상에 대한 점수 추정이며 설치 성공확률이나 신뢰구간이 아닙니다.",
        ],
        "parameters": [
            ("kernel", "svr_kernel", "관계를 계산하는 커널입니다. rbf는 비선형 관계를 표현합니다."),
            ("C", "svr_c", "허용 폭 밖 오차에 대한 벌점 강도입니다. 커질수록 학습 오차를 줄이는 데 더 비중을 둡니다."),
            ("epsilon", "svr_epsilon", "벌점을 주지 않는 오차의 허용 폭입니다. 목표가 0~100점이면 같은 점수 단위입니다."),
            ("gamma", "svr_gamma", "RBF 영향 범위의 계수입니다. scale은 학습 입력의 특성 수와 분산으로 자동 계산합니다."),
        ],
        "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",
    },
}


def prepare_solar(raw, header_row, addr_col, cap_col, cap_unit="kW",
                  status_col=None, selected_states=None, allow_zero=False,
                  drop_exact_duplicates=False):
    """부분 결측 구의 설비 합계가 과소 추정되지 않도록 확인 상태도 반환합니다."""
    original = make_table(raw, header_row)
    work = original.dropna(how="all").copy()
    logs = []
    add_log(logs, "태양광", "제목 행 적용·빈 행 제거", len(original), len(work))
    if addr_col not in work or cap_col not in work:
        raise ValueError("태양광 주소·용량 컬럼을 확인하세요.")
    if status_col and status_col != "사용 안 함":
        before = len(work)
        work = work[work[status_col].astype(str).isin(selected_states or [])].copy()
        add_log(logs, "태양광", "운영 상태 선택", before, len(work), ", ".join(selected_states or []))
    if work.empty:
        raise ValueError("선택한 운영 상태에 해당하는 태양광 기록이 없습니다.")
    duplicate_count = int(work.duplicated().sum())
    duplicate_gu = set()
    if duplicate_count and not drop_exact_duplicates:
        duplicate_gu = set(work.loc[work.duplicated(keep=False), addr_col].map(extract_gu).dropna())
    before = len(work)
    if drop_exact_duplicates:
        work = work.drop_duplicates().copy()
    add_log(logs, "태양광", "완전 동일 행 확인", before, len(work),
            f"중복 {duplicate_count}건. " + ("사용자 선택으로 완전 동일 행만 제거" if drop_exact_duplicates
                                      else "중복이 있는 구는 확인 전 시급도 산정 제외"))
    work["자치구"] = work[addr_col].map(extract_gu)
    work["용량_kW"] = to_number(work[cap_col]) * {"kW": 1., "MW": 1000., "W": .001}[cap_unit]
    unknown_address = int(work["자치구"].isna().sum())
    before = len(work)
    located = work.dropna(subset=["자치구"]).copy()
    add_log(logs, "태양광", "서울 자치구 확인", before, len(located),
            f"주소 확인 실패 {unknown_address}건. 해당 행은 구별 합계에 포함하지 않음")
    seen_gu = set(located["자치구"])
    invalid = located["용량_kW"].isna() | (located["용량_kW"] < 0)
    if not allow_zero:
        invalid |= located["용량_kW"].eq(0)
    incomplete_gu = set(located.loc[invalid, "자치구"]) | duplicate_gu
    invalid_rows = located.loc[invalid | located["자치구"].isin(duplicate_gu)].copy()
    clean = located.loc[~invalid].copy()
    add_log(logs, "태양광", "용량 확인", len(located), len(clean),
            f"결측·음수·미확인 0 등 {int(invalid.sum())}건. 일부 행이 빠진 구는 합계도 미확인 처리")
    if clean.empty:
        raise ValueError("유효한 태양광 용량이 없습니다. 제목 행·컬럼·원본 단위를 확인하세요.")
    lo, hi = iqr_limits(clean["용량_kW"])
    extremes = int(((clean["용량_kW"] < lo) | (clean["용량_kW"] > hi)).sum())
    add_log(logs, "태양광", "큰 설비 점검", len(clean), len(clean),
            f"IQR 기준 후보 {extremes}건. 실제 대형 설비일 수 있어 유지")
    agg = clean.groupby("자치구", as_index=False).agg(
        설비건수=("용량_kW", "size"), 설비용량_kW=("용량_kW", "sum"))
    agg["확인된부분합_kW"] = agg["설비용량_kW"]
    agg.loc[agg["자치구"].isin(incomplete_gu), "설비용량_kW"] = np.nan
    dates = []
    for col in original:
        if "기준일" in col:
            dates.extend(original[col].dropna().astype(str).unique().tolist())
    return {"original": original, "clean": clean, "agg": agg, "logs": logs,
            "seen_gu": seen_gu, "incomplete_gu": incomplete_gu, "invalid_rows": invalid_rows,
            "unknown_address": unknown_address, "duplicate_count": duplicate_count,
            "dates": sorted(set(dates))}


def combine_with_quality(power_data, solar_data):
    result = combine_demand_solar(power_data["annual"], solar_data["agg"])
    result["자료내등록없음"] = ~result["자치구"].isin(solar_data["seen_gu"])
    result["설비자료불완전"] = result["자치구"].isin(solar_data["incomplete_gu"])
    result["설비확인상태"] = np.select(
        [result["자료내등록없음"], result["설비자료불완전"]],
        ["자료 내 등록 없음·현황 미확인", "일부 기록 확인 필요"], default="등록 자료 확인")
    result["여름사용비중_pct"] = result["자치구"].map(
        power_data["shares"][["6월", "7월", "8월"]].sum(axis=1))
    sectors = power_data["sector_features"]
    if len(sectors.columns):
        result = result.merge(sectors.reset_index(), on="자치구", how="left", validate="one_to_one")
    return result


def read_evaluation_labels(content, suffix):
    raw = read_file(content, suffix)
    header = None
    for i in range(min(20, len(raw))):
        names = {normalize(x) for x in raw.iloc[i]}
        if "시급도점수" in names and ("자치구" in names or "자치구명" in names):
            header = i
            break
    if header is None:
        raise ValueError("평가 파일에 '자치구', '시급도점수' 두 열이 필요합니다.")
    labels = make_table(raw, header).dropna(how="all").copy()
    if "자치구" not in labels:
        labels = labels.rename(columns={"자치구명": "자치구"})
    labels["자치구"] = labels["자치구"].map(normalize)
    unknown = labels.loc[~labels["자치구"].isin(SEOUL_GU), "자치구"].tolist()
    if unknown:
        raise ValueError("평가 파일의 자치구 이름을 확인하세요: " + ", ".join(unknown))
    if labels["자치구"].duplicated().any():
        raise ValueError("평가 파일에 같은 자치구가 두 번 있습니다. 구별 점수는 한 개여야 합니다.")
    raw_scores = labels["시급도점수"]
    scores = to_number(raw_scores)
    bad = (raw_scores.notna() & scores.isna()) | (scores.notna() & ~scores.between(0, 100))
    if bad.any():
        raise ValueError("시급도점수는 0~100의 숫자 또는 빈칸이어야 합니다: "
                         + ", ".join(labels.loc[bad, "자치구"]))
    labels["시급도점수"] = scores
    if not scores.notna().any():
        raise ValueError("작성된 평가점수가 없습니다. 양식의 빈칸에 별도로 평가한 점수를 입력하세요.")
    return labels[["자치구", "시급도점수"]]


def format_axes(fig, x_format=",.0f", y_format=",.2f"):
    fig.update_xaxes(tickformat=x_format, exponentformat="none", showexponent="none")
    fig.update_yaxes(tickformat=y_format, exponentformat="none", showexponent="none")
    fig.update_layout(template="plotly_white", margin=dict(l=25, r=25, t=60, b=45))
    return fig


@st.cache_data(show_spinner=False)
def cached_models(frame, feature_cols, targets, use_log, params):
    return fit_urgency_models(frame, feature_cols, targets, use_log=use_log, params=params)


def score_plot(frame, score_column, title, key, coords=None, show_names=False):
    shown = frame.copy()
    if coords is None:
        x, y = "연간소비량_GWh", "설비용량_MW"
        labels = {x: "연간 전력 사용량 (GWh)", y: "태양광 설비용량 (MW)"}
    else:
        shown["전력량_표준점수"] = coords[:, 0]
        shown["설비용량_표준점수"] = coords[:, 1]
        x, y = "전력량_표준점수", "설비용량_표준점수"
        labels = {x: "전력량 · 전처리 후 표준점수", y: "설비용량 · 전처리 후 표준점수"}
    shown = shown.dropna(subset=[x, y, score_column])
    if shown.empty:
        st.info("표시할 점수가 없습니다. 입력 자료와 평가점수 수를 확인하세요.")
        return
    fig = px.scatter(shown, x=x, y=y, color=score_column, range_color=[0, 100],
                     color_continuous_scale="YlOrRd", hover_name="자치구",
                     text="자치구" if show_names else None,
                     hover_data={"연간소비량_GWh": ":,.2f", "설비용량_MW": ":,.3f",
                                 score_column: ":.2f"}, labels=labels, title=title)
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")), textposition="top center")
    fig.update_layout(height=490)
    format_axes(fig, ",.2f" if coords is not None else ",.0f", ",.2f")
    st.plotly_chart(fig, width="stretch", key=key)


def explain_model(name, params):
    info = MODEL_EXPLANATIONS[name]
    st.subheader(name)
    st.markdown("#### 쉬운 예시")
    st.write(info["easy"])
    st.markdown("#### 작동 원리와 핵심 아이디어")
    st.write(info["principle"])
    st.markdown("#### 이 프로젝트에서 비교하는 이유")
    st.write(info["why"])
    with st.expander("새 자료의 점수를 계산하는 순서 · 하이퍼파라미터", expanded=False):
        for i, step in enumerate(info["predict_steps"], 1):
            st.write(f"{i}. {step}")
        st.caption("하이퍼파라미터는 학습 전에 정하는 설정이며, 데이터에서 배우는 계수·분기 규칙과 구별합니다.")
        rows = []
        for pname, value, meaning in info["parameters"]:
            actual = params.get(value, value) if isinstance(value, str) else value
            rows.append({"이름": pname, "현재 값": str(actual), "정하는 것": meaning})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for limit in info["limits"]:
            st.write("• " + limit)
        st.markdown(f"[공식 문서]({info['source_url']})")


def render_model(name, result, frame, targets, features, params, rule_mode, space, show_names, metadata):
    explain_model(name, params)
    if result is None or result.get("model") is None:
        st.info((result or {}).get("message", "학습용 평가점수가 준비되지 않았습니다."))
        return
    st.divider()
    st.markdown("#### " + ("기준 점수 재현 평가" if rule_mode else "별도 평가점수에 대한 교차검증"))
    st.caption(f"{result['cv_name']} · 학습 대상 {result['n_labeled']}개 구. "
               "각 검증용 구를 제외한 자료로 전처리와 모델을 학습했습니다.")
    metrics = result.get("metrics", {})
    cols = st.columns(4)
    for col, label, key in zip(cols, ["MAE (점)", "RMSE (점)", "R²", "순위 상관"],
                               ["MAE", "RMSE", "R2", "Spearman"]):
        value = metrics.get(key, np.nan)
        col.metric(label, f"{value:.3f}" if pd.notna(value) else "산출 불가")
    baseline = result.get("baseline_metrics", {})
    if baseline:
        st.caption(f"학습 구의 평균점수만 예측한 기준선 MAE: {baseline.get('MAE', np.nan):.3f}점. "
                   "MAE·RMSE는 작을수록 좋고, R²는 정확도(%)가 아닙니다. "
                   "순위 상관은 목표 점수 순서와 얼마나 일치하는지 나타냅니다.")
    if rule_mode:
        st.info("현재 정답은 전력 수요와 설비 자료로 계산한 기준 점수입니다. "
                "검증 수치는 이 기준의 재현 정도이며 실제 설치 시급성·발전 성과를 검증한 정확도가 아닙니다.")
    if result.get("message"):
        st.caption(result["message"])
    view = st.radio("표시할 예측", ["교차검증 예측", "전체 평가자료 학습 후 추정"],
                    horizontal=True, key="view_" + name)
    oof = view == "교차검증 예측"
    pred = result["oof"] if oof else result["full"]
    shown = frame.copy()
    shown["학습대상점수"] = targets.reindex(shown.index)
    shown["예측시급도점수"] = pred["score"].reindex(shown.index)
    shown["범위제한전예측"] = pred["raw_score"].reindex(shown.index)
    shown["예측순위"] = shown["예측시급도점수"].rank(method="min", ascending=False).astype("Int64")
    shown["평가점수보유"] = shown["학습대상점수"].notna()
    if not oof:
        st.caption("평가점수가 있는 구는 학습자료에 대한 추정, 평가점수가 없는 구는 미평가 구에 대한 예측입니다. "
                   "이 화면의 학습자료 추정값으로 성능을 계산하지 않습니다.")
    coords = None
    if space == "모델 입력 보기":
        coords = result["model"].named_steps["preprocess"].transform(frame[features])
        st.caption("좌표는 최종 모델의 전처리 기준으로 표시합니다. 교차검증에서는 각 학습 폴드의 기준을 따로 사용합니다. "
                   "음수는 변환 후 평균보다 작다는 뜻이며 실제 전력량이 음수라는 뜻은 아닙니다.")
        if len(features) > 2:
            st.caption("추가 특성도 예측에 사용합니다. 그림은 전력량·설비용량 두 축만 보여줍니다.")
    else:
        st.caption("그래프는 GWh·MW, 학습 전 기준 단위는 MWh·kW입니다. 표시 방식 변경은 학습 결과를 바꾸지 않습니다.")
    score_plot(shown, "예측시급도점수", name + " · " + view, "plot_" + name, coords, show_names)
    ordered = shown.sort_values(["예측시급도점수", "자치구"], ascending=[False, True], na_position="last")
    columns = ["예측순위", "자치구", "학습대상점수", "예측시급도점수", "기준시급도점수",
               "연간소비량_GWh", "설비용량_MW", "평가점수보유"]
    st.dataframe(ordered[columns].round(3), hide_index=True, width="stretch")
    with st.expander("예측 범위·오차 확인"):
        clip = result.get("clipping", {})
        st.write(f"0~100 밖의 예측은 범위를 제한하여 표시·평가합니다. "
                 f"교차검증 {clip.get('oof_count', 0)}건 / 최종 추정 {clip.get('full_count', 0)}건.")
        if result.get("raw_metrics"):
            st.write("범위 제한 전 교차검증 지표", result["raw_metrics"])
        valid = shown.dropna(subset=["학습대상점수", "예측시급도점수"])
        if oof and not valid.empty:
            chart = px.scatter(valid, x="학습대상점수", y="예측시급도점수", hover_name="자치구",
                               title="목표 점수와 교차검증 예측")
            chart.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                            line=dict(color="#999", dash="dash"))
            chart.update_xaxes(range=[0, 100])
            chart.update_yaxes(range=[0, 100])
            st.plotly_chart(format_axes(chart, ",.1f", ",.1f"), width="stretch", key="error_" + name)
        st.caption("전체 자료를 표준화한 뒤 분할하지 않습니다. 점수·순위·점수 구성요소는 입력 변수로 사용하지 않습니다.")
    export = ordered.copy()
    for key, value in metadata.items():
        export[key] = value
    export["모델"] = name
    export["예측유형"] = view
    export["학습설정"] = str(params)
    slug = ["linear", "ridge", "lasso", "gradient_boosting", "svr"][MODEL_NAMES.index(name)]
    st.download_button("점수·순위·산정 기준 다운로드", csv_bytes(export), f"seoul_urgency_{slug}.csv",
                       "text/csv", key="download_" + slug)


def source_records(paths, uploaded=None):
    records = {}
    for path in paths:
        try:
            records[str(path.resolve())] = {"name": path.name, "suffix": path.suffix.lower(),
                                           "content": path.read_bytes()}
        except OSError as exc:
            st.warning(f"파일을 읽지 못했습니다: {path.name} ({exc})")
    if uploaded is not None:
        import hashlib
        content = uploaded.getvalue()
        identity = "upload::" + hashlib.sha256(content).hexdigest()[:16]
        records = {identity: {"name": uploaded.name, "suffix": Path(uploaded.name).suffix.lower(),
                              "content": content}, **records}
    return records


def main():
    st.set_page_config(page_title="서울 태양광 설치 검토 시급도", page_icon="☀️", layout="wide")
    st.title("☀️ 서울 태양광 추가 설치 검토 시급도")
    st.caption("2025년 전력 수요와 제공된 태양광 설비 목록으로 시급도 점수·순위를 비교합니다. "
               "선택한 다섯 회귀 모델을 같은 조건에서 검증합니다.")
    all_files = discover_files()
    power_paths = [p for p in all_files if ("energyuse" in p.stem.lower() or "전력" in p.stem)
                   and re.search(r"2025|(?<!\d)25(?!\d)", p.stem)]
    solar_paths = [p for p in all_files if "태양광" in p.stem]
    st.sidebar.header("데이터 선택")
    with st.sidebar.expander("CSV·Excel 직접 업로드", expanded=not bool(power_paths)):
        power_upload = st.file_uploader("전력 자료", type=["csv", "xlsx", "xls"], key="power_upload")
        solar_upload = st.file_uploader("태양광 자료", type=["csv", "xlsx", "xls"], key="solar_upload")
    powers = source_records(power_paths, power_upload)
    solars = source_records(solar_paths, solar_upload)
    if not powers:
        st.info("전력 CSV를 main.py 옆 또는 data 폴더에 두거나 사이드바에서 업로드하세요.")
        st.code("main.py\ndata/\n  energyUse_data 25.csv\n  시군구별전력사용량 2025.csv\n  서울시 태양광발전사업 정보.csv", language="text")
        st.stop()
    power_id = st.sidebar.selectbox("전력 사용량 파일", list(powers), format_func=lambda x: powers[x]["name"])
    power_source = powers[power_id]
    try:
        power_raw = read_file(power_source["content"], power_source["suffix"])
        power = prepare_power(power_raw)
        remembered = st.session_state.setdefault("saved_power_units", {})
        unit_options = ["kWh", "MWh", "GWh"]
        unit = st.sidebar.selectbox("전력 원본 단위", unit_options,
                                   index=unit_options.index(remembered.get(power_id, power["default_unit"])),
                                   key="unit_" + power_id)
        remembered[power_id] = unit
        if unit != power["unit"]:
            power = prepare_power(power_raw, unit)
        st.sidebar.caption("단위는 원본 설명으로 확인하세요. 화면 표시 단위와 구별합니다.")
    except Exception as exc:
        st.error(f"전력 자료 확인 필요: {exc}")
        st.stop()
    power_sources = {power_id: power}
    source_errors = []
    for key, record in powers.items():
        if key == power_id:
            continue
        try:
            power_sources[key] = prepare_power(read_file(record["content"], record["suffix"]), remembered.get(key))
        except Exception as exc:
            source_errors.append(record["name"] + ": " + str(exc))

    solar = None
    solar_raw = None
    solar_source = None
    solar_error = "태양광 자료를 선택하면 시급도와 모델 결과를 표시합니다."
    if solars:
        solar_id = st.sidebar.selectbox("태양광 파일", list(solars), format_func=lambda x: solars[x]["name"])
        solar_source = solars[solar_id]
        try:
            solar_raw = read_file(solar_source["content"], solar_source["suffix"])
            with st.sidebar.expander("태양광 원본 단위·컬럼 확인"):
                header = int(st.number_input("태양광 제목 행 (첫 행=1)", min_value=1,
                                            max_value=max(1, min(30, len(solar_raw))), value=1, key="header_" + solar_id))
                table = make_table(solar_raw, header - 1)
                columns = list(table.columns)
                addr = st.selectbox("주소 또는 자치구 컬럼", columns,
                                    index=guess_column(columns, ["소재지", "주소", "자치구", "시군구"]), key="addr_" + solar_id)
                cap = st.selectbox("설비용량 컬럼", columns, index=guess_column(columns, ["용량"]), key="cap_" + solar_id)
                cap_unit = st.selectbox("설비용량 원본 단위", ["kW", "MW", "W"], key="cap_unit_" + solar_id)
                status_choices = ["사용 안 함"] + columns
                default_status = next((c for c in columns if "가동상태" in c or "운영상태" in c), "사용 안 함")
                status = st.selectbox("운영 상태 컬럼", status_choices,
                                      index=status_choices.index(default_status), key="status_" + solar_id)
                selected_states = None
                if status != "사용 안 함":
                    states = sorted(table[status].dropna().astype(str).unique())
                    active = [s for s in states if normalize(s) in {"가동중", "운영중", "정상가동", "정상운영"}]
                    selected_states = st.multiselect("집계할 운영 상태", states, default=active or states,
                                                     key="states_" + solar_id + status)
                allow_zero = st.checkbox("용량 0을 확인된 실제 값으로 사용", value=False, key="allow_zero")
                drop_duplicates = st.checkbox("완전 동일한 행만 중복 제거", value=False, key="drop_duplicates")
                st.caption("주소가 같아도 서로 다른 시설 기록은 유지하고 구별로 합산합니다. 미확인 값은 0으로 채우지 않습니다. "
                           "자료기준일·설치연도를 이유로 기록을 자동 제외하지 않습니다.")
            solar = prepare_solar(solar_raw, header - 1, addr, cap, cap_unit, status,
                                  selected_states, allow_zero, drop_duplicates)
            solar_error = None
        except Exception as exc:
            solar_error = str(exc)

    st.sidebar.header("시급도 점수의 기준")
    target_mode = st.sidebar.radio("학습할 점수", ["기준 점수 재현 실험", "별도 평가점수 사용"], key="target_mode")
    rule_mode = target_mode == "기준 점수 재현 실험"
    with st.sidebar.expander("기준 점수 수식의 설정"):
        weight = st.slider("수요 규모 가중치", 0.0, 1.0, 0.5, 0.05)
        demand_ref = st.number_input("수요 기준값 (GWh)", min_value=1.0, value=2000.0, step=100.0)
        intensity_ref = st.number_input("수요 대비 설비 기준값 (kW/GWh)", min_value=0.001,
                                        value=1.0, step=0.1, format="%.3f")
        st.caption("가중치·기준값은 탐색용 가정입니다. 공인 시급도 기준이나 최적값이 아닙니다.")
    labels = None
    label_error = None
    label_name = "기준 수식"
    if not rule_mode:
        label_upload = st.sidebar.file_uploader("별도 평가점수 파일", type=["csv", "xlsx", "xls"], key="label_upload")
        if label_upload is not None:
            try:
                labels = read_evaluation_labels(label_upload.getvalue(), Path(label_upload.name).suffix.lower())
                label_name = label_upload.name
            except Exception as exc:
                label_error = str(exc)
        else:
            label_error = "자치구·시급도점수(0~100)가 있는 평가 파일을 업로드하세요. 빈 점수는 미평가로 남깁니다."
        st.sidebar.caption("기준 수식을 복사한 점수는 독립 평가가 아닙니다. 입력한 점수의 평가 근거를 함께 관리하세요.")
    template = pd.DataFrame({"자치구": sorted(SEOUL_GU), "시급도점수": [np.nan] * len(SEOUL_GU)})
    st.sidebar.download_button("평가점수 입력 양식", csv_bytes(template), "urgency_labels_template.csv", "text/csv")

    combined = combine_with_quality(power, solar) if solar is not None else None
    policy = None
    features = ["연간소비량_MWh", "설비용량_kW"]
    st.sidebar.header("회귀 입력·전처리")
    use_log = st.sidebar.checkbox("전력량·설비용량에 log1p 적용", value=True)
    st.sidebar.caption("모든 모델에 표준화를 적용하며, 검증 시 학습용 구 안에서만 평균·표준편차를 계산합니다.")
    if combined is not None:
        if st.sidebar.checkbox("여름(6~8월) 사용 비중 추가", value=False):
            features.append("여름사용비중_pct")
        sector_options = ["사용 안 함"] + [c for c in ["가정용비중_pct", "서비스업비중_pct"]
                                                 if c in combined and combined[c].notna().any()]
        if len(sector_options) > 1:
            sector = st.sidebar.selectbox("업종 비중 1개 추가", sector_options,
                                          format_func=lambda x: FEATURE_LABELS.get(x, x), key="sector_" + power_id)
            if sector != "사용 안 함":
                features.append(sector)
        policy = make_policy_scores(combined, demand_ref, intensity_ref, weight)
    params = {"ridge_alpha": 1.0, "lasso_alpha": 0.1, "gb_estimators": 100, "gb_learning_rate": 0.05,
              "gb_depth": 2, "gb_leaf": 3, "svr_c": 10.0, "svr_epsilon": 1.0,
              "svr_gamma": "scale", "svr_kernel": "rbf"}
    with st.sidebar.expander("다섯 모델의 하이퍼파라미터"):
        params["ridge_alpha"] = st.number_input("Ridge alpha", min_value=0.001, value=1.0, step=0.1)
        params["lasso_alpha"] = st.number_input("Lasso alpha", min_value=0.001, value=0.1, step=0.01, format="%.3f")
        params["gb_estimators"] = int(st.number_input("부스팅 나무 수", 20, 300, 100, 20))
        params["gb_learning_rate"] = st.number_input("부스팅 학습률", 0.001, 1.0, 0.05, 0.01, format="%.3f")
        params["gb_depth"] = int(st.number_input("부스팅 최대 깊이", 1, 5, 2))
        params["gb_leaf"] = int(st.number_input("부스팅 말단 최소 구 수", 2, 10, 3))
        params["svr_c"] = st.number_input("SVR C", 0.01, 1000.0, 10.0, 1.0)
        params["svr_epsilon"] = st.number_input("SVR epsilon (점)", 0.0, 20.0, 1.0, 0.1)
        params["svr_kernel"] = st.selectbox("SVR kernel", ["rbf", "linear"])
        st.caption("SVR gamma='scale'. 같은 검증 결과를 보고 설정을 반복 선택하면 낙관적 평가가 될 수 있습니다.")
    frame = None
    targets = None
    results = {}
    excluded = []
    if policy is not None:
        feature_values = policy[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = feature_values.notna().all(axis=1) & policy["기준시급도점수"].notna()
        frame = policy.loc[valid].copy()
        excluded = policy.loc[~valid, "자치구"].tolist()
        targets = (frame["기준시급도점수"].copy() if rule_mode else
                   frame["자치구"].map(labels.set_index("자치구")["시급도점수"]) if labels is not None else
                   pd.Series(np.nan, index=frame.index, dtype=float))
        if len(frame):
            with st.spinner("다섯 회귀 모델을 비교하고 있습니다..."):
                results = cached_models(frame, features, targets, use_log, params)
    metadata = {"전력자료": power_source["name"], "전력원본단위": unit,
                "태양광자료": solar_source["name"] if solar_source else "없음",
                "목표정의": target_mode, "평가자료": label_name,
                "수요기준_GWh": demand_ref, "설비비율기준_kW_GWh": intensity_ref,
                "수요가중치": weight, "로그변환": use_log, "입력변수": ", ".join(features)}
    summary = st.columns(4)
    summary[0].metric("전력 분석", f"{len(power['annual'])}개 구")
    summary[1].metric("연간 전력 수요", f"{power['annual'].sum()/1000:,.1f} GWh")
    capacity_total = solar["agg"]["설비용량_kW"].sum(min_count=1) if solar else np.nan
    summary[2].metric("확인된 등록 설비", f"{capacity_total/1000:,.2f} MW" if pd.notna(capacity_total) else "미확인")
    summary[3].metric("모델 평가점수", f"{int(targets.notna().sum())}개 구" if targets is not None else "0개 구")
    if rule_mode:
        st.info("기본 모드의 점수는 공개된 수식으로 만든 탐색용 기준입니다. 실제 시급도 정답이 없어서 "
                "모델은 그 기준을 재현하도록 학습합니다. 별도 평가점수가 있으면 사이드바에서 전환하세요.")
    else:
        st.info("사용자가 제공한 평가점수를 학습합니다. 입력 자료·평가 시점과 점수의 근거를 확인해 해석하세요.")
    if solar_error:
        st.warning(solar_error)
    if label_error:
        st.info(label_error)
    if excluded:
        st.warning("설비·선택 특성 미확인으로 모델에서 제외한 구: " + ", ".join(excluded))
    missing_power = sorted(SEOUL_GU - set(power["annual"].index))
    if missing_power:
        st.warning("전력 자료에서 제외한 구: " + ", ".join(missing_power))
    tab_models, tab_policy, tab_clean, tab_power, tab_compare = st.tabs(
        ["🧠 회귀 모델 비교", "📍 기준 점수·순위", "🧹 원본·전처리", "📊 전력 현황", "🔎 자료 비교·해석"])
    with tab_models:
        render_models(results, frame, targets, features, params, rule_mode, metadata)
    with tab_policy:
        render_policy(policy, demand_ref, intensity_ref, weight, metadata)
    with tab_clean:
        render_preprocessing(power_raw, power, solar_raw, solar, policy, frame, features, use_log)
    with tab_power:
        render_power(power)
    with tab_compare:
        render_source_comparison(powers, power_sources, source_errors, power_id, solar,
                                 demand_ref, intensity_ref, weight)


def render_models(results, frame, targets, features, params, rule_mode, metadata):
    st.subheader("같은 입력·같은 검증 분할로 비교하는 다섯 회귀 모델")
    st.caption("점수가 클수록 설정한 기준에서 추가 설치 검토 시급도가 큽니다. "
               "숫자 점수와 순위로만 표시합니다. 모델의 성능 순위를 자동 확정하지 않습니다.")
    rows = []
    for name in MODEL_NAMES:
        result = results.get(name, {})
        rows.append({"모델": name, **result.get("metrics", {}),
                     "평균점수 기준선 MAE": result.get("baseline_metrics", {}).get("MAE", np.nan),
                     "평가 구 수": result.get("n_labeled", 0), "상태": result.get("status", "자료 대기")})
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    st.caption("모든 지표는 학습에서 제외된 구의 교차검증 예측으로 계산합니다. "
               "25개 안팎의 구에서 나온 작은 지표 차이는 확정적인 우열로 해석하지 않습니다.")
    space = st.radio("산점도 표시 방식", ["실제 규모 보기", "모델 입력 보기"], horizontal=True, key="model_space")
    show_names = st.checkbox("그래프에 모든 구 이름 표시", value=False, key="model_names")
    tabs = st.tabs(MODEL_NAMES)
    for tab, name in zip(tabs, MODEL_NAMES):
        with tab:
            render_model(name, results.get(name), frame, targets, features, params,
                         rule_mode, space, show_names, metadata)


def render_policy(policy, demand_ref, intensity_ref, weight, metadata):
    st.subheader("기준 점수의 정의와 구별 순위")
    st.write("전력 수요가 크고, 그 수요에 비해 등록된 태양광 설비가 적을수록 "
             "점수가 커지도록 정한 탐색용 기준입니다. 가중치·기준값은 사이드바에서 바꿀 수 있습니다.")
    st.latex(r"S=100\left[w\frac{D}{D+D_0}+(1-w)\frac{I_0}{I+I_0}\right],\qquad I=\frac{C}{D}")
    st.write(f"D: 연간 전력 사용량(GWh), C: 설비용량(kW), I: 수요 대비 설비(kW/GWh). "
             f"현재 D₀={demand_ref:,.1f}GWh, I₀={intensity_ref:g}kW/GWh, w={weight:.2f}입니다.")
    st.caption("각 기준값에서 해당 지수는 0.5입니다. 전체 구의 최솟값·최댓값이나 순위를 이용해 점수를 "
               "만들지 않으므로, 다른 구가 추가되어도 같은 입력의 점수는 유지됩니다. "
               "설비비율은 실제 발전량이나 전력 자급률이 아닙니다.")
    with st.expander("점수 계산 예시와 해석"):
        st.write(f"가상 예시: D={demand_ref:,.1f}GWh, C={demand_ref*intensity_ref:,.1f}kW이면 "
                 "두 지수가 모두 0.5가 되어 현재 가중치에서도 기준 점수는 50점입니다.")
        st.write("수요 규모와 수요 대비 설비의 희소성을 함께 봅니다. 두 항 모두 수요와 관련되므로 "
                 "독립적인 두 평가를 더한 것이라고 해석하지 않습니다.")
        st.write("이 기준은 정책·학습 설명을 위한 가정입니다. 설치 가능 면적, 일사량, 그늘, 계통 접속, "
                 "비용과 실제 발전성과를 추가해야 현장 설치 판단에 가까워집니다.")
    if policy is None:
        st.info("태양광 자료를 준비하면 기준 점수를 계산합니다.")
        return
    shown = policy.copy()
    shown["기준순위"] = shown["기준시급도점수"].rank(ascending=False, method="min").astype("Int64")
    shown["수요반영점수"] = 100 * weight * shown["수요지수"]
    shown["설비희소반영점수"] = 100 * (1-weight) * shown["설비희소지수"]
    ranked = shown.sort_values(["기준시급도점수", "자치구"], ascending=[False, True], na_position="last")
    valid = ranked.dropna(subset=["기준시급도점수"])
    st.caption("점수가 같으면 같은 순위를 부여합니다. 작은 소수점 차이가 실제 시급성의 확실한 차이를 뜻하지는 않습니다.")
    if not valid.empty:
        fig = px.bar(valid.sort_values("기준시급도점수"), y="자치구", x="기준시급도점수",
                     orientation="h", color="기준시급도점수", range_color=[0, 100],
                     color_continuous_scale="YlOrRd", title="구별 기준 시급도 점수")
        fig.update_xaxes(range=[0, 100], tickformat=".0f")
        fig.update_layout(height=max(500, len(valid)*24), yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch", key="policy_rank")
        score_plot(shown, "기준시급도점수", "전력 수요·설비 규모와 기준 점수", "policy_scatter")
    columns = ["기준순위", "자치구", "기준시급도점수", "수요반영점수", "설비희소반영점수",
               "연간소비량_GWh", "설비용량_MW", "수요대비설비_kW_GWh", "설비확인상태"]
    st.dataframe(ranked[columns].round(3), hide_index=True, width="stretch")
    export = ranked.copy()
    for key, value in metadata.items():
        export[key] = value
    st.download_button("전체 기준 점수와 산정 근거 다운로드", csv_bytes(export),
                       "seoul_urgency_policy_scores.csv", "text/csv", key="policy_download")


def render_preprocessing(power_raw, power, solar_raw, solar, policy, frame, features, use_log):
    st.subheader("원본 → 구별 집계 → 모델 입력")
    st.write("원본 단위는 전력량 MWh·설비용량 kW로 정리합니다. "
             "화면의 GWh·MW 변환은 표시용이며, 모델 입력의 로그 변환 전에 단위를 임의로 바꾸지 않습니다.")
    st.caption("MWh는 기간 동안 사용한 에너지, kW는 설비의 출력 규모입니다. "
               "두 물리량은 다르며, 모델에서 숫자 스케일을 조정하는 것은 표준화가 맡습니다.")
    logs = pd.DataFrame(power["logs"] + (solar["logs"] if solar else []))
    st.dataframe(logs, hide_index=True, width="stretch")
    st.download_button("전처리 로그 다운로드", csv_bytes(logs), "preprocessing_log.csv", "text/csv")
    if solar:
        if solar["unknown_address"]:
            st.warning(f"자치구를 확인하지 못한 태양광 기록 {solar['unknown_address']}건은 집계에서 제외했습니다. 원본 주소를 확인하세요.")
        if solar["incomplete_gu"]:
            st.warning("일부 용량·중복 기록 확인이 필요한 구: " + ", ".join(sorted(solar["incomplete_gu"])))
        if not solar["invalid_rows"].empty:
            with st.expander("확인이 필요한 태양광 기록"):
                show_table(solar["invalid_rows"], hide_index=True)
    for note in power["sector_notes"]:
        st.info(note)
    if policy is not None:
        st.markdown("#### 구별 결합 자료와 확인 상태")
        st.dataframe(policy, hide_index=True, width="stretch")
        st.caption("미등록·부분 결측 구의 설비를 0으로 대체하지 않습니다. 일부 시설의 용량이 빠진 구도 "
                   "설비 합계를 미확인 처리합니다. 큰 실제 값은 이상치라는 이유만으로 삭제하지 않습니다.")
    if frame is not None and not frame.empty:
        st.markdown("#### 모델 입력 전처리 전후")
        raw = frame[features].to_numpy(dtype=float)
        logged = raw.copy()
        if use_log:
            for i, col in enumerate(features):
                if col in {"연간소비량_MWh", "설비용량_kW"}:
                    logged[:, i] = np.log1p(logged[:, i])
        scaled = build_preprocessor(features, use_log).fit_transform(frame[features])
        demo = pd.DataFrame({"자치구": frame["자치구"].to_numpy()})
        for i, col in enumerate(features):
            label = FEATURE_LABELS.get(col, col)
            demo[label + "_원본"] = raw[:, i]
            if use_log and col in {"연간소비량_MWh", "설비용량_kW"}:
                demo[label + "_로그변환"] = logged[:, i]
            demo[label + "_표준화"] = scaled[:, i]
        st.dataframe(demo.round(4), hide_index=True, width="stretch")
        st.caption("이 표는 전처리를 설명하기 위해 분석 대상 전체로 계산한 예시입니다. "
                   "성능 평가에서는 이 표를 사용하지 않고 학습 폴드 안에서 전처리를 다시 학습합니다.")
        st.latex(r"z=\frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}")
        st.caption("로그 변환을 켠 경우 x는 log(1+원본값)입니다. 표준화가 분포를 자동으로 정규분포로 바꾸는 것은 아닙니다.")
    with st.expander("전력 원본·자료형·결측치"):
        st.caption(f"원시 표 {len(power_raw):,}행. 화면에는 처음 100행을 보여줍니다.")
        show_table(power_raw.head(100), hide_index=True)
        st.dataframe(describe_columns(power["original"]), hide_index=True, width="stretch")
    if solar_raw is not None:
        with st.expander("태양광 원본·자료형·결측치"):
            show_table(solar_raw.head(100), hide_index=True)
            if solar:
                st.dataframe(describe_columns(solar["original"]), hide_index=True, width="stretch")
    with st.expander("전처리된 월별 전력 사용량 (MWh)"):
        st.dataframe(power["matrix"], width="stretch")
        st.download_button("월별 전력 데이터 다운로드", csv_bytes(power["matrix"].reset_index()),
                           "seoul_2025_monthly_MWh.csv", "text/csv")


def render_power(power):
    st.subheader("연간 전력 사용량과 월별 패턴")
    demand = (power["annual"].div(1000).rename("연간소비량_GWh").rename_axis("자치구")
              .reset_index().sort_values("연간소비량_GWh", ascending=False))
    fig = px.bar(demand, x="자치구", y="연간소비량_GWh", color="연간소비량_GWh",
                 color_continuous_scale="YlOrRd", labels={"연간소비량_GWh": "연간 전력 사용량 (GWh)"},
                 title="2025년 자치구별 연간 전력 사용량")
    fig.update_yaxes(tickformat=",.0f", exponentformat="none")
    fig.update_layout(height=450, xaxis=dict(categoryorder="array", categoryarray=demand["자치구"].tolist()))
    st.plotly_chart(fig, width="stretch", key="annual_power")
    left, right = st.columns(2)
    with left:
        st.markdown("**사용량 상위 3개 구**")
        st.dataframe(demand.head(3).round(2), hide_index=True, width="stretch")
    with right:
        st.markdown("**사용량 하위 3개 구**")
        st.dataframe(demand.tail(3).sort_values("연간소비량_GWh").round(2), hide_index=True, width="stretch")
    st.caption("총사용량에는 인구·건물 규모·업종 구성이 영향을 줍니다. 사용량만으로 효율이나 설치 가능성을 판단하지 않습니다.")
    mode = st.radio("월별 히트맵", ["연간 대비 월별 비율 (%)", "월별 사용량 (GWh)"], horizontal=True)
    relative = mode.startswith("연간")
    heat = (power["shares"] if relative else power["matrix"]/1000).reindex(demand["자치구"])
    fig = go.Figure(go.Heatmap(z=heat.to_numpy(), x=MONTHS, y=heat.index.tolist(), colorscale="YlOrRd",
                             colorbar=dict(title="비율 (%)" if relative else "GWh"),
                             hovertemplate="%{y}<br>%{x}: %{z:,.2f}<extra></extra>"))
    fig.update_layout(height=max(520, len(heat)*24), yaxis=dict(autorange="reversed"), title=mode)
    st.plotly_chart(fig, width="stretch", key="monthly_power")
    st.caption("월별 사용량은 낮 시간대 수요나 순간 최대전력과 다릅니다. "
               "월별 셀을 독립적인 설치 시급도 정답으로 간주하지 않습니다.")


def render_source_comparison(powers, sources, errors, selected, solar, demand_ref, intensity_ref, weight):
    st.subheader("전력 자료를 바꿨을 때의 차이")
    rows = []
    base = sources[selected]["annual"]
    for key, data in sources.items():
        common = base.index.intersection(data["annual"].index)
        baseline = base.reindex(common).sum()
        compare = data["annual"].reindex(common).sum()
        row = {"전력파일": powers[key]["name"], "현재선택": key == selected,
               "원본단위": data["unit"], "분석구수": len(data["annual"]),
               "연간합계_GWh": data["annual"].sum()/1000,
               "공통구수": len(common), "선택자료대비차이_pct": (compare-baseline)/baseline*100 if baseline else np.nan}
        if solar:
            scored = make_policy_scores(combine_with_quality(data, solar), demand_ref, intensity_ref, weight)
            top = scored.dropna(subset=["기준시급도점수"]).sort_values(
                ["기준시급도점수", "자치구"], ascending=[False, True]).head(5)
            row["같은수식_점수상위5개구"] = ", ".join(top["자치구"])
        rows.append(row)
    comparison = pd.DataFrame(rows)
    st.dataframe(comparison.round(3), hide_index=True, width="stretch")
    st.caption("각 파일은 같은 구에 대한 별도 집계 자료입니다. 합산·평균하거나 표본 수를 늘리는 데 사용하지 않습니다. "
               "아직 선택하지 않은 파일은 형식별 기본 단위로 읽으며, 파일별 단위 설정을 기억합니다.")
    if len(sources) > 1:
        with st.expander("전력 파일별 구별 사용량 비교 (GWh)"):
            detail = pd.concat({powers[key]["name"]: data["annual"]/1000 for key, data in sources.items()}, axis=1)
            st.dataframe(detail.round(3), width="stretch")
        st.info("자료 간 차이는 집계 대상·작성 기준 차이일 수 있습니다. 파일만으로 어느 쪽이 더 정확한지 확정하지 않습니다.")
    for error in errors:
        st.warning("자료 비교에서 제외: " + error)
    st.download_button("자료 비교 다운로드", csv_bytes(comparison), "seoul_power_source_comparison.csv", "text/csv")
    if solar and solar["dates"]:
        st.caption("태양광 자료에 표시된 기준일: " + ", ".join(solar["dates"]) + ". 기준일·설치연도를 이유로 자동 제외하지 않았습니다.")
    st.markdown("""
### 결과를 해석할 때
- 분석 대상은 구별 집계 자료입니다. 시설 기록이나 월별 셀 수가 독립적인 시급도 정답 수가 되지는 않습니다.
- 기본 점수는 공개된 수식으로 계산합니다. 이 수식을 학습한 모델은 기준 재현 실험이며, 실제 설치 시급성의 검증이 아닙니다.
- 별도 평가점수를 사용하더라도 작은 표본의 탐색적 검증입니다. 미래 예측을 검증하려면 시점을 분리한 자료가 필요합니다.
- 현재 자료의 시점을 맞춘 과거 예측이라고 주장하지 않습니다. 2025년 전력 수요와 제공된 설비 목록을 함께 분석합니다.
- 설비용량은 실제 발전량이 아니며, 설비 목록에 없다고 실제 설비가 없는 것은 아닙니다.
- 수요 대비 설비가 적다는 사실은 해당 구에 전력 공급 부족이 발생했다는 뜻이 아닙니다.
- 실제 입지를 결정하려면 설치 가능 면적·일사량·그늘·건물 구조·계통 접속·비용 정보를 추가해야 합니다.
""")


if __name__ == "__main__":
    main()

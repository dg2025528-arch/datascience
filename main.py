그럼 베이스라인을 뭘로 해야할까?
우리 프로젝트에서는 ‘훈련 데이터의 시급성 점수 평균을 모든 자치구에 똑같이 예측하는 모델’을 기본 베이스라인으로 추천해. 코드로는 **`DummyRegressor(strategy="mean")`**이야. 그리고 릿지는 입력 변수까지 사용하는 추가 비교 모델로 두면 좋아.

다만 평균 예측은 MSE·RMSE 기준에 잘 맞고, MAE를 최종 평가 기준으로 정한다면 중앙값 예측을 기준선으로 사용하는 것이 더 적절해. 이 차이까지 연결해서 설명할게.

1. 베이스라인이 하는 일
베이스라인은 우리가 만든 모델의 성능을 판단하기 위한 출발점이야.

예를 들어 SVR의 예측 오차가 8점이라고 해 보자. 이 숫자만으로는 성능이 좋은지 판단하기 어려워.

그런데 아주 단순하게 예측해도 오차가 5점이라면, 현재 SVR의 설정이나 입력 변수를 다시 살펴봐야겠지. 반대로 단순 예측의 오차가 20점이라면, SVR이 어느 정도 개선했는지 설명할 수 있어.

그래서 먼저 **“자치구별 전력 사용량이나 설비 정보를 활용하지 않아도 얻을 수 있는 성능”**을 측정하는 거야. DummyRegressor는 이런 비교에 사용하는 간단한 기준 모델이야. 

2. 평균 예측 베이스라인을 우리 프로젝트에 적용하면
훈련 자료에 있는 자치구들의 시급성 점수가 다음과 같다고 가정하자.

40
,
 
50
,
 
60
,
 
70
,
 
80
40, 50, 60, 70, 80
평균은:

y
ˉ
train
=
40
+
50
+
60
+
70
+
80
5
=
60
y
ˉ
​
  
train
​
 = 
5
40+50+60+70+80
​
 =60
그러면 베이스라인은 새로운 자치구의 전력 사용량이나 태양광 설비량과 관계없이 모두 60점으로 예측해.

예측할 자치구	베이스라인의 예측 점수
A구	60점
B구	60점
C구	60점
위 숫자는 설명용 가상 예시야.

일반식으로 쓰면 다음과 같아.

y
^
baseline
(
x
)
=
1
n
train
∑
i
=
1
n
train
y
i
y
^
​
  
baseline
​
 (x)= 
n 
train
​
 
1
​
  
i=1
∑
n 
train
​
 
​
 y 
i
​
 
​
 
여기서 평균 내는 대상은 **전력 사용량이 아니라, 예측하려는 목표인 ‘시급성 점수 
y
y’**야. DummyRegressor의 mean 전략은 훈련 목표값의 평균을 계속 출력하는 방식으로 작동해. 

모든 자치구가 동점이므로 이 베이스라인 자체로 설치 순위를 정할 수는 없어. 대신 자치구별 차이를 학습한 모델이 점수 예측을 얼마나 개선하는지 확인하는 기준이 돼.

3. 평균을 선택하는 근거와 중앙값을 선택할 경우
평균은 단순히 익숙해서 선택하는 값이 아니야.

모든 자료에 같은 숫자 하나를 예측하는 방법 중, 훈련 자료의 제곱오차 합을 가장 작게 만드는 값이 평균이야. 반면 절대오차 합을 가장 작게 만드는 값은 중앙값이야. 따라서 평가 지표와 다음처럼 연결할 수 있어. 

최종 평가에서 중요하게 볼 기준	적절한 상수 베이스라인	설정
MSE·RMSE: 큰 점수 오차에 더 큰 벌점을 줌	훈련 점수의 평균	strategy="mean"
MAE: 몇 점씩 틀리는지 절대오차로 평가	훈련 점수의 중앙값	strategy="median"
예를 들어 훈련 점수가 다음과 같으면:

20
,
 
25
,
 
30
,
 
35
,
 
90
20, 25, 30, 35, 90
평균은 40점이고, 중앙값은 30점이야.

평균 베이스라인은 모든 자치구를 40점으로 예측해.
중앙값 베이스라인은 모든 자치구를 30점으로 예측해.
90점처럼 큰 값이 평균을 끌어올리기 때문에 둘이 달라지는 거야.

지금 기본 베이스라인을 하나 정한다면 평균 예측으로 시작하면 돼. 이후 모델을 MAE로 최종 선정하기로 한다면 중앙값 베이스라인도 반드시 함께 비교하자. 이때 평균·중앙값이 각각 해당 훈련 손실을 최소화한다는 것이지, 새로운 검증 자료에서도 항상 더 좋다는 보장은 아니야.

4. 앞에서 말한 릿지는 어떤 역할인가?
베이스라인에도 여러 수준이 있어. 앞에서 릿지를 ‘기준 모델’이라고 한 것은 간단한 학습 모델을 기준으로 더 복잡한 모델의 필요성을 확인한다는 의미였어.

우리 프로젝트에서는 다음처럼 역할을 나누면 명확해.

역할	사용할 모델	확인하려는 내용
기본 베이스라인	평균 예측 DummyRegressor	자치구별 입력 정보를 사용하면 예측이 개선되는가?
간단한 회귀 비교 모델	Ridge	규제를 적용한 선형 관계만으로도 충분한가?
비선형 회귀 비교 모델	SVR, 그래디언트 부스팅	더 복잡한 관계를 학습하는 것이 추가로 도움이 되는가?
예를 들어 그래디언트 부스팅이 평균 예측보다 좋아도 릿지와 성능이 거의 같다면, 복잡한 모델을 사용할 이유를 더 검토할 수 있어. 릿지보다도 안정적으로 좋은 결과를 보인다면, 그래디언트 부스팅을 선택한 근거가 더 강해지겠지.

5. 실제 비교에서는 이 두 가지를 지켜야 해
첫째, 평균은 훈련 자료에서만 계산해야 해.

자료가 자치구별 한 행씩 총 25행이고, 5개씩 검증하는 5겹 교차 검증을 한다면:

20개 구의 점수로 평균을 계산해.
나머지 5개 구에 그 평균을 예측값으로 줘.
실제 점수와 비교해 오차를 계산해.
검증할 구를 바꾸고, 새로운 훈련 20개 구에서 평균을 다시 계산해.
검증할 구의 실제 점수까지 포함해 평균을 계산하면, 평가에 사용할 정보를 미리 본 셈이 돼. 

둘째, 모든 모델을 같은 자료 분할과 같은 평가 지표로 비교해야 해.

평균 베이스라인, 릿지, SVR, 그래디언트 부스팅을 동일한 검증 자치구에서 평가해야 차이를 공정하게 해석할 수 있어. MAE와 RMSE를 함께 기록하면, 평균적인 점수 오차와 큰 오차에 대한 성능을 각각 살펴볼 수 있어. 

우리 실험에는 평균 예측 베이스라인 + 릿지 비교 모델 + 선택한 최종 모델을 함께 두자. 그러면 발표에서도 “최종 모델의 오차가 몇 점이다”에 더해, **“단순한 평균 예측보다 얼마나 개선됐고, 선형 모델보다도 도움이 되는가”**까지 근거를 갖춰 설명할 수 있어.

아니 우리 베이스라인이 누구임:
우리 베이스라인은 **평균 예측 모델(DummyRegressor, strategy="mean")**이야.

훈련 데이터의 시급성 점수 평균을 모든 자치구에 똑같이 예측하는 모델이야. 예를 들어 평균이 60점이면, 어느 구든 60점으로 예측해.

스트림리세 베이스라인 넣놔야지

베이스라인을 추가한 
main.py 다운로드
 수정본은 완성했고, 실행 검증도 끝났어. 다만 실제 Streamlit 사이트에는 아직 반영되지 않았어. GitHub 저장소에 올리려면 플러그인 설치와 계정 연결이 필요해.


추가한 내용은 다음과 같아.

비교표 첫 행: 평균 예측 베이스라인을 넣고 기존 5개 모델과 MAE·RMSE·R²를 비교.
베이스라인 전용 탭: 학습 평균을 계산하는 방식, 검증 묶음별 평균, 구별 예측값 표시.
개선량 표시: 각 모델이 베이스라인보다 오차를 얼마나 줄였는지 표시.
베이스라인은 **`DummyRegressor(strategy="mean")`**이고, 다른 모델과 동일한 5겹 교차검증을 사용해. 실제 25개 구 데이터로 화면 실행을 확인했고, 기존 5개 모델의 예측값도 수정 전과 동일했어.

직접 적용하려면 GitHub 저장소의 를 위 파일로 교체해서 커밋하면 돼.


main.py
코드




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

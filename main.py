import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

plt.rcParams['font.family'] = 'Malgun Gothic'   # Mac은 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------
# 1. 데이터 읽기 (첫 줄은 제목이라 건너뜀, 숫자의 쉼표 제거)
# ------------------------------------------------------------
FILE = 'energyUse_data_25.csv'
df = pd.read_csv(FILE, skiprows=1, thousands=',', encoding='utf-8-sig')
df = df[df['자치구명'] != '총합'].copy()          # 총합 행 제외

months = [f'{m}월' for m in range(1, 13)]
pivot = df.set_index('자치구명')[months].astype(float)   # 구 x 월 (MWh)
pivot.columns = range(1, 13)
print(pivot.shape)   # (25, 12)이면 정상

# ------------------------------------------------------------
# 2. 구별 특성
# ------------------------------------------------------------
gu = pd.DataFrame(index=pivot.index)
gu['연간사용량_GWh'] = pivot.sum(axis=1) / 1000
spring = pivot[[4, 5]].mean(axis=1)
gu['여름_봄_비율'] = pivot[[7, 8]].mean(axis=1) / spring
gu['겨울_봄_비율'] = pivot[[12, 1, 2]].mean(axis=1) / spring
gu['계절변동계수'] = pivot.std(axis=1) / pivot.mean(axis=1)
gu = gu.reset_index().rename(columns={'자치구명': '구'})
n = len(gu)

# ------------------------------------------------------------
# 3. 그래프 1: 구별 연간 사용량 막대 (정렬)
# ------------------------------------------------------------
s = gu.sort_values('연간사용량_GWh')
colors = ['#1b9e77' if i < 3 else '#d95f02' if i >= n - 3 else '#9ecae1'
          for i in range(n)]
fig, ax = plt.subplots(figsize=(8, 9))
ax.barh(s['구'], s['연간사용량_GWh'], color=colors)
ax.set_xlabel('연간 전력 사용량 (GWh)')
ax.set_title('2025 서울 자치구별 연간 전력 사용량 (상위 3: 주황 / 하위 3: 초록)')
plt.tight_layout()
plt.savefig('1_bar_usage.png', dpi=200)
plt.show()
print('상위 3:', s['구'].tail(3).tolist()[::-1])
print('하위 3:', s['구'].head(3).tolist())

# ------------------------------------------------------------
# 4. 그래프 2: 히트맵 (구 x 월), 구별 연평균 대비 비율
# ------------------------------------------------------------
ratio = pivot.div(pivot.mean(axis=1), axis=0)
ratio = ratio.loc[s['구'][::-1]]          # 사용량 큰 구가 위로
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(ratio.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(12))
ax.set_xticklabels(months)
ax.set_yticks(range(n))
ax.set_yticklabels(ratio.index)
for i in range(n):                          # 칸 안에 값 표시
    for j in range(12):
        ax.text(j, i, f'{ratio.values[i, j]:.2f}', ha='center', va='center', fontsize=6)
ax.set_title('구 × 월 전력 사용량 패턴 (구별 연평균 = 1.0)')
plt.colorbar(im, ax=ax).set_label('연평균 대비 비율')
plt.tight_layout()
plt.savefig('2_heatmap.png', dpi=200)
plt.show()

peak = pivot.idxmax(axis=1)
print('\n[피크 월별 구]')
for m, g in peak.groupby(peak):
    print(f'{m}월:', g.index.tolist())

# ------------------------------------------------------------
# 5. 그래프 3: K-means 군집 + PCA 산점도
# ------------------------------------------------------------
features = ['연간사용량_GWh', '여름_봄_비율', '겨울_봄_비율', '계절변동계수']
X = StandardScaler().fit_transform(gu[features])

best_k, best = 2, -1
for k in range(2, 7):
    sc_ = silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X))
    print(f'k={k}: 실루엣 {sc_:.3f}')
    if sc_ > best:
        best_k, best = k, sc_
print('선택된 k =', best_k)

gu['군집'] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X)
pca = PCA(n_components=2)
p = pca.fit_transform(X)
gu['PC1'], gu['PC2'] = p[:, 0], p[:, 1]

fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(gu['PC1'], gu['PC2'], c=gu['군집'], cmap='Set2', s=120, edgecolor='k')
for _, r in gu.iterrows():
    ax.annotate(r['구'], (r['PC1'], r['PC2']),
                textcoords='offset points', xytext=(5, 5), fontsize=9)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('2025 자치구 전력 사용 특성 군집 (PCA)')
ax.legend(*sc.legend_elements(), title='군집')
plt.tight_layout()
plt.savefig('3_cluster_pca.png', dpi=200)
plt.show()

# ------------------------------------------------------------
# 6. 해석용 출력
# ------------------------------------------------------------
print('\n[군집별 평균 특성]')
print(gu.groupby('군집')[features].mean().round(2))
for c in sorted(gu['군집'].unique()):
    print(f'군집 {c}:', gu[gu['군집'] == c]['구'].tolist())
print('\n[PCA 축 기여도]')
print(pd.DataFrame(pca.components_.T, index=features, columns=['PC1', 'PC2']).round(2))

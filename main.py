#importing libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import shap


plt.style.use('seaborn-v0_8-whitegrid')
BLUE = '#3498db'; RED = '#e74c3c'; GREEN = '#2ecc71'; ORANGE = '#e67e22'




#loading data
app = pd.read_csv('application_train.csv')
bureau = pd.read_csv('bureau.csv')




#Data Cleaning Pipleline
app['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
missing_pct = app.isnull().mean()
drop_cols = list(set(
    missing_pct[missing_pct > 0.45].index.tolist() +
    [c for c in app.columns if 'FLAG_DOCUMENT' in c]
))
app.drop(columns=drop_cols, inplace=True)
print(f"Dropped {len(drop_cols)} unusable columns")

for col in app.select_dtypes(include=[np.number]).columns:
    app[col].fillna(app[col].median(), inplace=True)
for col in app.select_dtypes(include=['object']).columns:
    app[col].fillna(app[col].mode()[0], inplace=True)



#Feature Engineering
app['CREDIT_INCOME_RATIO'] = app['AMT_CREDIT']  / (app['AMT_INCOME_TOTAL'] + 1)
app['ANNUITY_INCOME_RATIO'] = app['AMT_ANNUITY'] / (app['AMT_INCOME_TOTAL'] + 1)
app['CREDIT_TERM'] = app['AMT_ANNUITY'] / (app['AMT_CREDIT'] + 1)
app['GOODS_PRICE_RATIO'] = app['AMT_GOODS_PRICE'] / (app['AMT_CREDIT'] + 1)
app['AGE_YEARS'] = (-app['DAYS_BIRTH'])    / 365
app['EMPLOYED_YEARS'] = (-app['DAYS_EMPLOYED']) / 365
app['EMPLOYMENT_PCT'] = app['EMPLOYED_YEARS']   / (app['AGE_YEARS'] + 1)
app['ID_PUBLISH_AGE'] = (-app['DAYS_ID_PUBLISH']) / 365
app['PHONE_CHANGE_AGE'] = (-app['DAYS_LAST_PHONE_CHANGE']) / 365
app['INCOME_PER_PERSON'] = app['AMT_INCOME_TOTAL'] / (app['CNT_FAM_MEMBERS'] + 1)
app['INCOME_PER_CHILD'] = app['AMT_INCOME_TOTAL'] / (app['CNT_CHILDREN'] + 1)

for col in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']:
    if col not in app.columns:
        app[col] = 0.5

app['EXT_SOURCE_MEAN'] = app[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
app['EXT_SOURCE_MIN']  = app[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].min(axis=1)
app['EXT_SOURCE_PROD'] = app['EXT_SOURCE_1'] * app['EXT_SOURCE_2'] * app['EXT_SOURCE_3']

bureau['AMT_CREDIT_MAX_OVERDUE'].fillna(0, inplace=True)
bureau['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)
bureau['CREDIT_DAY_OVERDUE'].fillna(0, inplace=True)
bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0, inplace=True)
bureau['AMT_CREDIT_SUM'].fillna(0, inplace=True)

bureau_agg = bureau.groupby('SK_ID_CURR').agg(
    total_prev_loans = ('SK_ID_BUREAU', 'count'),
    avg_credit_debt = ('AMT_CREDIT_SUM_DEBT', 'mean'),
    max_days_overdue = ('CREDIT_DAY_OVERDUE', 'max'),
    total_overdue_amount = ('AMT_CREDIT_SUM_OVERDUE', 'sum'),
    max_overdue_ever = ('AMT_CREDIT_MAX_OVERDUE', 'max'),
    active_credits = ('CREDIT_ACTIVE', lambda x: (x=='Active').sum()),
    bad_credits = ('CREDIT_ACTIVE', lambda x: (x=='Bad debt').sum()),
    total_credit_sum = ('AMT_CREDIT_SUM', 'sum'),
    credit_utilisation = ('AMT_CREDIT_SUM_DEBT', 'sum'),
    num_prolonged = ('CNT_CREDIT_PROLONG', 'sum'),
).reset_index()
bureau_agg['BUREAU_DEBT_RATIO']  = bureau_agg['credit_utilisation'] / (bureau_agg['total_credit_sum'] + 1)
bureau_agg['OVERDUE_LOAN_RATIO'] = bureau_agg['bad_credits'] / (bureau_agg['total_prev_loans'] + 1)

app = app.merge(bureau_agg, on='SK_ID_CURR', how='left')
bcols = bureau_agg.columns.drop('SK_ID_CURR').tolist()
app[bcols] = app[bcols].fillna(0)
print(f"Created {len(bcols)} bureau features  |  Final shape: {app.shape}")




#Encoding and Splitting
X = app.drop(columns=['TARGET','SK_ID_CURR'])
y = app['TARGET']

le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col].astype(str))

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

print(f"Train : {X_train.shape[0]:,}  ({y_train.mean()*100:.1f}% default)")
print(f"Val   : {X_val.shape[0]:,}  ({y_val.mean()*100:.1f}% default)  <-- used for threshold tuning only")
print(f"Test  : {X_test.shape[0]:,}  ({y_test.mean()*100:.1f}% default)  <-- final evaluation")





#SMOTE OVERSAMPLING
train_medians = X_train.median()
X_train_imp   = X_train.fillna(train_medians).fillna(0)
X_val_imp     = X_val.fillna(train_medians).fillna(0)
X_test_imp    = X_test.fillna(train_medians).fillna(0)

smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

print(f"Before --> {y_train.sum():,} defaulters ({y_train.mean()*100:.1f}%)")
print(f"After  --> {y_train_sm.sum():,} defaulters ({y_train_sm.mean()*100:.1f}%)")

# Training LightGBM algorithm
X_tr, X_es, y_tr, y_es = train_test_split(
    X_train_sm, y_train_sm, test_size=0.1, random_state=42, stratify=y_train_sm
)

model = lgb.LGBMClassifier(
    n_estimators      = 2000,
    learning_rate     = 0.03,
    max_depth         = 8,
    num_leaves        = 63,
    min_child_samples = 30,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    scale_pos_weight  = 2,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_es, y_es)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
)
print(f"Done. Best iteration: {model.best_iteration_}")






#Threshold Tuning on the Real Dataset
val_probs = model.predict_proba(X_val_imp)[:,1]

sweep = []
for t in np.linspace(0.05, 0.60, 300):
    yp   = (val_probs >= t).astype(int)
    rec  = recall_score(y_val, yp, zero_division=0)
    prec = precision_score(y_val, yp, zero_division=0)
    f1   = f1_score(y_val, yp, zero_division=0)
    sweep.append({'t': t, 'recall': rec, 'precision': prec, 'f1': f1})

sweep_df = pd.DataFrame(sweep)

good = sweep_df[sweep_df['recall'] >= 0.60]
if len(good) > 0:
    best_row = good.loc[good['f1'].idxmax()]
else:
    best_row = sweep_df.loc[sweep_df['f1'].idxmax()]

best_threshold = float(best_row['t'])
print(f"Default threshold 0.50 --> recall = {sweep_df[sweep_df.t >= 0.499].iloc[0]['recall']:.3f}")
print(f"Tuned threshold  {best_threshold:.3f} --> recall = {best_row['recall']:.3f}  (target ≥ 60%)")







#Evaluation pipeline
y_prob      = model.predict_proba(X_test_imp)[:,1]
y_pred_050  = (y_prob >= 0.50).astype(int)
y_pred_opt  = (y_prob >= best_threshold).astype(int)

def metrics_block(yt, yp, yprob, label):
    d = {
        'Accuracy' : accuracy_score(yt, yp),
        'Precision': precision_score(yt, yp, zero_division=0),
        'Recall'   : recall_score(yt, yp, zero_division=0),
        'F1'       : f1_score(yt, yp, zero_division=0),
        'ROC-AUC'  : roc_auc_score(yt, yprob),
    }
    print(f"\n  [{label}]")
    for k, v in d.items():
        star = ' <-- KEY' if k == 'Recall' else ''
        print(f"  {k:<12}: {v:.4f}{star}")
    return d

r_050 = metrics_block(y_test, y_pred_050, y_prob, "LightGBM @ 0.50")
r_opt = metrics_block(y_test, y_pred_opt, y_prob, f"LightGBM @ tuned {best_threshold:.3f}")

print(f"\n--------------GAINS FROM THRESHOLD TUNING--------------")
for k in ['Recall','ROC-AUC','Precision','F1']:
    d = r_opt[k] - r_050[k]
    print(f"  {k:<12}: {r_050[k]:.3f} --> {r_opt[k]:.3f}  ({'up' if d>=0 else 'down'}{abs(d):.3f})")







#SHAP EXPLAINABILITY
explainer = shap.TreeExplainer(model)
X_shap    = X_test_imp.sample(1000, random_state=42)
sv_raw    = explainer.shap_values(X_shap)

if isinstance(sv_raw, list):
    sv = sv_raw[1]
elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
    sv = sv_raw[:,:,1]
else:
    sv = sv_raw

mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=X_test_imp.columns).sort_values(ascending=False)
print("Top 10 features:")
for i, (f, v) in enumerate(mean_shap.head(10).items(), 1):
    print(f"  {i:2d}. {f:<40s}  {v:.5f}")






#Fairness Analysis
app_test = app.loc[X_test.index].copy()
app_test['y_true'] = y_test.values
app_test['y_pred'] = y_pred_opt
app_test['y_prob'] = y_prob
app_test['INCOME_BRACKET'] = pd.qcut(
    app_test['AMT_INCOME_TOTAL'], q=4,
    labels=['Low','Middle-Low','Middle-High','High']
)

fairness_dfs = {}
def get_fairness(col):
    rows = []
    for grp in sorted(app_test[col].astype(str).unique()):
        m  = app_test[col].astype(str) == grp
        gt = app_test.loc[m,'y_true']
        gp = app_test.loc[m,'y_pred']
        if gt.sum() < 10:
            continue
        rec = recall_score(gt, gp, zero_division=0)
        tn  = ((gt==0)&(gp==0)).sum()
        fp  = ((gt==0)&(gp==1)).sum()
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0
        rows.append({'group':grp,'n':int(m.sum()),'recall':rec,'fpr':fpr})
    return pd.DataFrame(rows)

fairness_dfs['Gender'] = get_fairness('CODE_GENDER')
fairness_dfs['Family'] = get_fairness('NAME_FAMILY_STATUS')
fairness_dfs['Income'] = get_fairness('INCOME_BRACKET')

for label, df in fairness_dfs.items():
    avg_fpr = df['fpr'].mean()
    print(f"\n  {label}:")
    for _, row in df.iterrows():
        flag = '  ⚠️  bias risk' if row['fpr'] > avg_fpr * 1.5 else ''
        print(f"    {row['group']:<26s}  n={row['n']:>6,}  Recall={row['recall']:.3f}  FPR={row['fpr']:.3f}{flag}")








#Creating Visualizations


# Fig 1: Data Overview
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('Data Overview & Key Features', fontsize=14, fontweight='bold')

vc = y.value_counts()
axes[0].bar(['No Default','Default'], [vc[0],vc[1]], color=[BLUE,RED], edgecolor='white', lw=2)
for rect, cnt in zip(axes[0].patches, [vc[0],vc[1]]):
    axes[0].text(rect.get_x()+rect.get_width()/2, rect.get_height()+500,
                 f'{cnt:,}\n({cnt/len(y)*100:.1f}%)', ha='center', fontsize=9)
axes[0].set_title('Class Imbalance\n(8% default)', fontweight='bold')
axes[0].set_ylim(0, vc[0]*1.15)

for ax, (feat, title, cap) in zip(axes[1:], [
    ('EXT_SOURCE_MEAN', 'External Credit Score Mean\n(strongest predictor)', 1.0),
    ('CREDIT_INCOME_RATIO', 'Credit-to-Income Ratio\n(engineered)', 20),
    ('AGE_YEARS', 'Applicant Age\n(younger = higher risk)', 70)
]):
    d0 = app.loc[app.TARGET==0, feat].clip(upper=cap)
    d1 = app.loc[app.TARGET==1, feat].clip(upper=cap)
    ax.hist(d0, bins=40, alpha=0.65, color=BLUE, label='No Default', density=True)
    ax.hist(d1, bins=40, alpha=0.65, color=RED,  label='Default',    density=True)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(feat); ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig('fig1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig1_data_overview.png")

# Fig 2: SMOTE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('SMOTE: Fixing the Class Imbalance Problem', fontsize=14, fontweight='bold')
for ax, (title, yd) in zip(axes, [
    ('Before SMOTE\n(8% defaults — model ignores them)', y_train),
    ('After SMOTE\n(30% defaults — model learns the pattern)', y_train_sm)
]):
    vc2 = pd.Series(yd).value_counts()
    b = ax.bar(['No Default','Default'], [vc2[0],vc2[1]], color=[BLUE,RED], edgecolor='white', lw=2)
    for bar, cnt in zip(b, [vc2[0],vc2[1]]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f'{cnt:,}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title(title, fontweight='bold'); ax.set_ylabel('Count')
plt.tight_layout()
fig.savefig('fig2_smote.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig2_smote.png")

# Fig 3: Threshold Tuning
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle('Threshold Tuning on Real Validation Data', fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(sweep_df['t'], sweep_df['recall'],    color=RED,   lw=2, label='Recall')
ax.plot(sweep_df['t'], sweep_df['precision'], color=BLUE,  lw=2, label='Precision')
ax.plot(sweep_df['t'], sweep_df['f1'],        color=GREEN, lw=2, label='F1-Score')
ax.axvline(0.5,             color='gray',  linestyle=':',  lw=1.5, label='Default (0.5)')
ax.axvline(best_threshold,  color=ORANGE, linestyle='--', lw=2.5, label=f'Chosen ({best_threshold:.2f})')
ax.axhline(0.60, color=RED, linestyle=':', lw=1, alpha=0.4, label='Target recall 60%')
ax.set_xlabel('Decision Threshold'); ax.set_ylabel('Score')
ax.set_title('Metric vs. Threshold\n(on real validation data)', fontweight='bold')
ax.legend(fontsize=7); ax.set_xlim(0.05, 0.60); ax.set_ylim(0, 1)

ax = axes[1]
p_c, r_c, _ = precision_recall_curve(y_val, val_probs)
ax.plot(r_c, p_c, color='purple', lw=2)
ax.scatter([best_row['recall']], [best_row['precision']], s=150, color=ORANGE, zorder=5,
           label=f'Chosen: R={best_row["recall"]:.2f}, P={best_row["precision"]:.2f}')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision–Recall Tradeoff', fontweight='bold')
ax.legend(); ax.fill_between(r_c, p_c, alpha=0.1, color='purple')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax = axes[2]
mn = ['Recall','ROC-AUC','F1','Precision']
x = np.arange(len(mn)); w = 0.35
ax.bar(x-w/2, [r_050[m] for m in mn], w, label='threshold=0.50', color='#95a5a6', alpha=0.9)
ax.bar(x+w/2, [r_opt[m] for m in mn], w, label=f'threshold={best_threshold:.2f}', color=GREEN, alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(mn)
ax.set_ylim(0, 1); ax.set_ylabel('Score')
ax.set_title('Before vs. After Threshold Tuning', fontweight='bold')
ax.legend(fontsize=9)
for bar in list(ax.patches)[len(mn):]:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{bar.get_height():.2f}', ha='center', fontsize=8, fontweight='bold', color='darkgreen')

plt.tight_layout()
fig.savefig('fig3_threshold_tuning.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig3_threshold_tuning.png")

# Fig 4: Evaluation Dashboard
fig = plt.figure(figsize=(20, 10))
fig.suptitle(f'Final Model Evaluation Dashboard  |  LightGBM + SMOTE + Threshold={best_threshold:.2f}',
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

ax = fig.add_subplot(gs[0,0])
cm = confusion_matrix(y_test, y_pred_opt)
tn2, fp2, fn2, tp2 = cm.ravel()
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
            xticklabels=['Pred: No Default','Pred: Default'],
            yticklabels=['True: No Default','True: Default'])
ax.set_title(f'Confusion Matrix\nTP={tp2:,}  FN={fn2:,}  FP={fp2:,}  TN={tn2:,}', fontweight='bold')

ax = fig.add_subplot(gs[0,1])
fpr_a, tpr_a, _ = roc_curve(y_test, y_prob)
auc_v = roc_auc_score(y_test, y_prob)
ax.plot(fpr_a, tpr_a, color=RED, lw=2.5, label=f'LightGBM AUC={auc_v:.3f}')
ax.plot([0,1],[0,1],'k--',lw=1,label='Random (0.5)')
ax.fill_between(fpr_a, tpr_a, alpha=0.12, color=RED)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve', fontweight='bold'); ax.legend()

ax = fig.add_subplot(gs[0,2])
ax.hist(y_prob[y_test==0], bins=60, alpha=0.65, color=BLUE, density=True, label='No Default')
ax.hist(y_prob[y_test==1], bins=60, alpha=0.65, color=RED,  density=True, label='Default')
ax.axvline(best_threshold, color='black', lw=2, linestyle='--', label=f'Threshold={best_threshold:.2f}')
ax.set_xlabel('Predicted Probability of Default')
ax.set_title('Score Distribution\n(further apart = better model)', fontweight='bold')
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1,0])
mn2 = ['Recall','ROC-AUC','F1','Precision','Accuracy']
mv2 = [r_opt[m] for m in mn2]
cols_m = [GREEN if v>=0.6 else ORANGE if v>=0.4 else RED for v in mv2]
ax.barh(mn2, mv2, color=cols_m, edgecolor='white', lw=1.5)
ax.set_xlim(0,1); ax.axvline(0.5, color='gray', lw=1, linestyle='--')
ax.set_title('Final Metric Scores', fontweight='bold')
for i, (name, val) in enumerate(zip(mn2, mv2)):
    ax.text(val+0.01, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

ax = fig.add_subplot(gs[1,1:])
top15 = mean_shap.head(15)
ax.barh(top15.index[::-1], top15.values[::-1], color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 15)))
ax.set_xlabel('Mean |SHAP Value|')
ax.set_title('Top 15 Features Driving Predictions (SHAP)', fontweight='bold')
for i, val in enumerate(top15.values[::-1]):
    ax.text(val+0.0001, i, f'{val:.4f}', va='center', fontsize=8)

plt.tight_layout()
fig.savefig('fig4_evaluation_dashboard.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig4_evaluation_dashboard.png")

# Fig 5: SHAP Deep Dive
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('SHAP Explainability: Why Did the Model Predict Default?', fontsize=14, fontweight='bold')

top20 = mean_shap.head(20)
axes[0].barh(top20.index[::-1], top20.values[::-1],
             color=plt.cm.coolwarm(np.linspace(0.8, 0.2, 20)))
axes[0].set_xlabel('Mean |SHAP Value|')
axes[0].set_title('Top 20 Most Influential Features', fontweight='bold')

ax = axes[1]
top10 = mean_shap.head(10).index.tolist()
shap_df = pd.DataFrame(sv, columns=X_test_imp.columns)
feat_df = X_shap.reset_index(drop=True)
ymap    = {f: i for i, f in enumerate(top10[::-1])}
for feat in top10:
    svc = shap_df[feat].values
    fvc = feat_df[feat].values
    fn  = (fvc - fvc.min()) / (fvc.max() - fvc.min() + 1e-9)
    sc  = ax.scatter(svc, [ymap[feat]]*len(svc), c=fn, cmap='RdBu_r', alpha=0.35, s=15, vmin=0, vmax=1)
ax.set_yticks(list(ymap.values())); ax.set_yticklabels(list(ymap.keys()), fontsize=9)
ax.axvline(0, color='black', lw=1.5)
ax.set_xlabel('SHAP value  (positive --> default | negative --> no default)')
ax.set_title('SHAP Scatter\nRed = high feature value, Blue = low value', fontweight='bold')
plt.colorbar(sc, ax=ax, label='Feature value (normalised)', shrink=0.6)

plt.tight_layout()
fig.savefig('fig5_shap.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig5_shap.png")

# Fig 6: Fairness
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
fig.suptitle('Fairness Analysis: Is the Model Biased Against Any Group?', fontsize=14, fontweight='bold')

for ax, (label, df_f) in zip(axes, fairness_dfs.items()):
    grps = df_f['group'].tolist()
    recs = df_f['recall'].tolist()
    fprs = df_f['fpr'].tolist()
    x = np.arange(len(grps)); w = 0.35
    ax.bar(x-w/2, recs, w, color=RED,  alpha=0.8, label='Recall (catching defaults)')
    ax.bar(x+w/2, fprs, w, color=BLUE, alpha=0.8, label='FPR (wrongly flagged)')
    ax.set_xticks(x); ax.set_xticklabels(grps, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(0,1); ax.set_title(f'By {label}', fontweight='bold'); ax.legend(fontsize=7)
    for i, (r, f) in enumerate(zip(recs, fprs)):
        ax.text(i-w/2, r+0.01, f'{r:.2f}', ha='center', fontsize=7, color='darkred',  fontweight='bold')
        ax.text(i+w/2, f+0.01, f'{f:.2f}', ha='center', fontsize=7, color='darkblue', fontweight='bold')
    ax.axhline(np.mean(recs), color='darkred',  linestyle='--', lw=1, alpha=0.6)
    ax.axhline(np.mean(fprs), color='darkblue', linestyle='--', lw=1, alpha=0.6)

fig.text(0.5, -0.02,
    "Red = Recall (catching actual defaulters). Blue = False Positive Rate (good applicants wrongly flagged). Dashed lines = group averages.",
    ha='center', fontsize=8, style='italic', color='gray')
plt.tight_layout()
fig.savefig('fig6_fairness.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig6_fairness.png")







#Summary pipeline
print(f"""
  Recall    : {r_opt['Recall']:.3f}  - catches {r_opt['Recall']*100:.0f}% of real defaulters
  ROC-AUC   : {r_opt['ROC-AUC']:.3f}  - {r_opt['ROC-AUC']*100:.1f}% discrimination ability
  Precision : {r_opt['Precision']:.3f}  - {r_opt['Precision']*100:.0f}% of flags are real defaults
  F1-Score  : {r_opt['F1']:.3f}

  For context, this means that for every 100 real defaulters:
    Caught : ~{r_opt['Recall']*100:.0f}
    Missed : ~{(1-r_opt['Recall'])*100:.0f}

  Top predictors (SHAP):""")
for i, (f, v) in enumerate(mean_shap.head(5).items(), 1):
    print(f"    {i}. {f} ({v:.4f})")

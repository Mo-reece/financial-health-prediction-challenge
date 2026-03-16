"""Patch solution.ipynb in-place: fix cells 3, 5, 6 for pandas 3.x / XGBoost 3.x."""
import json

CELL3_NEW = """\
# ── Cell 3: Data Cleaning ─────────────────────────────────────────────────────

# Don't-know canonical token
DONTKNOW = "Dont_know"

# Patterns to unify as DONTKNOW (order matters — longer/more-specific first)
DONTKNOW_VARIANTS = [
    " Do not know / N\\u200e/A",   # leading space + U+200E
    "Don\\u2019t know or N/A",
    "Don't know or N/A",
    "Don\\u2019t know (Do not show)",
    "Don't know (Do not show)",
    "Don?t know / doesn?t apply",
    "Don\\u2019t Know",
    "Don't Know",
    "Don\\u2019t know",
    "Don't know",
]


def _str_cols(df):
    \"\"\"All non-numeric columns (works for both object and pandas 3.x StringDtype).\"\"\"
    return [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c]) and c != 'Target']


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    str_like = _str_cols(df)

    # 3a. Normalise Unicode characters across all string columns
    for col in str_like:
        df[col] = df[col].str.replace('\\u2019', "'", regex=False)
        df[col] = df[col].str.replace('\\u200e', '', regex=False)
        df[col] = df[col].str.strip()

    # 3b. Harmonise all Don't-know variants → DONTKNOW
    dontknow_variants_normalised = [
        v.replace('\\u2019', "'").replace('\\u200e', '').strip()
        for v in DONTKNOW_VARIANTS
    ]
    dontknow_set = set(dontknow_variants_normalised)
    for col in str_like:
        df[col] = df[col].apply(
            lambda x: DONTKNOW if (isinstance(x, str) and x in dontknow_set) else x
        )

    # 3c. Column-specific fixes
    if 'current_problem_cash_flow' in df.columns:
        df['current_problem_cash_flow'] = df['current_problem_cash_flow'].replace({'0': 'No'})

    if 'compliance_income_tax' in df.columns:
        df['compliance_income_tax'] = df['compliance_income_tax'].replace({'Refused': DONTKNOW})

    # 3d. keeps_financial_records → ordinal numeric
    if 'keeps_financial_records' in df.columns:
        kfr_map = {
            'No': 0,
            'Yes': 1,
            'Yes, sometimes': 1,
            'Yes, always': 2,
        }
        df['keeps_financial_records'] = df['keeps_financial_records'].map(kfr_map)
        # NaN stays NaN (was already missing or unknown)

    return df


train = clean_df(train)
test  = clean_df(test)
print('Cleaning done.')
print('current_problem_cash_flow unique:', train['current_problem_cash_flow'].unique())
print('compliance_income_tax unique:', train['compliance_income_tax'].unique())
print('keeps_financial_records unique:', train['keeps_financial_records'].unique())
"""

CELL5_NEW = """\
# ── Cell 5: Encoding & Feature Split ─────────────────────────────────────────

TARGET_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
INV_TARGET = {v: k for k, v in TARGET_MAP.items()}

y = train['Target'].map(TARGET_MAP).values

# Drop ID / Target from features
DROP_COLS = ['ID', 'Target']
feature_cols = [c for c in train.columns if c not in DROP_COLS]

# Identify categorical columns: any non-numeric column.
# pandas 3.x reads CSVs as StringDtype (not object), so dtype == 'object' misses them.
cat_cols_original = [c for c in feature_cols
                     if not pd.api.types.is_numeric_dtype(train[c])]
num_cols = [c for c in feature_cols if c not in cat_cols_original]

print(f'Categorical cols: {len(cat_cols_original)}')
print(f'Numeric cols: {len(num_cols)}')
print(f'Total features: {len(feature_cols)}')

# ── Build CatBoost matrices (raw strings, NaN → 'MISSING') ──────────────────
def prep_cb(df, cols, cat_cols):
    X = df[cols].copy()
    for c in cat_cols:
        X[c] = X[c].fillna('MISSING').astype(str)
    return X

X_cb_full_train = prep_cb(train, feature_cols, cat_cols_original)
X_cb_test       = prep_cb(test,  feature_cols, cat_cols_original)
cat_feature_indices = [X_cb_full_train.columns.get_loc(c) for c in cat_cols_original]

# ── Build XGBoost / LightGBM matrices (label-encoded, NaN preserved) ────────
# Use a sentinel instead of relying on str(nan) which differs between pandas versions.
NA_SENTINEL = '__NA__'

combined = pd.concat([train[feature_cols], test[feature_cols]], axis=0, ignore_index=True)

label_encoders = {}
for col in cat_cols_original:
    le = LabelEncoder()
    combined[col] = combined[col].fillna(NA_SENTINEL).astype(str)
    le.fit(combined[col])
    combined[col] = le.transform(combined[col]).astype(float)
    # Restore sentinel → NaN so XGB/LGB native missing-value handling kicks in
    if NA_SENTINEL in le.classes_:
        na_label = float(le.transform([NA_SENTINEL])[0])
        combined.loc[combined[col] == na_label, col] = np.nan
    label_encoders[col] = le

n_train = len(train)
X_le_train = combined.iloc[:n_train].values.astype(float)
X_le_test  = combined.iloc[n_train:].values.astype(float)

print('Encoding done.')
print(f'X_le_train: {X_le_train.shape}  X_le_test: {X_le_test.shape}')
"""

CELL6_NEW = """\
# ── Cell 6: XGBoost 10-Fold CV ───────────────────────────────────────────────

classes = np.array([0, 1, 2])
class_weights_arr = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = {0: class_weights_arr[0],
                     1: class_weights_arr[1],
                     2: class_weights_arr[2]}

# NOTE: XGBoost ≥3.x requires early_stopping_rounds in the constructor, not fit()
params_xgb = dict(
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1.0,
    n_estimators=2000,
    early_stopping_rounds=100,
    tree_method='hist',
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=SEED,
    verbosity=0,
)

skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

oof_xgb  = np.zeros((n_train, 3))
test_xgb = np.zeros((len(test), 3))

print('Training XGBoost...')
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_le_train, y)):
    X_tr, X_va = X_le_train[tr_idx], X_le_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    sw = np.array([class_weight_dict[yi] for yi in y_tr])

    model = xgb.XGBClassifier(**params_xgb)
    model.fit(
        X_tr, y_tr,
        sample_weight=sw,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    oof_xgb[va_idx]  = model.predict_proba(X_va)
    test_xgb        += model.predict_proba(X_le_test) / N_FOLD

    fold_f1 = f1_score(y_va, oof_xgb[va_idx].argmax(1), average='macro')
    print(f'  Fold {fold+1:2d} | best_iter={model.best_iteration:4d} | F1={fold_f1:.4f}')

xgb_oof_f1 = f1_score(y, oof_xgb.argmax(1), average='macro')
print(f'\\nXGBoost OOF Macro F1: {xgb_oof_f1:.4f}')
"""


def src_to_list(s):
    """Convert a multi-line string into the notebook cell source list format."""
    lines = s.split('\n')
    return [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


nb = json.load(open('solution.ipynb'))

# Map: cell index → new source
patches = {3: CELL3_NEW, 5: CELL5_NEW, 6: CELL6_NEW}

for idx, new_src in patches.items():
    nb['cells'][idx]['source'] = src_to_list(new_src)
    nb['cells'][idx]['outputs'] = []
    nb['cells'][idx]['execution_count'] = None
    print(f'Patched cell {idx}')

with open('solution.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('solution.ipynb written.')

# Verify
nb2 = json.load(open('solution.ipynb'))
for idx in patches:
    src = ''.join(nb2['cells'][idx]['source'])
    issues = []
    if "dtype == 'object'" in src: issues.append('OLD dtype check')
    if "select_dtypes('object')" in src: issues.append('OLD select_dtypes')
    if "le.transform(['nan'])" in src: issues.append('OLD nan transform')
    if issues:
        print(f'Cell {idx} STILL has issues: {issues}')
    else:
        print(f'Cell {idx} OK')

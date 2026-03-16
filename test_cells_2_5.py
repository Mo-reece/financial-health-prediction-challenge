import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings; warnings.filterwarnings('ignore')

train = pd.read_csv('Train.csv')
test  = pd.read_csv('Test.csv')
print(f'Train: {train.shape}  Test: {test.shape}')

# ── Cell 3 (fixed) ────────────────────────────────────────────────────────────
DONTKNOW = 'Dont_know'
DONTKNOW_VARIANTS = [
    ' Do not know / N\u200e/A',
    "Don\u2019t know or N/A", "Don't know or N/A",
    "Don\u2019t know (Do not show)", "Don't know (Do not show)",
    "Don?t know / doesn?t apply",
    "Don\u2019t Know", "Don't Know",
    "Don\u2019t know", "Don't know",
]
def _str_cols(df):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != 'Target']

def clean_df(df):
    df = df.copy()
    for col in _str_cols(df):
        df[col] = df[col].str.replace('\u2019', "'", regex=False)
        df[col] = df[col].str.replace('\u200e', '', regex=False)
        df[col] = df[col].str.strip()
    dontknow_set = set(v.replace('\u2019',"'").replace('\u200e','').strip() for v in DONTKNOW_VARIANTS)
    for col in _str_cols(df):
        df[col] = df[col].apply(lambda x: DONTKNOW if isinstance(x, str) and x in dontknow_set else x)
    if 'current_problem_cash_flow' in df.columns:
        df['current_problem_cash_flow'] = df['current_problem_cash_flow'].replace({'0': 'No'})
    if 'compliance_income_tax' in df.columns:
        df['compliance_income_tax'] = df['compliance_income_tax'].replace({'Refused': DONTKNOW})
    if 'keeps_financial_records' in df.columns:
        df['keeps_financial_records'] = df['keeps_financial_records'].map(
            {'No':0,'Yes':1,'Yes, sometimes':1,'Yes, always':2})
    return df

train = clean_df(train)
test  = clean_df(test)
print('Cleaning done. country dtype:', repr(train['country'].dtype))

# ── Cell 4 ────────────────────────────────────────────────────────────────────
FIN_COLS = ['personal_income','business_expenses','business_turnover']
HAVE_NOW_FIN = ['has_mobile_money','has_credit_card','has_loan_account','has_internet_banking','has_debit_card']
HAVE_NOW_INS = ['motor_vehicle_insurance','medical_insurance','funeral_insurance']

def compute_country_stats(df, cols):
    return {col: df.groupby('country')[col].agg(['median','std']) for col in cols}

def engineer_features(df, country_stats=None, is_train=True, df_train_ref=None):
    df = df.copy()
    df['log_business_turnover'] = np.log1p(df['business_turnover'])
    df['log_personal_income']   = np.log1p(df['personal_income'])
    df['log_business_expenses'] = np.log1p(df['business_expenses'])
    df['expense_to_turnover']   = df['business_expenses'] / (df['business_turnover'] + 1)
    df['profit_margin']         = (df['business_turnover'] - df['business_expenses']) / (df['business_turnover'] + 1)
    df['income_to_expenses']    = df['personal_income'] / (df['business_expenses'] + 1)
    df['turnover_minus_expenses'] = df['business_turnover'] - df['business_expenses']
    if country_stats is None and is_train:
        country_stats = compute_country_stats(df, FIN_COLS)
    for col in FIN_COLS:
        stat = country_stats[col]
        medians = df['country'].map(stat['median'])
        stds    = df['country'].map(stat['std']).replace(0,1).fillna(1)
        df[f'{col}_zscore'] = (df[col] - medians) / stds
        if is_train:
            df[f'{col}_pctrank'] = df.groupby('country')[col].rank(pct=True)
        else:
            for c in df['country'].unique():
                mask_te = df['country'] == c
                if df_train_ref is not None:
                    tv = df_train_ref.loc[df_train_ref['country']==c, col].dropna().values
                    if len(tv) > 0:
                        df.loc[mask_te, f'{col}_pctrank'] = df.loc[mask_te, col].apply(
                            lambda v, t=tv: np.nan if pd.isna(v) else float((t<=v).mean()))
                    else:
                        df.loc[mask_te, f'{col}_pctrank'] = np.nan
    for col in HAVE_NOW_FIN + HAVE_NOW_INS:
        if col in df.columns:
            flag = f'{col}_have_now'
            df[flag] = (df[col] == 'Have now').astype(float)
            df.loc[df[col].isna(), flag] = np.nan
    df['financial_product_count'] = df[[f'{c}_have_now' for c in HAVE_NOW_FIN if f'{c}_have_now' in df.columns]].sum(axis=1, min_count=1)
    df['insurance_product_count'] = df[[f'{c}_have_now' for c in HAVE_NOW_INS if f'{c}_have_now' in df.columns]].sum(axis=1, min_count=1)
    df['total_financial_access']  = df['financial_product_count'].fillna(0) + df['insurance_product_count'].fillna(0)
    def yes_flag(s): return (s=='Yes').astype(float).where(s.notna(), np.nan)
    pos_flags = [yes_flag(df[c]) for c in ['attitude_stable_business_environment','attitude_satisfied_with_achievement','attitude_more_successful_next_year','perception_insurance_important'] if c in df.columns]
    neg_flags = [yes_flag(df[c]) for c in ['attitude_worried_shutdown','perception_cannot_afford_insurance'] if c in df.columns]
    if 'current_problem_cash_flow' in df.columns:
        cpf = df['current_problem_cash_flow']
        neg_flags.append((cpf=='Yes').astype(float).where(cpf.notna(), np.nan))
    if pos_flags: df['positive_attitude_count'] = pd.concat(pos_flags, axis=1).sum(axis=1, min_count=1)
    if neg_flags: df['negative_attitude_count'] = pd.concat(neg_flags, axis=1).sum(axis=1, min_count=1)
    if 'positive_attitude_count' in df.columns and 'negative_attitude_count' in df.columns:
        df['net_attitude'] = df['positive_attitude_count'].fillna(0) - df['negative_attitude_count'].fillna(0)
    hm = ['has_mobile_money','has_credit_card','has_loan_account','has_internet_banking','has_debit_card','medical_insurance','funeral_insurance','uses_friends_family_savings','uses_informal_lender','motor_vehicle_insurance','future_risk_theft_stock','marketing_word_of_mouth','problem_sourcing_money','covid_essential_service','motivation_make_more_money','offers_credit_to_customers','current_problem_cash_flow']
    df['missing_count']    = df[[c for c in hm if c in df.columns]].isna().sum(axis=1)
    df['missing_fraction'] = df.isna().sum(axis=1) / df.shape[1]
    if 'business_age_years' in df.columns:
        bam = df['business_age_months'].fillna(0) if 'business_age_months' in df.columns else 0
        df['total_business_months'] = df['business_age_years'] * 12 + bam
        df['turnover_per_year']     = df['business_turnover'] / (df['business_age_years'] + 1)
        df['age_when_started']      = df['owner_age'] - df['business_age_years']
    if 'has_insurance' in df.columns:
        df['has_insurance_flag'] = (df['has_insurance'] == 'Yes').astype(float)
        df['has_insurance_x_log_turnover'] = df['has_insurance_flag'] * df['log_business_turnover'].fillna(0)
    if 'keeps_financial_records' in df.columns:
        kfr = df['keeps_financial_records']
        df['keeps_records_always'] = (kfr==2).astype(float).where(kfr.notna(), np.nan)
        df['keeps_records_always_x_log_turnover'] = df['keeps_records_always'].fillna(0) * df['log_business_turnover'].fillna(0)
    if 'compliance_income_tax' in df.columns:
        df['compliance_yes'] = (df['compliance_income_tax'] == 'Yes').astype(float)
        if 'keeps_records_always' in df.columns:
            df['compliance_yes_x_keeps_records_always'] = df['compliance_yes'].fillna(0) * df['keeps_records_always'].fillna(0)
    country_enc = {'eswatini':1,'lesotho':2,'malawi':3,'zimbabwe':4}
    df['country_code'] = df['country'].map(country_enc).astype(float)
    df['country_x_financial_product_count'] = df['country_code'] * df['financial_product_count'].fillna(0)
    return df, country_stats

train, country_stats = engineer_features(train, is_train=True)
test,  _             = engineer_features(test, country_stats=country_stats, is_train=False, df_train_ref=train)
print(f'FE done: train {train.shape}  test {test.shape}')

# ── Cell 5 (fixed) ────────────────────────────────────────────────────────────
TARGET_MAP = {'Low':0,'Medium':1,'High':2}
INV_TARGET = {v:k for k,v in TARGET_MAP.items()}
y = train['Target'].map(TARGET_MAP).values
DROP_COLS = ['ID','Target']
feature_cols = [c for c in train.columns if c not in DROP_COLS]
cat_cols_original = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c])]
num_cols = [c for c in feature_cols if c not in cat_cols_original]
print(f'Categorical: {len(cat_cols_original)}  Numeric: {len(num_cols)}  Total: {len(feature_cols)}')

def prep_cb(df, cols, cat_cols):
    X = df[cols].copy()
    for c in cat_cols:
        X[c] = X[c].fillna('MISSING').astype(str)
    return X

X_cb_full_train = prep_cb(train, feature_cols, cat_cols_original)
X_cb_test       = prep_cb(test,  feature_cols, cat_cols_original)
cat_feature_indices = [X_cb_full_train.columns.get_loc(c) for c in cat_cols_original]

NA_SENTINEL = '__NA__'
combined = pd.concat([train[feature_cols], test[feature_cols]], axis=0, ignore_index=True)
label_encoders = {}
for col in cat_cols_original:
    le = LabelEncoder()
    combined[col] = combined[col].fillna(NA_SENTINEL).astype(str)
    le.fit(combined[col])
    combined[col] = le.transform(combined[col]).astype(float)
    if NA_SENTINEL in le.classes_:
        na_label = float(le.transform([NA_SENTINEL])[0])
        combined.loc[combined[col] == na_label, col] = np.nan
    label_encoders[col] = le

n_train = len(train)
X_le_train = combined.iloc[:n_train].values.astype(float)
X_le_test  = combined.iloc[n_train:].values.astype(float)
print('Encoding done.')
print(f'X_le_train: {X_le_train.shape}  X_le_test: {X_le_test.shape}')
print('ALL CELLS 2-5 PASSED')

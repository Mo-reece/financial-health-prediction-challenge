import os, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# Cells 2-3
train = pd.read_csv('Train.csv')
test  = pd.read_csv('Test.csv')

DONTKNOW = 'Dont_know'
DONTKNOW_VARIANTS = [
    ' Do not know / N\u200e/A',
    "Don\u2019t know or N/A", "Don't know or N/A",
    "Don\u2019t know (Do not show)", "Don't know (Do not show)",
    "Don?t know / doesn?t apply",
    "Don\u2019t Know", "Don't Know",
    "Don\u2019t know", "Don't know",
]
def clean_df(df):
    df = df.copy()
    obj_cols = df.select_dtypes('object').columns.tolist()
    for col in obj_cols:
        df[col] = df[col].str.replace('\u2019', "'", regex=False)
        df[col] = df[col].str.replace('\u200e', '', regex=False)
        df[col] = df[col].str.strip()
    dontknow_set = set([v.replace('\u2019',"'").replace('\u200e','').strip() for v in DONTKNOW_VARIANTS])
    for col in obj_cols:
        df[col] = df[col].apply(lambda x: DONTKNOW if (isinstance(x, str) and x in dontknow_set) else x)
    if 'current_problem_cash_flow' in df.columns:
        df['current_problem_cash_flow'] = df['current_problem_cash_flow'].replace({'0': 'No'})
    if 'compliance_income_tax' in df.columns:
        df['compliance_income_tax'] = df['compliance_income_tax'].replace({'Refused': DONTKNOW})
    if 'keeps_financial_records' in df.columns:
        kfr_map = {'No': 0, 'Yes': 1, 'Yes, sometimes': 1, 'Yes, always': 2}
        df['keeps_financial_records'] = df['keeps_financial_records'].map(kfr_map)
    return df

train = clean_df(train)
test  = clean_df(test)

# Cell 4
FIN_COLS = ['personal_income', 'business_expenses', 'business_turnover']
HAVE_NOW_FIN = ['has_mobile_money', 'has_credit_card', 'has_loan_account', 'has_internet_banking', 'has_debit_card']
HAVE_NOW_INS = ['motor_vehicle_insurance', 'medical_insurance', 'funeral_insurance']
POS_ATTITUDE = ['attitude_stable_business_environment','attitude_satisfied_with_achievement','attitude_more_successful_next_year','perception_insurance_important']
NEG_ATTITUDE = ['attitude_worried_shutdown','current_problem_cash_flow','perception_cannot_afford_insurance']
HIGH_MISSING_COLS = ['has_mobile_money','has_credit_card','has_loan_account','has_internet_banking','has_debit_card','medical_insurance','funeral_insurance','uses_friends_family_savings','uses_informal_lender','motor_vehicle_insurance','future_risk_theft_stock','marketing_word_of_mouth','problem_sourcing_money','covid_essential_service','motivation_make_more_money','offers_credit_to_customers','current_problem_cash_flow']

def compute_country_stats(df_train, cols):
    stats = {}
    for col in cols:
        stats[col] = df_train.groupby('country')[col].agg(['median', 'std'])
    return stats

def engineer_features(df, country_stats=None, is_train=True, df_train_ref=None):
    df = df.copy()
    df['expense_to_turnover'] = df['business_expenses'] / (df['business_turnover'] + 1)
    df['profit_margin'] = (df['business_turnover'] - df['business_expenses']) / (df['business_turnover'] + 1)
    df['income_to_expenses'] = df['personal_income'] / (df['business_expenses'] + 1)
    df['turnover_minus_expenses'] = df['business_turnover'] - df['business_expenses']
    df['log_personal_income'] = np.log1p(df['personal_income'])
    df['log_business_expenses'] = np.log1p(df['business_expenses'])
    df['log_business_turnover'] = np.log1p(df['business_turnover'])
    if country_stats is None and is_train:
        country_stats = compute_country_stats(df, FIN_COLS)
    for col in FIN_COLS:
        stat = country_stats[col]
        medians = df['country'].map(stat['median'])
        stds = df['country'].map(stat['std']).replace(0, 1).fillna(1)
        df[f'{col}_zscore'] = (df[col] - medians) / stds
        if is_train:
            df[f'{col}_pctrank'] = df.groupby('country')[col].rank(pct=True)
        else:
            for country in df['country'].unique():
                mask_te = df['country'] == country
                if df_train_ref is not None:
                    mask_tr = df_train_ref['country'] == country
                    train_vals = df_train_ref.loc[mask_tr, col].dropna().values
                    if len(train_vals) > 0:
                        def _pct(v, tv=train_vals):
                            if pd.isna(v): return np.nan
                            return float((tv <= v).mean())
                        df.loc[mask_te, f'{col}_pctrank'] = df.loc[mask_te, col].apply(_pct)
                    else:
                        df.loc[mask_te, f'{col}_pctrank'] = np.nan
                else:
                    df.loc[mask_te, f'{col}_pctrank'] = df.loc[mask_te, col].rank(pct=True)
    for col in HAVE_NOW_FIN + HAVE_NOW_INS:
        if col in df.columns:
            flag_col = f'{col}_have_now'
            df[flag_col] = (df[col] == 'Have now').astype(float)
            df.loc[df[col].isna(), flag_col] = np.nan
    df['financial_product_count'] = df[[f'{c}_have_now' for c in HAVE_NOW_FIN if f'{c}_have_now' in df.columns]].sum(axis=1, min_count=1)
    df['insurance_product_count'] = df[[f'{c}_have_now' for c in HAVE_NOW_INS if f'{c}_have_now' in df.columns]].sum(axis=1, min_count=1)
    df['total_financial_access'] = df['financial_product_count'].fillna(0) + df['insurance_product_count'].fillna(0)
    def yes_flag(series):
        return (series == 'Yes').astype(float).where(series.notna(), np.nan)
    pos_flags = [yes_flag(df[c]) for c in POS_ATTITUDE if c in df.columns]
    neg_flags = [yes_flag(df[c]) for c in NEG_ATTITUDE if c in df.columns]
    if 'current_problem_cash_flow' in df.columns:
        cpf = df['current_problem_cash_flow']
        cpf_flag = (cpf == 'Yes').astype(float).where(cpf.notna(), np.nan)
        neg_flags = [yes_flag(df[c]) for c in NEG_ATTITUDE if c in df.columns and c != 'current_problem_cash_flow']
        neg_flags.append(cpf_flag)
    if pos_flags:
        df['positive_attitude_count'] = pd.concat(pos_flags, axis=1).sum(axis=1, min_count=1)
    if neg_flags:
        df['negative_attitude_count'] = pd.concat(neg_flags, axis=1).sum(axis=1, min_count=1)
    if 'positive_attitude_count' in df.columns and 'negative_attitude_count' in df.columns:
        df['net_attitude'] = df['positive_attitude_count'].fillna(0) - df['negative_attitude_count'].fillna(0)
    existing_hm = [c for c in HIGH_MISSING_COLS if c in df.columns]
    df['missing_count'] = df[existing_hm].isna().sum(axis=1)
    df['missing_fraction'] = df.isna().sum(axis=1) / df.shape[1]
    if 'business_age_years' in df.columns and 'business_age_months' in df.columns:
        df['total_business_months'] = df['business_age_years'] * 12 + df['business_age_months'].fillna(0)
    if 'business_age_years' in df.columns:
        df['turnover_per_year'] = df['business_turnover'] / (df['business_age_years'] + 1)
        df['age_when_started'] = df['owner_age'] - df['business_age_years']
    if 'has_insurance' in df.columns:
        df['has_insurance_flag'] = (df['has_insurance'] == 'Yes').astype(float)
        df['has_insurance_x_log_turnover'] = df['has_insurance_flag'] * df['log_business_turnover'].fillna(0)
    if 'keeps_financial_records' in df.columns:
        kfr = df['keeps_financial_records']
        df['keeps_records_always'] = (kfr == 2).astype(float).where(kfr.notna(), np.nan)
        df['keeps_records_always_x_log_turnover'] = df['keeps_records_always'].fillna(0) * df['log_business_turnover'].fillna(0)
    if 'compliance_income_tax' in df.columns:
        df['compliance_yes'] = (df['compliance_income_tax'] == 'Yes').astype(float)
        if 'keeps_records_always' in df.columns:
            df['compliance_yes_x_keeps_records_always'] = df['compliance_yes'].fillna(0) * df['keeps_records_always'].fillna(0)
    country_enc = {'eswatini': 1, 'lesotho': 2, 'malawi': 3, 'zimbabwe': 4}
    df['country_code'] = df['country'].map(country_enc).astype(float)
    df['country_x_financial_product_count'] = df['country_code'] * df['financial_product_count'].fillna(0)
    return df, country_stats

train, country_stats = engineer_features(train, is_train=True)
test, _ = engineer_features(test, country_stats=country_stats, is_train=False, df_train_ref=train)
print(f'FE done. Train: {train.shape} Test: {test.shape}')

# Cell 5
TARGET_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
INV_TARGET = {v: k for k, v in TARGET_MAP.items()}
y = train['Target'].map(TARGET_MAP).values
DROP_COLS = ['ID', 'Target']
feature_cols = [c for c in train.columns if c not in DROP_COLS]
cat_cols_original = [c for c in feature_cols if train[c].dtype == 'object']
num_cols = [c for c in feature_cols if c not in cat_cols_original]
print(f'Categorical cols: {len(cat_cols_original)}')
print(f'Numeric cols: {len(num_cols)}')
print(f'Total features: {len(feature_cols)}')

def prep_cb(df, cols, cat_cols):
    X = df[cols].copy()
    for c in cat_cols:
        X[c] = X[c].fillna('MISSING').astype(str)
    return X

X_cb_full_train = prep_cb(train, feature_cols, cat_cols_original)
X_cb_test       = prep_cb(test,  feature_cols, cat_cols_original)
cat_feature_indices = [X_cb_full_train.columns.get_loc(c) for c in cat_cols_original]

combined = pd.concat([train[feature_cols], test[feature_cols]], axis=0, ignore_index=True)
label_encoders = {}
for col in cat_cols_original:
    le = LabelEncoder()
    combined[col] = combined[col].astype(str)
    le.fit(combined[col])
    combined[col] = le.transform(combined[col]).astype(float)
    nan_label = le.transform(['nan'])[0]
    combined.loc[combined[col] == nan_label, col] = np.nan
    label_encoders[col] = le

n_train = len(train)
X_le_train = combined.iloc[:n_train].values.astype(float)
X_le_test  = combined.iloc[n_train:].values.astype(float)
print('Encoding done.')
print(f'X_le_train: {X_le_train.shape}  X_le_test: {X_le_test.shape}')

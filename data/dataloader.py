"""
Data loading and preprocessing functions for the multi-agent dropout prediction system
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def load_uci_timeseries(file_path: str) -> pd.DataFrame:
    """
    Loads the UCI Student Dropout dataset and converts it into a time-series
    format, keeping as many relevant features as possible. This version is
    robust against minor differences in column names and includes essential checks.
    """
    from utils.logger import log
    
    log(f"Loading and processing UCI dataset from: {file_path}")
    try:
        df_uci = pd.read_csv(file_path)
    except FileNotFoundError:
        log(f"Error: UCI dataset file not found at '{file_path}'. Please check the path.")
        return pd.DataFrame()

    # --- Pre-computation check for the Target column ---
    if 'Target' not in df_uci.columns:
        log("--> FATAL ERROR: The essential 'Target' column was not found in your CSV file.")
        log(f"--> Available columns are: {df_uci.columns.tolist()}")
        log("--> Please ensure your CSV file contains the 'Target' column.")
        return pd.DataFrame() # Stop execution

    # Define a precise mapping from the exact original column names to clean names.
    column_mapping = {
        'Marital status': 'marital_status',
        'Application mode': 'application_mode',
        'Application order': 'application_order',
        'Course': 'course',
        'Daytime/evening attendance': 'daytime_evening_attendance',
        'Previous qualification': 'previous_qualification',
        'Previous qualification (grade)': 'previous_qualification_grade',
        'Nacionality': 'nationality',
        'Nationality': 'nationality',
        "Mother's qualification": 'mothers_qualification',
        "Father's qualification": 'fathers_qualification',
        "Mother's occupation": 'mothers_occupation',
        "Father's occupation": 'fathers_occupation',
        'Admission grade': 'admission_grade',
        'Displaced': 'displaced',
        'Educational special needs': 'educational_special_needs',
        'Debtor': 'debtor',
        'Tuition fees up to date': 'tuition_fees_up_to_date',
        'Gender': 'gender',
        'Scholarship holder': 'scholarship_holder',
        'Age at enrollment': 'age_at_enrollment',
        'International': 'international',
        'Curricular units 1st sem (credited)': 'cu_1st_sem_credited',
        'Curricular units 1st sem (enrolled)': 'cu_1st_sem_enrolled',
        'Curricular units 1st sem (evaluations)': 'cu_1st_sem_evaluations',
        'Curricular units 1st sem (approved)': 'cu_1st_sem_approved',
        'Curricular units 1st sem (grade)': 'cu_1st_sem_grade',
        'Curricular units 1st sem (without evaluations)': 'cu_1st_sem_without_evals',
        'Curricular units 2nd sem (credited)': 'cu_2nd_sem_credited',
        'Curricular units 2nd sem (enrolled)': 'cu_2nd_sem_enrolled',
        'Curricular units 2nd sem (evaluations)': 'cu_2nd_sem_evaluations',
        'Curricular units 2nd sem (approved)': 'cu_2nd_sem_approved',
        'Curricular units 2nd sem (grade)': 'cu_2nd_sem_grade',
        'Curricular units 2nd sem (without evaluations)': 'cu_2nd_sem_without_evals',
        'Unemployment rate': 'unemployment_rate',
        'Inflation rate': 'inflation_rate',
        'GDP': 'gdp',
        'Target': 'target'
    }

    # Filter the mapping to only include columns that actually exist in the loaded CSV
    existing_columns = {k: v for k, v in column_mapping.items() if k in df_uci.columns}
    df_uci = df_uci.rename(columns=existing_columns)


    # Create a unique student ID
    df_uci['student_id'] = df_uci.index

    # Define which columns are static based on their new, clean names
    static_cols = [
        'student_id', 'marital_status', 'application_mode', 'application_order', 'course',
        'daytime_evening_attendance', 'previous_qualification', 'previous_qualification_grade',
        'nationality', 'mothers_qualification', 'fathers_qualification', 'mothers_occupation',
        'fathers_occupation', 'admission_grade', 'displaced', 'educational_special_needs',
        'debtor', 'tuition_fees_up_to_date', 'gender', 'scholarship_holder',
        'age_at_enrollment', 'international', 'unemployment_rate', 'inflation_rate', 'gdp'
    ]
    # Ensure all defined static columns actually exist in the dataframe before using them
    static_cols = [col for col in static_cols if col in df_uci.columns]


    # Map semester-specific columns to generic names
    semester_map = {
        '1st': {
            'cu_1st_sem_credited': 'sem_credited',
            'cu_1st_sem_enrolled': 'sem_enrolled',
            'cu_1st_sem_evaluations': 'sem_evaluations',
            'cu_1st_sem_approved': 'sem_approved',
            'cu_1st_sem_grade': 'sem_grade',
            'cu_1st_sem_without_evals': 'sem_without_evals'
        },
        '2nd': {
            'cu_2nd_sem_credited': 'sem_credited',
            'cu_2nd_sem_enrolled': 'sem_enrolled',
            'cu_2nd_sem_evaluations': 'sem_evaluations',
            'cu_2nd_sem_approved': 'sem_approved',
            'cu_2nd_sem_grade': 'sem_grade',
            'cu_2nd_sem_without_evals': 'sem_without_evals'
        }
    }

    timeseries_data = []

    for _, row in tqdm(df_uci.iterrows(), desc="▸ UCI Student TS", total=len(df_uci)):
        static_features = row[static_cols].to_dict()
        final_result = row['target']
        dropout_status = 1 if final_result == 'Dropout' else 0

        # --- Snapshot 1: End of 1st Semester ---
        snap1 = static_features.copy()
        for uci_col, generic_col in semester_map['1st'].items():
            if uci_col in row:
                snap1[generic_col] = row[uci_col]
        snap1['snapshot_day'] = 90
        snap1['dropout'] = dropout_status
        snap1['sem_grade_trend'] = 0 # No trend yet
        timeseries_data.append(snap1)

        # --- Snapshot 2: End of 2nd Semester ---
        snap2 = static_features.copy()
        for uci_col, generic_col in semester_map['2nd'].items():
            if uci_col in row:
                snap2[generic_col] = row[uci_col]
        snap2['snapshot_day'] = 180
        snap2['dropout'] = dropout_status
        # Calculate trend from 1st to 2nd semester grade
        if 'sem_grade' in snap1 and 'sem_grade' in snap2:
             snap2['sem_grade_trend'] = snap2['sem_grade'] - snap1['sem_grade']
        else:
            snap2['sem_grade_trend'] = 0
        timeseries_data.append(snap2)

    df_ts = pd.DataFrame(timeseries_data)

    # Add placeholder columns for compatibility with agents that need them
    placeholder_cols = [
        'clicks_total', 'clicks_mean', 'clicks_std', 'clicks_trend', 'clicks_volatility',
        'clicks_recent_vs_early', 'engagement_consistency', 'active_days', 'max_daily_clicks',
        'days_since_last_activity', 'assessments_completed', 'assessment_score_mean',
        'assessment_score_trend', 'assessments_on_time', 'assessment_submission_delay_mean',
        'days_into_course', 'is_early_stage', 'is_mid_stage', 'is_late_stage'
    ]
    for col in placeholder_cols:
        if col not in df_ts.columns:
            df_ts[col] = 0

    log(f"Created rich UCI time-series dataset: {len(df_ts)} snapshots for {df_uci['student_id'].nunique()} learners")
    return df_ts


def load_oulad_timeseries(folder_path: str, snapshot_days: list) -> pd.DataFrame:
    """
    Load and process OULAD data into time series format
    Creates temporal features for each student at multiple time points
    """
    from utils.logger import log
    from data.timeseries_features import create_temporal_features
    
    log("Loading OULAD data for time series processing...")

    # Load all OULAD tables (assuming they exist as separate CSVs)
    try:
        assessments = pd.read_csv(os.path.join(folder_path, "assessments.csv"))
        courses = pd.read_csv(os.path.join(folder_path,"courses.csv"))
        student_info = pd.read_csv(os.path.join(folder_path, "studentInfo.csv"))
        student_registration = pd.read_csv(os.path.join(folder_path,"studentRegistration.csv"))
        student_assessment = pd.read_csv(os.path.join(folder_path,"studentAssessment.csv"))
        student_vle = pd.read_csv(os.path.join(folder_path, "studentVle.csv"))
        vle = pd.read_csv(os.path.join(folder_path,"vle.csv"))
        log("Successfully loaded all OULAD tables")
    except FileNotFoundError as e:
        log(f"Error loading OULAD files: {e}")
        # Fallback to combined file if individual files don't exist
        log("Falling back to combined CSV file...")
        combined_df = pd.read_csv(os.path.join("raw_datasets", "OULAD", "oulad_all_combined.csv"))
        return process_combined_timeseries(combined_df, snapshot_days)

    # Create time series dataset
    timeseries_data = []

    # Get unique student-presentation combinations
    student_presentations = student_info[['id_student', 'code_module', 'code_presentation']].drop_duplicates()

    for _, row in tqdm(
            student_presentations.iterrows(),
            desc="▸ Student TS",
            total=len(student_presentations),
            unit="stud"
        ):
        student_id = row['id_student']
        module = row['code_module']
        presentation = row['code_presentation']

        # Get student static info
        student_static = student_info[
            (student_info['id_student'] == student_id) &
            (student_info['code_module'] == module) &
            (student_info['code_presentation'] == presentation)
        ].iloc[0]

        # Get course length
        course_info = courses[
            (courses['code_module'] == module) &
            (courses['code_presentation'] == presentation)
        ]
        if len(course_info) == 0:
            continue
        course_length = course_info.iloc[0]['module_presentation_length']

        # Create time series for multiple snapshot days
        for snapshot_day in snapshot_days:
            if snapshot_day > course_length:
                continue

            # Create temporal features up to snapshot_day
            temporal_features = create_temporal_features(
                student_id, module, presentation, snapshot_day,
                student_vle, student_assessment, assessments, vle
            )

            # Combine with static features
            record = {
                'student_id': student_id,
                'code_module': module,
                'code_presentation': presentation,
                'snapshot_day': snapshot_day,
                'time_remaining': course_length - snapshot_day,

                # Static features (encoded)
                'gender': student_static['gender'],
                'region': student_static['region'],
                'highest_education': student_static['highest_education'],
                'imd_band': student_static['imd_band'],
                'age_band': student_static['age_band'],
                'num_of_prev_attempts': student_static['num_of_prev_attempts'],
                'studied_credits': student_static['studied_credits'],
                'disability': student_static['disability'],

                # Handle missing 'date_registration' safely
                'date_registration': student_static['date_registration'] if 'date_registration' in student_static else None,

                # Target (dropout within next PRED_HORIZON days)
                'final_result': student_static['final_result'],
                'dropout': 1 if student_static['final_result'] in ['Withdrawn', 'Fail'] else 0
            }

            # Add temporal features
            record.update(temporal_features)
            timeseries_data.append(record)


    df_ts = pd.DataFrame(timeseries_data)
    log(f"Created time series dataset: {len(df_ts)} records from {len(student_presentations)} students")
    print(f"Created time series dataset: {len(df_ts)} records from {len(student_presentations)} students")

    return df_ts


def load_xuetangx_timeseries(train_path: str,
                             test_path: str,
                             user_info_path: str,
                             snapshot_days=(7, 14, 21, 30, 45, 60),
                             time_window=30) -> pd.DataFrame:
    """
    Load and process XuetangX data into time series format
    """
    from utils.logger import log
    from data.timeseries_features import calculate_trend
    
    log("Loading XuetangX data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logs = pd.concat([train_df, test_df], ignore_index=True)
    profiles = pd.read_csv(user_info_path)

    # --- 1. Tidy timestamps ---
    logs["timestamp"] = pd.to_datetime(logs["timestamp"])
    logs["day"]       = (logs["timestamp"].dt.floor("D") - 
                         logs.groupby("username")["timestamp"].transform("min").dt.floor("D")).dt.days + 1

    # --- 2. Total clicks per row ---
    action_cols = [c for c in logs.columns if c.startswith("action_")]
    logs["row_clicks"] = logs[action_cols].sum(axis=1)

    # --- 3. Build snapshot records ---
    records = []
    for (user, course), grp in tqdm(logs.groupby(["username", "course_id"]),
                                    desc="▸ XuetangX students", unit="stud"):
        
        for snap in snapshot_days:
            # Use cumulative data up to the snapshot day
            sub = grp[grp['day'] <= snap]
            if sub.empty:
                continue

            # --- Feature calculation on the cumulative data ---
            daily_clicks = sub.groupby('day')['row_clicks'].sum().reindex(range(1, snap + 1), fill_value=0)
            
            clicks_total = daily_clicks.sum()
            clicks_mean  = daily_clicks.mean()
            clicks_std   = daily_clicks.std()

            rec = dict(
                student_id=user, snapshot_day=snap, days_into_course=snap,
                is_early_stage=int(snap <= 14), is_mid_stage=int(14 < snap <= 45), is_late_stage=int(snap > 45),
                clicks_total=clicks_total, clicks_mean=clicks_mean, clicks_std=clicks_std,
                clicks_trend=calculate_trend(daily_clicks), clicks_volatility=clicks_std / (clicks_mean + 1e-6),
                clicks_recent_vs_early=daily_clicks.tail(7).mean() / (daily_clicks.head(7).mean() + 1e-6),
                engagement_consistency=1 - clicks_std / (clicks_mean + 1e-6),
                active_days=int((daily_clicks > 0).sum()), max_daily_clicks=int(daily_clicks.max()),
                days_since_last_activity=int(snap - sub["day"].max())
            )

            # --- Assessment placeholders ---
            rec.update(dict(
                assessments_completed=0, assessment_score_mean=0, assessment_score_trend=0,
                assessments_on_time=0, assessment_submission_delay_mean=0, banked_assessments=0
            ))

            # --- Static profile merge ---
            prof_query = profiles["username"] == user if "username" in profiles.columns else profiles["user_id"] == user
            prof = profiles.loc[prof_query].head(1)
            if len(prof):
                prof_row = prof.iloc[0]
                rec.update({k: prof_row.get(k, "unknown") for k in ["gender", "age_band", "highest_education", "imd_band"]})
                rec.update({k: prof_row.get(k, "0") for k in ["disability", "num_of_prev_attempts"]})
                rec.update({"studied_credits": prof_row.get("credits", 0), "date_registration": 0})
            else:
                 rec.update(dict(gender="unknown", age_band="unknown", highest_education="unknown", imd_band="unknown",
                    disability="0", num_of_prev_attempts=0, studied_credits=0, date_registration=0))
            
            # --- Target label ---
            truth_now = sub["truth"].max() if "truth" in sub else 0
            rec["dropout"] = int(truth_now == 1)
            records.append(rec)

    if not records:
        raise ValueError("FATAL: Still no records generated. Please double-check the 'day' calculation and snapshot days.")

    df_ts = pd.DataFrame(records)
    log(f"Created XuetangX time-series dataset: {len(df_ts)} snapshots for {df_ts['student_id'].nunique()} learners")
    return df_ts


def process_combined_timeseries(df_combined, snapshot_days=[7, 14, 21, 30, 45, 60]):
    """
    Process combined CSV into time series format (fallback method)
    """
    from utils.logger import log
    from data.timeseries_features import calculate_trend
    
    log("Processing combined CSV for time series...")

    # Assume combined CSV has daily interaction columns (day_0, day_1, etc.)
    day_columns = [col for col in df_combined.columns if col.startswith('day_')]

    timeseries_data = []

    for _, student_row in tqdm(df_combined.iterrows(), desc="Processing combined data", total=len(df_combined), unit="stud"):
        student_id = student_row.get('student_id', student_row.get('id_student'))

        for snapshot_day in snapshot_days:
            if snapshot_day >= len(day_columns):
                continue

            # Get clicks up to snapshot day
            clicks_history = student_row[day_columns[:snapshot_day + 1]].values

            # Create temporal features
            temporal_features = {
                'clicks_total': np.sum(clicks_history),
                'clicks_mean': np.mean(clicks_history),
                'clicks_std': np.std(clicks_history),
                'clicks_trend': calculate_trend(pd.Series(clicks_history)),
                'clicks_volatility': np.std(clicks_history) / (np.mean(clicks_history) + 1),
                'clicks_recent_vs_early': (np.mean(clicks_history[-7:]) / (np.mean(clicks_history[:7]) + 1)) if len(clicks_history) >= 14 else 1,
                'active_days': np.sum(clicks_history > 0),
                'max_daily_clicks': np.max(clicks_history),
                'days_since_last_activity': len(clicks_history) - np.max(np.where(clicks_history > 0)[0]) - 1 if np.any(clicks_history > 0) else len(clicks_history)
            }

            # Combine with static features
            record = {
                'student_id': student_id,
                'snapshot_day': snapshot_day,
                'dropout': 1 if student_row.get('final_result') in ['Withdrawn', 'Fail'] else 0
            }

            # Add static features
            static_features = ['gender', 'age_band', 'imd_band', 'highest_education',
                             'disability', 'num_of_prev_attempts', 'date_registration']
            for feat in static_features:
                if feat in student_row:
                    record[feat] = student_row[feat]

            # Add temporal features
            record.update(temporal_features)
            timeseries_data.append(record)
    print("✅ Time series data processed combined.")
    return pd.DataFrame(timeseries_data)


def encode_categorical_features(df_ts, dataset_type='oulad'):
    """
    Encode categorical variables for time series data
    """
    from utils.logger import log
    
    if dataset_type == 'uci':
        categorical_cols = [
            'gender', 'marital_status', 'course', 'daytime_evening_attendance', 'previous_qualification',
            'nationality', 'mothers_qualification', 'fathers_qualification', 'mothers_occupation',
            'fathers_occupation', 'displaced', 'educational_special_needs', 'debtor',
            'tuition_fees_up_to_date', 'scholarship_holder', 'international',
            'region', 'highest_education', 'imd_band', 'age_band', 'disability'
        ]
    else:
        categorical_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    
    for col in categorical_cols:
        if col in df_ts.columns:
            df_ts[col] = LabelEncoder().fit_transform(df_ts[col].astype(str))
    
    return df_ts


def prepare_features_and_target(df_ts, dataset_type='oulad', use_uci=False, use_xuetangx=False):
    """
    Prepare features and target variables from the time series data
    """
    from utils.logger import log
    
    if use_uci:
        # ───────── Create features and target
        log("Separating features and target variable...")

        # Define columns that should NEVER be used as features
        NON_FEATURE_COLS = ['dropout', 'student_id', 'target', 'final_result']

        # Select all columns that are not in our non-feature list
        feature_cols = [col for col in df_ts.columns if col not in NON_FEATURE_COLS]

        # Create the feature matrix X_ts and target vector y_ts
        # This ensures all relevant columns are kept and are converted to numeric types
        X_ts = df_ts[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        y_ts = df_ts['dropout']
        idx_ts = df_ts.index.to_numpy()
    else:
        feature_cols = [
            # Static features
            'gender', 'imd_band', 'highest_education', 'age_band', 'disability',
            'num_of_prev_attempts', 'date_registration', 'studied_credits',

            # Temporal features
            'clicks_total', 'clicks_mean', 'clicks_std', 'clicks_trend', 'clicks_volatility',
            'clicks_recent_vs_early', 'engagement_consistency', 'active_days', 'max_daily_clicks',
            'days_since_last_activity', 'assessments_completed', 'assessment_score_mean',
            'assessment_score_trend', 'assessments_on_time', 'assessment_submission_delay_mean',
            'days_into_course', 'is_early_stage', 'is_mid_stage', 'is_late_stage'
        ]

        # Filter available columns
        available_cols = [col for col in feature_cols if col in df_ts.columns]
        X_ts = df_ts[available_cols].fillna(0)
        y_ts = df_ts['dropout']
        idx_ts = df_ts.index.to_numpy()
    
    return X_ts, y_ts, idx_ts
"""
Script to combine individual OULAD CSV files into a single combined file
"""
import pandas as pd
import os
import argparse
from pathlib import Path

def combine_oulad_data(input_dir, output_file):
    """
    Combine OULAD data files into a single CSV file
    """
    print(f"Loading OULAD data from: {input_dir}")
    
    # Define required files
    required_files = [
        'assessments.csv',
        'courses.csv', 
        'studentInfo.csv',
        'studentRegistration.csv',
        'studentAssessment.csv',
        'studentVle.csv',
        'vle.csv'
    ]
    
    # Check if all required files exist
    for file in required_files:
        file_path = os.path.join(input_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Load all data files
    assessments = pd.read_csv(os.path.join(input_dir, 'assessments.csv'))
    courses = pd.read_csv(os.path.join(input_dir, 'courses.csv'))
    student_info = pd.read_csv(os.path.join(input_dir, 'studentInfo.csv'))
    student_registration = pd.read_csv(os.path.join(input_dir, 'studentRegistration.csv'))
    student_assessment = pd.read_csv(os.path.join(input_dir, 'studentAssessment.csv'))
    student_vle = pd.read_csv(os.path.join(input_dir, 'studentVle.csv'))
    vle = pd.read_csv(os.path.join(input_dir, 'vle.csv'))
    
    print("All OULAD data files loaded successfully")
    print(f"Student Info: {len(student_info)} records")
    print(f"Assessments: {len(assessments)} records")
    print(f"Student Assessments: {len(student_assessment)} records")
    print(f"Student VLE: {len(student_vle)} records")
    
    # Join student_assessment with assessments to get code_module and code_presentation
    student_assessment_full = student_assessment.merge(assessments[['id_assessment', 'code_module', 'code_presentation']], 
                                                      on='id_assessment', how='left')
    
    # Create a summary of early VLE activity for each student
    early_vle_activity = student_vle[student_vle['date'] <= 7].groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index()
    early_vle_activity.rename(columns={'sum_click': 'early_clicks'}, inplace=True)
    
    # Calculate average score for each student
    assessment_scores = student_assessment_full.groupby('id_student').agg({
        'score': ['mean', 'count']
    }).reset_index()
    assessment_scores.columns = ['id_student', 'early_score', 'assessment_count']
    
    # Create a summary of VLE activity types for each student
    vle_summary = student_vle.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].agg([
        'sum', 'mean', 'std', 'max', 'count'
    ]).reset_index()
    vle_summary.columns = [
        'id_student', 'code_module', 'code_presentation',
        'clicks_total', 'clicks_mean', 'clicks_std', 
        'max_daily_clicks', 'active_days'
    ]
    
    # Count different types of VLE activities
    activity_counts = student_vle.merge(vle[['id_site', 'activity_type']], left_on='id_site', right_on='id_site', how='left')
    activity_summary = activity_counts.groupby(['id_student', 'code_module', 'code_presentation'])['activity_type'].value_counts().unstack(fill_value=0).reset_index()
    
    # Create assessments summary for each student
    assessment_summary = student_assessment_full.groupby(['id_student', 'code_module', 'code_presentation']).agg({
        'id_assessment': 'count',
        'score': 'mean'
    }).reset_index()
    assessment_summary.columns = [
        'id_student', 'code_module', 'code_presentation',
        'assessments_completed', 'assessment_score_mean'
    ]
    
    # Combine all data using student_info as the main table
    combined_df = student_info.merge(courses, on=['code_module', 'code_presentation'], how='left')
    combined_df = combined_df.merge(student_registration, on=['code_module', 'code_presentation', 'id_student'], how='left')
    combined_df = combined_df.merge(vle_summary, on=['code_module', 'code_presentation', 'id_student'], how='left')
    combined_df = combined_df.merge(early_vle_activity, on=['code_module', 'code_presentation', 'id_student'], how='left', suffixes=('', '_early'))
    combined_df = combined_df.merge(assessment_scores, on=['id_student'], how='left')
    combined_df = combined_df.merge(assessment_summary, on=['code_module', 'code_presentation', 'id_student'], how='left', suffixes=('', '_assess'))
    combined_df = combined_df.merge(activity_summary, on=['code_module', 'code_presentation', 'id_student'], how='left', suffixes=('', '_activity'))
    
    # Fill NaN values with defaults
    combined_df = combined_df.fillna(0)
    
    # Create a dropout column based on final_result
    combined_df['dropout'] = combined_df['final_result'].apply(lambda x: 1 if x in ['Withdrawn', 'Fail'] else 0)
    
    # Ensure all required columns exist
    required_columns = [
        'code_module', 'code_presentation', 'id_student', 'gender', 'region',
        'highest_education', 'imd_band', 'age_band', 'num_of_prev_attempts',
        'studied_credits', 'disability', 'final_result', 'dropout',
        'date_registration', 'date_unregistration', 'clicks_total', 'clicks_mean',
        'clicks_std', 'max_daily_clicks', 'active_days', 'early_clicks', 'early_score',
        'assessments_completed', 'assessment_score_mean', 'module_presentation_length'
    ]
    
    # Add any missing required columns with default values
    for col in required_columns:
        if col not in combined_df.columns:
            print(f"Adding missing column: {col}")
            combined_df[col] = 0
            
    # Reorder columns to have key ones first
    available_columns = [col for col in required_columns if col in combined_df.columns]
    other_columns = [col for col in combined_df.columns if col not in required_columns]
    final_columns = available_columns + other_columns
    combined_df = combined_df[final_columns]
    
    # Save the combined data
    combined_df.to_csv(output_file, index=False)
    print(f"Combined OULAD data saved to: {output_file}")
    print(f"Total records: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")

def main():
    parser = argparse.ArgumentParser(description='Combine OULAD data files into a single CSV')
    parser.add_argument('--input_dir', type=str, default='raw_datasets/OULAD/data',
                        help='Path to directory containing individual OULAD CSV files')
    parser.add_argument('--output_file', type=str, default='raw_datasets/OULAD/oulad_all_combined.csv',
                        help='Path to output combined CSV file')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    combine_oulad_data(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
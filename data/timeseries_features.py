"""
Time series feature engineering functions for the multi-agent dropout prediction system
"""
import pandas as pd
import numpy as np


def calculate_trend(series):
    """Calculate linear trend of a time series"""
    if len(series) < 2:
        return 0
    x = np.arange(len(series))
    y = series.values
    if np.std(x) == 0 or np.std(y) == 0:
        return 0
    return np.corrcoef(x, y)[0, 1]


def create_temporal_features(student_id, module, presentation, snapshot_day,
                           student_vle, student_assessment, assessments, vle):
    """
    Create temporal features for a student up to snapshot_day
    """
    features = {}

    # 1. VLE INTERACTION TIME SERIES
    vle_data = student_vle[
        (student_vle['id_student'] == student_id) &
        (student_vle['code_module'] == module) &
        (student_vle['code_presentation'] == presentation) &
        (student_vle['date'] <= snapshot_day)
    ]

    if len(vle_data) > 0:
        # Daily click patterns (time series features)
        daily_clicks = vle_data.groupby('date')['sum_click'].sum().reindex(
            range(max(1, snapshot_day - 30 + 1), snapshot_day + 1), fill_value=0
        )

        # Time series statistical features
        features.update({
            'clicks_total': vle_data['sum_click'].sum(),
            'clicks_mean': daily_clicks.mean(),
            'clicks_std': daily_clicks.std(),
            'clicks_trend': calculate_trend(daily_clicks),
            'clicks_volatility': daily_clicks.std() / (daily_clicks.mean() + 1),
            'clicks_recent_vs_early': daily_clicks.tail(7).mean() / (daily_clicks.head(7).mean() + 1),
            'engagement_consistency': 1 - (daily_clicks.std() / (daily_clicks.mean() + 1)),
            'active_days': (daily_clicks > 0).sum(),
            'max_daily_clicks': daily_clicks.max(),
            'days_since_last_activity': max(0, snapshot_day - vle_data['date'].max()) if len(vle_data) > 0 else snapshot_day
        })

        # Activity type patterns
        vle_with_type = vle_data.merge(vle, on=['id_site'])
        activity_patterns = vle_with_type.groupby('activity_type')['sum_click'].sum()
        for activity_type in ['homepage', 'content', 'resource', 'url', 'forumng', 'quiz']:
            features[f'clicks_{activity_type}'] = activity_patterns.get(activity_type, 0)

        # Weekly patterns
        for week in range(1, min(5, snapshot_day // 7 + 1)):
            week_start = (week - 1) * 7 + 1
            week_end = min(week * 7, snapshot_day)
            week_clicks = vle_data[
                (vle_data['date'] >= week_start) & (vle_data['date'] <= week_end)
            ]['sum_click'].sum()
            features[f'clicks_week_{week}'] = week_clicks
    else:
        # Fill with zeros if no VLE data
        for key in ['clicks_total', 'clicks_mean', 'clicks_std', 'clicks_trend',
                   'clicks_volatility', 'clicks_recent_vs_early', 'engagement_consistency',
                   'active_days', 'max_daily_clicks', 'days_since_last_activity']:
            features[key] = 0
        for activity_type in ['homepage', 'content', 'resource', 'url', 'forumng', 'quiz']:
            features[f'clicks_{activity_type}'] = 0
        for week in range(1, 5):
            features[f'clicks_week_{week}'] = 0

    # 2. ASSESSMENT TIME SERIES
    assessment_data = student_assessment[
        (student_assessment['id_student'] == student_id)
    ]

    # Get assessments for this module/presentation up to snapshot_day
    module_assessments = assessments[
        (assessments['code_module'] == module) &
        (assessments['code_presentation'] == presentation) &
        (assessments['date'] <= snapshot_day)
    ]

    if len(assessment_data) > 0 and len(module_assessments) > 0:
        completed_assessments = assessment_data[
            assessment_data['id_assessment'].isin(module_assessments['id_assessment'])
        ]

        if len(completed_assessments) > 0:
            # Merge completed assessments with module assessments
            # Ensure we rename the 'date' column in module_assessments explicitly before merge
            module_assessments = module_assessments.rename(columns={'date': 'date_module'})

            # Merge the completed assessments with the module assessments
            merged_assessments = completed_assessments.merge(
                module_assessments, on='id_assessment', suffixes=('_completed', '_module')
            )

            # Print columns after merge to check the column names
            # print("Merged assessments columns:", merged_assessments.columns)

            # Now you can safely use 'date_module' in the comparison if it exists
            if 'date_module' in merged_assessments.columns:
                assessments_on_time = (merged_assessments['date_submitted'] <= merged_assessments['date_module']).sum()
            else:
                # print("Error: 'date_module' column is missing in the merged DataFrame.")
                assessments_on_time = 0  # Handle the missing column case

            # Update the calculation of 'assessment_submission_delay_mean'
            features.update({
                'assessments_completed': len(completed_assessments),
                'assessment_score_mean': completed_assessments['score'].mean(),
                'assessment_score_trend': calculate_trend(completed_assessments.sort_values('date_submitted')['score']),
                'assessments_on_time': assessments_on_time,
                'assessment_submission_delay_mean': completed_assessments.apply(
                    lambda x: max(0, x['date_submitted'] - module_assessments[
                        module_assessments['id_assessment'] == x['id_assessment']
                    ]['date_module'].iloc[0]) if len(module_assessments[
                        module_assessments['id_assessment'] == x['id_assessment']
                    ]) > 0 else 0, axis=1
                ).mean(),
                'banked_assessments': completed_assessments['is_banked'].sum()
            })
        else:
            # If no completed assessments, set the features to 0
            for key in ['assessments_completed', 'assessment_score_mean', 'assessment_score_trend',
                       'assessments_on_time', 'assessment_submission_delay_mean', 'banked_assessments']:
                features[key] = 0
    else:
        # If no assessment data or no module assessments, set the features to 0
        for key in ['assessments_completed', 'assessment_score_mean', 'assessment_score_trend',
                   'assessments_on_time', 'assessment_submission_delay_mean', 'banked_assessments']:
            features[key] = 0

    # 3. TEMPORAL CONTEXTUAL FEATURES
    features.update({
        'days_into_course': snapshot_day,
        'is_early_stage': 1 if snapshot_day <= 14 else 0,
        'is_mid_stage': 1 if 14 < snapshot_day <= 45 else 0,
        'is_late_stage': 1 if snapshot_day > 45 else 0,
        'week_of_course': (snapshot_day - 1) // 7 + 1
    })

    return features
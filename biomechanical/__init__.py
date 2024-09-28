import numpy as np

# Classification thresholds
angle_thresholds = {
    'Left Knee Flexion': {'extreme': 30, 'high': 45, 'moderate': 60},
    'Right Knee Flexion': {'extreme': 30, 'high': 45, 'moderate': 60},
    'Left Hip Flexion': {'extreme': 15, 'high': 30, 'moderate': 45},
    'Right Hip Flexion': {'extreme': 15, 'high': 30, 'moderate': 45},
    'Left Knee Valgus': {'moderate': 5, 'high': 10, 'extreme': 15},
    'Right Knee Valgus': {'moderate': 5, 'high': 10, 'extreme': 15},
    'Left Hip Adduction': {'moderate': 10, 'high': 15, 'extreme': 20},
    'Right Hip Adduction': {'moderate': 10, 'high': 15, 'extreme': 20}
}

def score_angle(value, thresholds, higher_is_riskier=False):
    """Calculate a risk score and category for an angle."""
    if higher_is_riskier:
        # For angles where higher values are riskier
        if value <= thresholds['moderate']:
            risk_score = 0
        elif value >= thresholds['extreme']:
            risk_score = 100
        elif value <= thresholds['high']:
            # Moderate to High Risk
            risk_score = ((value - thresholds['moderate']) / (thresholds['high'] - thresholds['moderate'])) * 33.33
        else:
            # High to Extreme Risk
            risk_score = 66.66 + ((value - thresholds['high']) / (thresholds['extreme'] - thresholds['high'])) * 33.34
    else:
        # For angles where lower values are riskier
        if value >= thresholds['moderate']:
            risk_score = 0
        elif value <= thresholds['extreme']:
            risk_score = 100
        elif value >= thresholds['high']:
            # Moderate to High Risk
            risk_score = ((thresholds['moderate'] - value) / (thresholds['moderate'] - thresholds['high'])) * 33.33
        else:
            # High to Extreme Risk
            risk_score = 66.66 + ((thresholds['high'] - value) / (thresholds['high'] - thresholds['extreme'])) * 33.34

    # Determine risk category
    if risk_score == 0:
        risk_category = 'Normal'
    elif risk_score <= 33.33:
        risk_category = 'Moderate Risk'
    elif risk_score <= 66.66:
        risk_category = 'High Risk'
    else:
        risk_category = 'Extreme Risk'

    return risk_score, risk_category

def score_angles(angles):
    """Score all angles based on risk thresholds and categorize them."""
    scored_angles = {}
    for angle_type, angle_value in angles.items():
        assert angle_type in angle_thresholds, f"Angle '{angle_type}' not found in thresholds"
        thresholds = angle_thresholds[angle_type]

        # Determine if higher values are riskier (e.g., Knee Valgus, Hip Adduction)
        higher_is_riskier = angle_type in ['Left Knee Valgus', 'Right Knee Valgus', 'Left Hip Adduction', 'Right Hip Adduction']

        risk_score, risk_category = score_angle(angle_value, thresholds, higher_is_riskier)
        scored_angles[angle_type] = {
            'value': angle_value,
            'risk_score': risk_score,
            'risk_category': risk_category
        }
    return scored_angles

# Thresholds for angular accelerations (degrees/second^2)
acceleration_thresholds = {
    'Knee Flexion': {'moderate': 3000, 'high': 5000, 'extreme': 7000},
    'Hip Flexion': {'moderate': 2000, 'high': 4000, 'extreme': 6000},
    'Knee Valgus': {'moderate': 1000, 'high': 2000, 'extreme': 3000},
    'Hip Adduction': {'moderate': 1000, 'high': 2000, 'extreme': 3000},
}

def score_acceleration(value, thresholds):
    """Calculate a risk score and category for an acceleration."""
    if value <= thresholds['moderate']:
        risk_score = 0
    elif value >= thresholds['extreme']:
        risk_score = 100
    elif value <= thresholds['high']:
        # Moderate to High Risk
        risk_score = ((value - thresholds['moderate']) / (thresholds['high'] - thresholds['moderate'])) * 33.33
    else:
        # High to Extreme Risk
        risk_score = 66.66 + ((value - thresholds['high']) / (thresholds['extreme'] - thresholds['high'])) * 33.34

    # Determine risk category
    if risk_score == 0:
        risk_category = 'Normal'
    elif risk_score <= 33.33:
        risk_category = 'Moderate Risk'
    elif risk_score <= 66.66:
        risk_category = 'High Risk'
    else:
        risk_category = 'Extreme Risk'

    return risk_score, risk_category

def score_accelerations(accelerations):
    """Score accelerations based on risk thresholds and categorize them."""
    scored_accelerations = {}
    for joint, acceleration_values in accelerations.items():
        acceleration_value = abs(acceleration_values[-1]) if len(acceleration_values) > 0 else 0

        # Determine the type of joint (e.g., 'Knee Flexion')
        joint_type = ' '.join(joint.split()[1:])  # Remove 'Left' or 'Right'

        assert joint_type in acceleration_thresholds, f"Joint type '{joint_type}' not found in acceleration thresholds"
        thresholds = acceleration_thresholds[joint_type]

        risk_score, risk_category = score_acceleration(acceleration_value, thresholds)
        scored_accelerations[joint] = {
            'value': acceleration_value,
            'risk_score': risk_score,
            'risk_category': risk_category
        }
    return scored_accelerations

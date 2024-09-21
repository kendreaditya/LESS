"""
Landing Error Scoring System (LESS) Implementation

Implaments the Landing Error Scoring System (LESS) based on the data from Padua et al. (2009).
"""

def knee_flexion_angle_at_initial_contact(knee_flexion_angle):
    """
    1. Knee Flexion Angle at Initial Contact
    Parameters:
    knee_flexion_angle (float): The knee flexion angle in degrees.
    Returns:
    int: Returns 0 if the knee flexion angle is greater than 30 degrees, otherwise returns 1.
    """
    return 0 if knee_flexion_angle > 30 else 1

def hip_flexion_angle_at_initial_contact(hip_flexion_angle):
    """
    2. Hip Flexion Angle at Initial Contact
    Parameters:
    hip_flexion_angle (float): The hip flexion angle in degrees.
    Returns:
    int: Returns 0 if the hip is flexed (angle > 0), otherwise returns 1.
    """
    return 0 if hip_flexion_angle > 0 else 1

def trunk_flexion_angle_at_initial_contact(trunk_flexion_angle):
    """
    3. Trunk Flexion Angle at Initial Contact
    Parameters:
    trunk_flexion_angle (float): The trunk flexion angle in degrees.
    Returns:
    int: Returns 0 if the trunk is flexed (angle > 0), otherwise returns 1.
    """
    return 0 if trunk_flexion_angle > 0 else 1

def ankle_plantar_flexion_angle_at_initial_contact(foot_landing):
    """
    4. Ankle Plantar-flexion Angle at Initial Contact
    Parameters:
    foot_landing (str): The foot landing pattern ('toe_to_heel' or 'heel_to_toe' or 'flat').
    Returns:
    int: Returns 0 if landing toe to heel, otherwise returns 1.
    """
    return 0 if foot_landing == 'toe_to_heel' else 1

def knee_valgus_angle_at_initial_contact(knee_valgus_angle):
    """
    5. Knee Valgus Angle at Initial Contact
    Parameters:
    knee_valgus_angle (float): The knee valgus angle in degrees.
    Returns:
    int: Returns 1 if the knee is in valgus (angle > 0), otherwise returns 0.
    """
    return 1 if knee_valgus_angle > 0 else 0

def lateral_trunk_flexion_angle_at_initial_contact(lateral_trunk_flexion):
    """
    6. Lateral Trunk Flexion Angle at Initial Contact
    Parameters:
    lateral_trunk_flexion (bool): True if the trunk is flexed laterally, False otherwise.
    Returns:
    int: Returns 1 if the trunk is flexed laterally, otherwise returns 0.
    """
    return 1 if lateral_trunk_flexion else 0

def stance_width_wide(stance_width, shoulder_width):
    """
    7. Stance Width - Wide
    Parameters:
    stance_width (float): The width of the stance.
    shoulder_width (float): The width of the shoulders.
    Returns:
    int: Returns 1 if stance width is greater than shoulder width, otherwise returns 0.
    """
    return 1 if stance_width > shoulder_width else 0

def stance_width_narrow(stance_width, shoulder_width):
    """
    8. Stance Width - Narrow
    Parameters:
    stance_width (float): The width of the stance.
    shoulder_width (float): The width of the shoulders.
    Returns:
    int: Returns 1 if stance width is less than shoulder width, otherwise returns 0.
    """
    return 1 if stance_width < shoulder_width else 0

def foot_position_toe_in(foot_rotation):
    """
    9. Foot Position - Toe In
    Parameters:
    foot_rotation (float): The foot rotation angle in degrees (positive for internal rotation).
    Returns:
    int: Returns 1 if foot is internally rotated more than 30 degrees, otherwise returns 0.
    """
    return 1 if foot_rotation > 30 else 0

def foot_position_toe_out(foot_rotation):
    """
    10. Foot Position - Toe Out
    Parameters:
    foot_rotation (float): The foot rotation angle in degrees (negative for external rotation).
    Returns:
    int: Returns 1 if foot is externally rotated more than 30 degrees, otherwise returns 0.
    """
    return 1 if foot_rotation < -30 else 0

def symmetric_initial_foot_contact(is_symmetric):
    """
    11. Symmetric Initial Foot Contact
    Parameters:
    is_symmetric (bool): True if the initial foot contact is symmetric, False otherwise.
    Returns:
    int: Returns 0 if the initial foot contact is symmetric, otherwise returns 1.
    """
    return 0 if is_symmetric else 1

def knee_flexion_displacement(initial_flexion, max_flexion):
    """
    12. Knee Flexion Displacement
    Parameters:
    initial_flexion (float): Initial knee flexion angle in degrees.
    max_flexion (float): Maximum knee flexion angle in degrees.
    Returns:
    int: Returns 0 if knee flexes more than 45 degrees from initial contact to max flexion, otherwise returns 1.
    """
    return 0 if (max_flexion - initial_flexion) > 45 else 1

def hip_flexion_at_max_knee_flexion(initial_hip_flexion, hip_flexion_at_max_knee_flexion):
    """
    13. Hip Flexion at Max Knee Flexion
    Parameters:
    initial_hip_flexion (float): Initial hip flexion angle in degrees.
    hip_flexion_at_max_knee_flexion (float): Hip flexion angle at max knee flexion in degrees.
    Returns:
    int: Returns 0 if hip flexes more from initial contact to max knee flexion, otherwise returns 1.
    """
    return 0 if hip_flexion_at_max_knee_flexion > initial_hip_flexion else 1

def trunk_flexion_at_max_knee_flexion(initial_trunk_flexion, trunk_flexion_at_max_knee_flexion):
    """
    14. Trunk Flexion at Max Knee Flexion
    Parameters:
    initial_trunk_flexion (float): Initial trunk flexion angle in degrees.
    trunk_flexion_at_max_knee_flexion (float): Trunk flexion angle at max knee flexion in degrees.
    Returns:
    int: Returns 0 if trunk flexes more from initial contact to max knee flexion, otherwise returns 1.
    """
    return 0 if trunk_flexion_at_max_knee_flexion > initial_trunk_flexion else 1

def knee_valgus_displacement(max_knee_valgus):
    """
    15. Knee Valgus Displacement
    Parameters:
    max_knee_valgus (float): Maximum knee valgus angle in degrees.
    Returns:
    int: Returns 1 if max knee valgus is greater than a threshold (e.g., 10 degrees), otherwise returns 0.
    """
    VALGUS_THRESHOLD = 10  # This threshold might need adjustment based on specific criteria
    return 1 if max_knee_valgus > VALGUS_THRESHOLD else 0

def joint_displacement(displacement):
    """
    16. Joint Displacement
    Parameters:
    displacement (str): 'soft', 'average', or 'stiff'
    Returns:
    int: Returns 0 for 'soft', 1 for 'average', and 2 for 'stiff'
    """
    if displacement == 'soft':
        return 0
    elif displacement == 'average':
        return 1
    else:  # 'stiff'
        return 2

def overall_impression(landing_quality):
    """
    17. Overall Impression
    Parameters:
    landing_quality (str): 'excellent', 'average', or 'poor'
    Returns:
    int: Returns 0 for 'excellent', 1 for 'average', and 2 for 'poor'
    """
    if landing_quality == 'excellent':
        return 0
    elif landing_quality == 'average':
        return 1
    else:  # 'poor'
        return 2

def calculate_less_score(scores):
    """
    Calculate the total LESS score
    Parameters:
    scores (list): List of individual scores for each of the 17 items
    Returns:
    int: Total LESS score
    """
    return sum(scores)

def interpret_less_score(total_score):
    """
    Interpret the total LESS score
    Parameters:
    total_score (int): Total LESS score
    Returns:
    str: Interpretation of the score
    """
    if total_score <= 4:
        return "Excellent"
    elif total_score <= 5:
        return "Good"
    elif total_score <= 6:
        return "Moderate"
    else:
        return "Poor"

if __name__ == "__main__":
    def calculate_level_scores():
        """
        Calculate LESS scores for each level (L1, L2, L3, L4) using mean values from the tables.
        Returns:
        dict: A dictionary containing the total score and interpretation for each level.
        """
        level_data = {
            'L1': {
                'knee_flexion': 18.28,
                'hip_flexion': -31.17,
                'trunk_flexion': 89.68 - 80.57,  # Approximation based on available data
                'ankle_plantar_flexion': 'heel_to_toe',  # Assumption based on low score
                'knee_valgus': 1.67,
                'lateral_trunk_flexion': False,  # Assumption based on low score
                'stance_width': 'normal',  # Assumption
                'foot_rotation': 0,  # Assumption based on low scores for toe-in and toe-out
                'symmetric_foot_contact': True,  # Assumption based on low score
                'knee_flexion_displacement': 71.39,
                'hip_flexion_displacement': 49.35,
                'trunk_flexion_displacement': 9.11,  # Calculated from Table 3
                'knee_valgus_displacement': -11.02,
                'joint_displacement': 'soft',  # Assumption based on high displacement values
                'overall_impression': 'excellent'  # Based on being in the L1 category
            },
            'L2': {
                'knee_flexion': 16.61,
                'hip_flexion': -28.92,
                'trunk_flexion': 81.31 - 68.7,  # Approximation
                'ankle_plantar_flexion': 'heel_to_toe',  # Assumption
                'knee_valgus': 0.62,
                'lateral_trunk_flexion': False,  # Assumption
                'stance_width': 'normal',  # Assumption
                'foot_rotation': 0,  # Assumption
                'symmetric_foot_contact': True,  # Assumption
                'knee_flexion_displacement': 64.70,
                'hip_flexion_displacement': 39.86,
                'trunk_flexion_displacement': 12.61,  # Calculated from Table 3
                'knee_valgus_displacement': -12.29,
                'joint_displacement': 'average',  # Assumption
                'overall_impression': 'good'  # Based on being in the L2 category
            },
            'L3': {
                'knee_flexion': 16.32,
                'hip_flexion': -28.15,
                'trunk_flexion': 77.77 - 62.63,  # Approximation
                'ankle_plantar_flexion': 'flat',  # Assumption
                'knee_valgus': 0.28,
                'lateral_trunk_flexion': True,  # Assumption
                'stance_width': 'wide',  # Assumption
                'foot_rotation': 15,  # Assumption
                'symmetric_foot_contact': False,  # Assumption
                'knee_flexion_displacement': 61.44,
                'hip_flexion_displacement': 34.48,
                'trunk_flexion_displacement': 15.14,  # Calculated from Table 3
                'knee_valgus_displacement': -12.81,
                'joint_displacement': 'average',  # Assumption
                'overall_impression': 'moderate'  # Based on being in the L3 category
            },
            'L4': {
                'knee_flexion': 15.87,
                'hip_flexion': -26.64,
                'trunk_flexion': 71.38 - 53.03,  # Approximation
                'ankle_plantar_flexion': 'toe_to_heel',  # Assumption
                'knee_valgus': -0.15,
                'lateral_trunk_flexion': True,  # Assumption
                'stance_width': 'wide',  # Assumption
                'foot_rotation': -35,  # Assumption
                'symmetric_foot_contact': False,  # Assumption
                'knee_flexion_displacement': 55.52,
                'hip_flexion_displacement': 26.50,
                'trunk_flexion_displacement': 18.35,  # Calculated from Table 3
                'knee_valgus_displacement': -14.27,
                'joint_displacement': 'stiff',  # Assumption
                'overall_impression': 'poor'  # Based on being in the L4 category
            }
        }

        results = {}
        for level, data in level_data.items():
            scores = [
                knee_flexion_angle_at_initial_contact(data['knee_flexion']),
                hip_flexion_angle_at_initial_contact(data['hip_flexion']),
                trunk_flexion_angle_at_initial_contact(data['trunk_flexion']),
                ankle_plantar_flexion_angle_at_initial_contact(data['ankle_plantar_flexion']),
                knee_valgus_angle_at_initial_contact(data['knee_valgus']),
                lateral_trunk_flexion_angle_at_initial_contact(data['lateral_trunk_flexion']),
                stance_width_wide(data['stance_width'], 'normal'),
                stance_width_narrow(data['stance_width'], 'normal'),
                foot_position_toe_in(data['foot_rotation']),
                foot_position_toe_out(data['foot_rotation']),
                symmetric_initial_foot_contact(data['symmetric_foot_contact']),
                knee_flexion_displacement(data['knee_flexion'], data['knee_flexion'] + data['knee_flexion_displacement']),
                hip_flexion_at_max_knee_flexion(data['hip_flexion'], data['hip_flexion'] - data['hip_flexion_displacement']),
                trunk_flexion_at_max_knee_flexion(data['trunk_flexion'], data['trunk_flexion'] + data['trunk_flexion_displacement']),
                knee_valgus_displacement(data['knee_valgus_displacement']),
                joint_displacement(data['joint_displacement']),
                overall_impression(data['overall_impression'])
            ]
            total_score = calculate_less_score(scores)
            interpretation = interpret_less_score(total_score)
            results[level] = {'total_score': total_score, 'interpretation': interpretation}
        
        return results

    level_scores = calculate_level_scores()
    for level, result in level_scores.items():
        print(f"{level}: Total LESS Score: {result['total_score']}, Interpretation: {result['interpretation']}")
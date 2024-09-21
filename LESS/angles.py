"""
This is a module for analyzing joint angles during the gait cycle

Implamented based on the data from LESS (Padua et al.) (2009) Knimematics Tables
"""

import math
from typing import List, Tuple, Dict


def calculate_mean_std(values: List[float]) -> Tuple[float, float]:
    """Calculate the mean and standard deviation of a list of values."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = math.sqrt(variance)
    return mean, std_dev

def calculate_zscore(value: float, mean: float, std_dev: float) -> float:
    """Calculate the z-score for a given value."""
    return (value - mean) / std_dev

def get_category(zscore: float) -> str:
    """Determine the category based on the z-score."""
    if zscore > 0.5:
        return "Excellent"
    elif -0.5 < zscore <= 0.5:
        return "Good"
    elif -1.5 < zscore <= -0.5:
        return "Moderate"
    else:
        return "Poor"

class AngleDistribution:
    def __init__(self, excellent: float, good: float, moderate: float, poor: float):
        self.values = [excellent, good, moderate, poor]
        self.mean, self.std_dev = calculate_mean_std(self.values)

    def get_zscore_and_category(self, angle: float) -> Dict[str, float]:
        zscore = calculate_zscore(angle, self.mean, self.std_dev)
        category = get_category(zscore)
        return {"zscore": zscore, "category": category}

# Initial Contact Angles
knee_flexion_ic = AngleDistribution(18.28, 16.61, 16.32, 15.87)
knee_valgus_ic = AngleDistribution(1.67, 0.62, 0.28, -0.15)
tibial_rotation_ic = AngleDistribution(-1.61, -0.99, -0.64, 0.35)
hip_flexion_ic = AngleDistribution(-31.17, -28.92, -28.15, -26.64)
hip_adduction_ic = AngleDistribution(-11.10, -10.39, -9.88, -10.12)
hip_rotation_ic = AngleDistribution(-4.20, -4.69, -4.00, -4.12)

# Peak Angles Over Stance
knee_flexion_peak = AngleDistribution(89.68, 81.31, 77.77, 71.38)
knee_valgus_peak = AngleDistribution(-11.02, -12.29, -12.81, -14.27)
tibial_rotation_peak = AngleDistribution(15.89, 15.28, 14.86, 14.69)
hip_flexion_peak = AngleDistribution(-80.57, -68.7, -62.63, -53.03)
hip_adduction_peak = AngleDistribution(0.69, 1.16, 1.70, 1.65)
hip_rotation_peak = AngleDistribution(6.45, 4.16, 4.36, 3.71)

# Displacement Over Stance
knee_flexion_disp = AngleDistribution(71.39, 64.70, 61.44, 55.52)
knee_valgus_disp = AngleDistribution(-12.69, -12.87, -13.07, -14.16)
tibial_rotation_disp = AngleDistribution(17.45, 16.29, 15.50, 14.38)
hip_flexion_disp = AngleDistribution(-49.35, -39.86, -34.48, -26.50)
hip_adduction_disp = AngleDistribution(11.81, 11.56, 11.58, 11.82)
hip_rotation_disp = AngleDistribution(10.64, 9.04, 8.37, 7.93)

def analyze_angle(angle: float, distribution: AngleDistribution) -> Dict[str, float]:
    """
    Analyze an angle measurement using the provided distribution.
    > 0.5: Excellent
    -0.5 to 0.5: Good
    -1.5 to -0.5: Moderate
    < -1.5: Poor
    """
    return distribution.get_zscore_and_category(angle)

# Dictionary to store all distributions
angle_distributions = {
    "Initial Contact": {
        "Knee Flexion": knee_flexion_ic,
        "Knee Valgus": knee_valgus_ic,
        "Tibial Rotation": tibial_rotation_ic,
        "Hip Flexion": hip_flexion_ic,
        "Hip Adduction": hip_adduction_ic,
        "Hip Rotation": hip_rotation_ic
    },
    "Peak Angle": {
        "Knee Flexion": knee_flexion_peak,
        "Knee Valgus": knee_valgus_peak,
        "Tibial Rotation": tibial_rotation_peak,
        "Hip Flexion": hip_flexion_peak,
        "Hip Adduction": hip_adduction_peak,
        "Hip Rotation": hip_rotation_peak
    },
    "Displacement": {
        "Knee Flexion": knee_flexion_disp,
        "Knee Valgus": knee_valgus_disp,
        "Tibial Rotation": tibial_rotation_disp,
        "Hip Flexion": hip_flexion_disp,
        "Hip Adduction": hip_adduction_disp,
        "Hip Rotation": hip_rotation_disp
    }
}

def analyze_all_angles(angles: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Analyze all provided angles and return results."""
    results = {}
    for phase, measurements in angles.items():
        results[phase] = {}
        for measurement, angle in measurements.items():
            distribution = angle_distributions[phase][measurement]
            results[phase][measurement] = analyze_angle(angle, distribution)
    return results

if __name__ == "__main__":
    # Example angles for analysis
    example_angles = {
        "Initial Contact": {
            "Knee Flexion": 17.5,
            "Knee Valgus": 0.5,
            "Tibial Rotation": -1.0,
            "Hip Flexion": -29.0,
            "Hip Adduction": -10.5,
            "Hip Rotation": -4.5
        },
        "Peak Angle": {
            "Knee Flexion": 80.0,
            "Knee Valgus": -12.5,
            "Tibial Rotation": 15.0,
            "Hip Flexion": -65.0,
            "Hip Adduction": 1.2,
            "Hip Rotation": 5.0
        },
        "Displacement": {
            "Knee Flexion": 63.0,
            "Knee Valgus": -13.0,
            "Tibial Rotation": 16.0,
            "Hip Flexion": -38.0,
            "Hip Adduction": 11.7,
            "Hip Rotation": 9.0
        }
    }

    results = analyze_all_angles(example_angles)

    for phase, measurements in results.items():
        print(f"\n{phase}:")
        for measurement, result in measurements.items():
            print(f"  {measurement}:")
            print(f"    Angle: {example_angles[phase][measurement]:.2f}")
            print(f"    Z-Score: {result['zscore']:.2f}")
            print(f"    Category: {result['category']}")
#!/usr/bin/env python3
"""
Analysis of the strictness of ASSM triggering conditions in _compute_small_target_mask function
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set Chinese font support
def set_chinese_font():
    """Set matplotlib Chinese font"""
    # Try different Chinese fonts
    font_candidates = [
        'SimHei', 
        'Microsoft YaHei', 
        'WenQuanYi Micro Hei', 
        'Source Han Sans CN', 
        'Noto Sans CJK SC', 
        'DengXian', 
        'FangSong',
        'KaiTi'
    ]
    
    font_found = False
    for font in font_candidates:
        try:
            mpl.rcParams['font.sans-serif'] = [font]
            # Solve negative sign display problem
            mpl.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = ['sans-serif']
            # Test if available
            test_fig = plt.figure(figsize=(1, 1))
            test_ax = test_fig.add_subplot(111)
            test_ax.set_title('Test')
            plt.close(test_fig)
            font_found = True
            print(f"Using Chinese font: {font}")
            break
        except:
            continue
    
    if not font_found:
        print("Warning: No suitable Chinese font found, Chinese characters in charts may appear as garbled")
        # Fallback font
        mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
        mpl.rcParams['axes.unicode_minus'] = False

def analyze_assm_conditions():
    """Analyze the strictness of current ASSM triggering conditions"""
    
    print("=== ASSM Module Triggering Condition Analysis ===")
    print()
    
    # Current hardcoded thresholds
    AREA_THRESHOLD = 0.01  # Area threshold
    SCORE_THRESHOLD = 0.2  # Confidence threshold
    DEFAULT_AREA_FOR_CX_CY = 0.005  # Default area for [cx,cy] format
    
    print("1. Current Hardcoded Thresholds:")
    print(f"   - Area threshold: {AREA_THRESHOLD} (target area must be < {AREA_THRESHOLD})")
    print(f"   - Confidence threshold: {SCORE_THRESHOLD} (max confidence must be < {SCORE_THRESHOLD})")
    print(f"   - Default area for [cx,cy] format: {DEFAULT_AREA_FOR_CX_CY}")
    print()
    
    # Analysis of area threshold meaning
    print("2. Area Threshold Analysis:")
    # Assuming input image is 640x640, pixel size corresponding to different area thresholds
    img_size = 640
    area_in_pixels = AREA_THRESHOLD * (img_size ** 2)
    side_length = np.sqrt(area_in_pixels)
    
    print(f"   - In a {img_size}x{img_size} image, area threshold {AREA_THRESHOLD} corresponds to:")
    print(f"     * Pixel area: {area_in_pixels:.1f} pixels²")
    print(f"     * Equivalent square side length: {side_length:.1f} pixels")
    print(f"   - This means only targets with side length less than {side_length:.1f} pixels are considered 'small targets'")
    print()
    
    # Confidence threshold analysis
    print("3. Confidence Threshold Analysis:")
    print(f"   - Confidence threshold {SCORE_THRESHOLD} means:")
    print(f"     * Only tokens with max class confidence < {SCORE_THRESHOLD} can use ASSM")
    print(f"     * After softmax, {SCORE_THRESHOLD} is a relatively low confidence")
    print(f"     * Most valid detection targets will exceed this threshold")
    print()
    
    # Simulate trigger rate under different conditions
    print("4. Condition Strictness Simulation:")
    
    # Generate simulation data
    num_queries = 300
    num_batches = 100
    
    # Simulate area distribution (log-normal distribution, simulating real target area distribution)
    areas = np.random.lognormal(mean=-3, sigma=1, size=(num_batches, num_queries))
    areas = np.clip(areas, 0.001, 0.5)  # Limit to reasonable range
    
    # Simulate confidence distribution (beta distribution, simulating confidence during training)
    scores = np.random.beta(a=2, b=3, size=(num_batches, num_queries))
    
    # Calculate trigger rate under current conditions
    area_mask = areas < AREA_THRESHOLD
    score_mask = scores < SCORE_THRESHOLD
    combined_mask = area_mask & score_mask
    
    area_trigger_rate = np.mean(area_mask) * 100
    score_trigger_rate = np.mean(score_mask) * 100
    combined_trigger_rate = np.mean(combined_mask) * 100
    
    print(f"   Simulation results (based on {num_batches} batches, {num_queries} queries each):")
    print(f"   - Area condition trigger rate: {area_trigger_rate:.2f}%")
    print(f"   - Confidence condition trigger rate: {score_trigger_rate:.2f}%")
    print(f"   - Combined condition trigger rate: {combined_trigger_rate:.2f}%")
    print()
    
    # Analysis of trigger rates under different thresholds
    print("5. Trigger Rate Analysis under Different Thresholds:")
    area_thresholds = [0.005, 0.01, 0.02, 0.03, 0.05]
    score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print("   Area Threshold Impact:")
    for area_th in area_thresholds:
        area_rate = np.mean(areas < area_th) * 100
        combined_rate = np.mean((areas < area_th) & (scores < SCORE_THRESHOLD)) * 100
        print(f"     Area threshold={area_th:.3f}: Area trigger rate={area_rate:.1f}%, Combined rate={combined_rate:.1f}%")
    
    print("\n   Confidence Threshold Impact:")
    for score_th in score_thresholds:
        score_rate = np.mean(scores < score_th) * 100
        combined_rate = np.mean((areas < AREA_THRESHOLD) & (scores < score_th)) * 100
        print(f"     Confidence threshold={score_th:.1f}: Confidence trigger rate={score_rate:.1f}%, Combined rate={combined_rate:.1f}%")
    
    print()
    
    # Problem summary
    print("6. Problems with Current Conditions:")
    print("   ✗ Area threshold is too strict:")
    print(f"     - {AREA_THRESHOLD} area threshold only covers {area_trigger_rate:.1f}% of queries")
    print(f"     - In a 640x640 image, this is equivalent to only {side_length:.1f}x{side_length:.1f} pixel targets")
    print("   ✗ Confidence threshold may be too low:")
    print(f"     - {SCORE_THRESHOLD} confidence threshold has a coverage of {score_trigger_rate:.1f}% during training")
    print("   ✗ Combined conditions are too strict:")
    print(f"     - Probability of both conditions being met is only {combined_trigger_rate:.1f}%")
    print("   ✗ Hardcoded thresholds lack adaptability:")
    print("     - Cannot dynamically adjust based on training stage, dataset characteristics, etc.")
    print()
    
    # Improvement suggestions
    print("7. Improvement Suggestions:")
    print("   ✓ Relax area threshold:")
    print("     - Suggest increasing area threshold from 0.01 to 0.02-0.03")
    print("     - Or use relative area threshold (percentage relative to image size)")
    print("   ✓ Adjust confidence threshold:")
    print("     - Suggest increasing confidence threshold from 0.2 to 0.3-0.4")
    print("     - Or use dynamic threshold (based on confidence distribution of current batch)")
    print("   ✓ Add configurable parameters:")
    print("     - Make thresholds configurable parameters rather than hardcoded")
    print("     - Support threshold scheduling during training")
    print("   ✓ Consider other triggering conditions:")
    print("     - Add gradient information or loss information")
    print("     - Consider difficulty level of target categories")

def visualize_trigger_conditions():
    """Visualize distribution of triggering conditions"""
    try:
        import matplotlib.pyplot as plt
        
        # Set Chinese font
        set_chinese_font()
        
        # Generate simulation data
        num_samples = 10000
        areas = np.random.lognormal(mean=-3, sigma=1, size=num_samples)
        areas = np.clip(areas, 0.001, 0.5)
        scores = np.random.beta(a=2, b=3, size=num_samples)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Area distribution
        axes[0, 0].hist(areas, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(0.01, color='red', linestyle='--', linewidth=2, label='Current threshold=0.01')
        axes[0, 0].set_xlabel('Area')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Target Area Distribution')
        axes[0, 0].legend()
        
        # Confidence distribution
        axes[0, 1].hist(scores, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(0.2, color='red', linestyle='--', linewidth=2, label='Current threshold=0.2')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].legend()
        
        # 2D distribution
        axes[1, 0].scatter(areas, scores, alpha=0.5, s=1)
        axes[1, 0].axvline(0.01, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].axhline(0.2, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].fill_between([0, 0.01], [0, 0], [0.2, 0.2], alpha=0.2, color='red', label='ASSM Trigger Region')
        axes[1, 0].set_xlabel('Area')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].set_title('Area vs Confidence Distribution')
        axes[1, 0].legend()
        
        # Trigger rate under different thresholds
        area_thresholds = np.linspace(0.005, 0.05, 20)
        trigger_rates = []
        for area_th in area_thresholds:
            rate = np.mean((areas < area_th) & (scores < 0.2)) * 100
            trigger_rates.append(rate)
        
        axes[1, 1].plot(area_thresholds, trigger_rates, 'b-', linewidth=2)
        axes[1, 1].axvline(0.01, color='red', linestyle='--', alpha=0.7, label='Current area threshold')
        axes[1, 1].set_xlabel('Area threshold')
        axes[1, 1].set_ylabel('ASSM trigger rate (%)')
        axes[1, 1].set_title('Effect of Area Threshold on Trigger Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/zhanghaochen/ssm_detr/assm_conditions_analysis.png', dpi=300, bbox_inches='tight')
        print("\nChart saved to: /home/zhanghaochen/ssm_detr/assm_conditions_analysis.png")
        
    except ImportError:
        print("\nNote: matplotlib not installed, visualization skipped")

if __name__ == "__main__":
    analyze_assm_conditions()
    visualize_trigger_conditions()
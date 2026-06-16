import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('results_chinese_native_v4pro.csv')

# Calculate reasoning length (character count) per question
df['reasoning_length'] = df['Reasoning'].str.len()

# Split by correctness
correct = df[df['Is Correct'] == True]['reasoning_length']
incorrect = df[df['Is Correct'] == False]['reasoning_length']

# ===== Descriptive Statistics =====
print("=" * 50)
print("Chinese Native V4-Pro - Reasoning Length Statistics (chars)")
print("=" * 50)

print(f"\nTotal questions: {len(df)}")
print(f"Correct: {len(correct)}, Incorrect: {len(incorrect)}")
print(f"Accuracy: {len(correct)/len(df)*100:.1f}%")

print(f"\n--- Correct ---")
print(f"Mean:   {correct.mean():.0f}")
print(f"Median: {correct.median():.0f}")
print(f"Std:    {correct.std():.0f}")
print(f"Min:    {correct.min():.0f}")
print(f"Max:    {correct.max():.0f}")

print(f"\n--- Incorrect ---")
print(f"Mean:   {incorrect.mean():.0f}")
print(f"Median: {incorrect.median():.0f}")
print(f"Std:    {incorrect.std():.0f}")
print(f"Min:    {incorrect.min():.0f}")
print(f"Max:    {incorrect.max():.0f}")

# Incorrect / Correct ratio
ratio = incorrect.mean() / correct.mean()
print(f"\nIncorrect/Correct mean ratio: {ratio:.2f}x")

# ===== Welch's t-test =====
if len(incorrect) >= 2:
    t_stat, p_value = stats.ttest_ind(correct, incorrect, equal_var=False)
    
    # Cohen's d
    pooled_std = np.sqrt((correct.std()**2 + incorrect.std()**2) / 2)
    cohens_d = (incorrect.mean() - correct.mean()) / pooled_std
    
    print(f"\n--- Welch's t-test ---")
    print(f"t = {t_stat:.3f}")
    print(f"p = {p_value:.4f}")
    print(f"Cohen's d = {cohens_d:.2f}")
    
    if p_value < 0.05:
        print("-> Significant difference (p < 0.05)")
    else:
        print("-> Not significant (p >= 0.05), likely due to small sample size")
else:
    print("\nFewer than 2 incorrect samples, cannot perform t-test")

# ===== Save results to file =====
output_file = 'stats_chinese_native_v4pro.txt'
with open(output_file, 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("Chinese Native V4-Pro - Reasoning Length Statistics (chars)\n")
    f.write("=" * 50 + "\n")
    f.write(f"\nTotal questions: {len(df)}\n")
    f.write(f"Correct: {len(correct)}, Incorrect: {len(incorrect)}\n")
    f.write(f"Accuracy: {len(correct)/len(df)*100:.1f}%\n")
    f.write(f"\n--- Correct ---\n")
    f.write(f"Mean:   {correct.mean():.0f}\n")
    f.write(f"Median: {correct.median():.0f}\n")
    f.write(f"Std:    {correct.std():.0f}\n")
    f.write(f"Min:    {correct.min():.0f}\n")
    f.write(f"Max:    {correct.max():.0f}\n")
    f.write(f"\n--- Incorrect ---\n")
    f.write(f"Mean:   {incorrect.mean():.0f}\n")
    f.write(f"Median: {incorrect.median():.0f}\n")
    f.write(f"Std:    {incorrect.std():.0f}\n")
    f.write(f"Min:    {incorrect.min():.0f}\n")
    f.write(f"Max:    {incorrect.max():.0f}\n")
    f.write(f"\nIncorrect/Correct mean ratio: {ratio:.2f}x\n")
    if len(incorrect) >= 2:
        f.write(f"\n--- Welch's t-test ---\n")
        f.write(f"t = {t_stat:.3f}\n")
        f.write(f"p = {p_value:.4f}\n")
        f.write(f"Cohen's d = {cohens_d:.2f}\n")
    else:
        f.write("\nFewer than 2 incorrect samples, cannot perform t-test\n")

print(f"\nResults saved to {output_file}")

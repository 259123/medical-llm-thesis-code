import pandas as pd
import os

conditions = [
    ('results_english_original_v4pro.csv',   'English Original'),
    ('results_english_translated_v4pro.csv', 'English Translated'),
    ('results_chinese_native_v4pro.csv',     'Chinese Native'),
]

output_file = 'errors_v4pro.txt'

with open(output_file, 'w', encoding='utf-8') as out:
    for csv_file, label in conditions:
        if not os.path.exists(csv_file):
            out.write(f"[MISSING FILE: {csv_file}]\n\n")
            continue

        df = pd.read_csv(csv_file)
        df['reasoning_length'] = df['Reasoning'].str.len()
        incorrect = df[df['Is Correct'] == False].reset_index()

        out.write("=" * 70 + "\n")
        out.write(f"  {label} — {len(incorrect)} incorrect\n")
        out.write("=" * 70 + "\n\n")

        if len(incorrect) == 0:
            out.write("No errors.\n\n")
            continue

        for i, row in incorrect.iterrows():
            out.write(f"--- Error {i+1} (row {row['index']+1}) ---\n")
            out.write(f"Correct Answer : {row['Correct Answer']}\n")
            out.write(f"Model Answer   : {row['Model Answer']}\n")
            out.write(f"Reasoning Length: {row['reasoning_length']} chars\n\n")
            out.write("QUESTION:\n")
            out.write(str(row['Question']) + "\n\n")
            out.write("OPTIONS:\n")
            out.write(str(row['Options']) + "\n\n")
            out.write("REASONING TRACE:\n")
            out.write(str(row['Reasoning']) + "\n")
            out.write("\n" + "-" * 70 + "\n\n")

print(f"Done! Errors saved to {output_file}")

import argparse
import os
import re
import pandas as pd


def parse_bool(x):
    """Convert common CSV truth values to bool."""
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    x = str(x).strip().lower()
    return x in {"true", "1", "yes", "y", "correct"}


def english_word_count(text):
    """
    Count English-like word tokens and numbers.
    This is a simple surface measure, not a linguistic parser.
    """
    if pd.isna(text):
        return 0
    text = str(text)
    tokens = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?", text)
    return len(tokens)


def nonspace_char_count(text):
    """
    Count non-whitespace characters.
    Useful for Chinese translated text because Chinese does not segment words by spaces.
    """
    if pd.isna(text):
        return 0
    text = str(text)
    return len(re.sub(r"\s+", "", text))


def chinese_char_count(text):
    """Count CJK Unified Ideograph characters only."""
    if pd.isna(text):
        return 0
    text = str(text)
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def contains_any(text, markers):
    if pd.isna(text):
        return False
    text = str(text).lower()
    return any(marker.lower() in text for marker in markers)


EN_CUE_MARKERS = {
    "temporal": [
        "after", "before", "during", "since", "sudden", "suddenly",
        "day", "days", "week", "weeks", "month", "months", "year", "years",
        "old", "history of"
    ],
    "negation": [
        "no", "not", "without", "denies", "deny", "negative", "absent",
        "never", "neither"
    ],
    "modifier_or_severity": [
        "complete", "partial", "severe", "mild", "acute", "chronic",
        "progressive", "persistent", "recurrent", "normal", "abnormal",
        "elevated", "decreased", "increased"
    ],
    "management_or_treatment": [
        "next step", "treatment", "therapy", "management", "administer",
        "give", "surgery", "surgical", "operation", "procedure",
        "antibiotic", "anticoagulation", "heparin", "indomethacin"
    ],
    "safety_or_contraindication": [
        "contraindication", "contraindicated", "allergy", "allergic",
        "adverse", "toxicity", "history of", "risk", "retinal",
        "pregnant", "pregnancy"
    ],
}

ZH_CUE_MARKERS = {
    "temporal": [
        "后", "前", "期间", "以来", "突然", "天", "日", "周", "月", "年",
        "岁", "病史", "既往"
    ],
    "negation": [
        "无", "未", "不", "没有", "否认", "阴性", "缺乏"
    ],
    "modifier_or_severity": [
        "完全", "部分", "严重", "轻度", "急性", "慢性", "进行性",
        "持续", "反复", "正常", "异常", "升高", "降低", "增加", "减少"
    ],
    "management_or_treatment": [
        "下一步", "治疗", "处理", "管理", "给予", "使用", "手术",
        "抗生素", "抗凝", "肝素", "吲哚美辛"
    ],
    "safety_or_contraindication": [
        "禁忌", "过敏", "不良反应", "毒性", "风险", "视网膜",
        "妊娠", "怀孕", "既往史"
    ],
}


def add_basic_metrics(df, prefix, language):
    """
    Add basic surface-level length metrics.
    language='en' uses word counts as primary.
    language='zh' uses character counts as primary.
    """
    question_col = "Question"
    options_col = "Options"

    if question_col not in df.columns or options_col not in df.columns:
        raise ValueError(f"Expected columns 'Question' and 'Options'. Found: {list(df.columns)}")

    if language == "en":
        df[f"{prefix}_question_words"] = df[question_col].apply(english_word_count)
        df[f"{prefix}_options_words"] = df[options_col].apply(english_word_count)
        df[f"{prefix}_total_words"] = (
            df[f"{prefix}_question_words"] + df[f"{prefix}_options_words"]
        )

        df[f"{prefix}_question_nonspace_chars"] = df[question_col].apply(nonspace_char_count)
        df[f"{prefix}_options_nonspace_chars"] = df[options_col].apply(nonspace_char_count)
        df[f"{prefix}_total_nonspace_chars"] = (
            df[f"{prefix}_question_nonspace_chars"] + df[f"{prefix}_options_nonspace_chars"]
        )

    elif language == "zh":
        df[f"{prefix}_question_nonspace_chars"] = df[question_col].apply(nonspace_char_count)
        df[f"{prefix}_options_nonspace_chars"] = df[options_col].apply(nonspace_char_count)
        df[f"{prefix}_total_nonspace_chars"] = (
            df[f"{prefix}_question_nonspace_chars"] + df[f"{prefix}_options_nonspace_chars"]
        )

        df[f"{prefix}_question_chinese_chars"] = df[question_col].apply(chinese_char_count)
        df[f"{prefix}_options_chinese_chars"] = df[options_col].apply(chinese_char_count)
        df[f"{prefix}_total_chinese_chars"] = (
            df[f"{prefix}_question_chinese_chars"] + df[f"{prefix}_options_chinese_chars"]
        )

        # Also count English-like words in translated text, because medical abbreviations may remain.
        df[f"{prefix}_english_like_words"] = (
            df[question_col].apply(english_word_count) + df[options_col].apply(english_word_count)
        )
    else:
        raise ValueError("language must be 'en' or 'zh'.")

    return df


def add_cue_markers(df, prefix, markers):
    text = (df["Question"].fillna("").astype(str) + " " + df["Options"].fillna("").astype(str))
    for cue_type, cue_markers in markers.items():
        df[f"{prefix}_cue_{cue_type}"] = text.apply(lambda x: contains_any(x, cue_markers))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--english", required=True, help="Path to results_english_original.csv")
    parser.add_argument("--translated", required=True, help="Path to results_english_translated.csv")
    parser.add_argument("--outdir", default="linguistic_analysis", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    en = pd.read_csv(args.english)
    zh = pd.read_csv(args.translated)

    if len(en) != len(zh):
        raise ValueError(f"Files have different number of rows: English={len(en)}, translated={len(zh)}")

    # Add question IDs based on row order: Q1--Q100.
    en = en.copy()
    zh = zh.copy()
    en["qid"] = [f"Q{i+1}" for i in range(len(en))]
    zh["qid"] = [f"Q{i+1}" for i in range(len(zh))]

    # Correctness.
    en["en_correct"] = en["Is Correct"].apply(parse_bool)
    zh["zh_correct"] = zh["Is Correct"].apply(parse_bool)

    # Surface-level length metrics.
    en = add_basic_metrics(en, prefix="en", language="en")
    zh = add_basic_metrics(zh, prefix="zh", language="zh")

    # Cue marker detection.
    en = add_cue_markers(en, prefix="en", markers=EN_CUE_MARKERS)
    zh = add_cue_markers(zh, prefix="zh", markers=ZH_CUE_MARKERS)

    # Merge matched questions.
    merged = pd.merge(
        en,
        zh,
        on="qid",
        suffixes=("_enfile", "_zhfile")
    )

    # Length ratio: translated Chinese nonspace characters / English total words.
    # This is only a crude surface ratio, not semantic equivalence.
    merged["zh_chars_per_en_word"] = (
        merged["zh_total_nonspace_chars"] / merged["en_total_words"].replace(0, pd.NA)
    )

    # Correctness transitions.
    def transition(row):
        if row["en_correct"] and row["zh_correct"]:
            return "Correct in both"
        if (not row["en_correct"]) and (not row["zh_correct"]):
            return "Incorrect in both"
        if (not row["en_correct"]) and row["zh_correct"]:
            return "Incorrect only in English original"
        if row["en_correct"] and (not row["zh_correct"]):
            return "Incorrect only in EN-to-ZH translated"
        return "Unknown"

    merged["transition"] = merged.apply(transition, axis=1)

    # Save full matched metrics.
    full_out = os.path.join(args.outdir, "matched_surface_linguistic_metrics.csv")
    merged.to_csv(full_out, index=False)

    # Summary length table.
    length_summary = pd.DataFrame({
        "Metric": [
            "Mean English question-stem word count",
            "Mean English options word count",
            "Mean English total item word count",
            "Mean translated Chinese question-stem nonspace character count",
            "Mean translated Chinese options nonspace character count",
            "Mean translated Chinese total item nonspace character count",
            "Mean translated Chinese characters per English word",
        ],
        "Value": [
            merged["en_question_words"].mean(),
            merged["en_options_words"].mean(),
            merged["en_total_words"].mean(),
            merged["zh_question_nonspace_chars"].mean(),
            merged["zh_options_nonspace_chars"].mean(),
            merged["zh_total_nonspace_chars"].mean(),
            merged["zh_chars_per_en_word"].mean(),
        ]
    })

    length_summary["Value"] = length_summary["Value"].round(2)
    length_out = os.path.join(args.outdir, "length_summary.csv")
    length_summary.to_csv(length_out, index=False)

    # Transition summary.
    transition_summary = (
        merged["transition"]
        .value_counts()
        .rename_axis("Transition type")
        .reset_index(name="Count")
    )
    transition_out = os.path.join(args.outdir, "correctness_transition_summary.csv")
    transition_summary.to_csv(transition_out, index=False)

    # Cue summary across all 100 matched questions.
    cue_rows = []
    for cue_type in EN_CUE_MARKERS.keys():
        en_col = f"en_cue_{cue_type}"
        zh_col = f"zh_cue_{cue_type}"
        cue_rows.append({
            "Cue type": cue_type,
            "English marker present count": int(merged[en_col].sum()),
            "Translated Chinese marker present count": int(merged[zh_col].sum()),
            "Both present count": int((merged[en_col] & merged[zh_col]).sum()),
            "English only count": int((merged[en_col] & ~merged[zh_col]).sum()),
            "Translated Chinese only count": int((~merged[en_col] & merged[zh_col]).sum()),
            "Neither present count": int((~merged[en_col] & ~merged[zh_col]).sum()),
        })

    cue_summary = pd.DataFrame(cue_rows)
    cue_out = os.path.join(args.outdir, "cue_marker_summary.csv")
    cue_summary.to_csv(cue_out, index=False)

    # Changed correctness cases only.
    changed = merged[merged["en_correct"] != merged["zh_correct"]].copy()

    selected_cols = [
        "qid",
        "transition",
        "Question_enfile",
        "Options_enfile",
        "Correct Answer_enfile",
        "Model Answer_enfile",
        "Model Answer_zhfile",
        "Question_zhfile",
        "Options_zhfile",
        "en_question_words",
        "en_options_words",
        "en_total_words",
        "zh_question_nonspace_chars",
        "zh_options_nonspace_chars",
        "zh_total_nonspace_chars",
        "zh_chars_per_en_word",
    ]

    # Add cue columns.
    for cue_type in EN_CUE_MARKERS.keys():
        selected_cols.append(f"en_cue_{cue_type}")
        selected_cols.append(f"zh_cue_{cue_type}")

    changed_out = os.path.join(args.outdir, "changed_correctness_cases.csv")
    changed[selected_cols].to_csv(changed_out, index=False)

    # Print readable output.
    print("\n=== Length summary ===")
    print(length_summary.to_string(index=False))

    print("\n=== Correctness transition summary ===")
    print(transition_summary.to_string(index=False))

    print("\n=== Cue marker summary ===")
    print(cue_summary.to_string(index=False))

    print("\n=== Changed correctness cases ===")
    if len(changed) == 0:
        print("No changed correctness cases.")
    else:
        print(changed[["qid", "transition", "en_correct", "zh_correct"]].to_string(index=False))

    print("\nSaved files:")
    print(f"- {full_out}")
    print(f"- {length_out}")
    print(f"- {transition_out}")
    print(f"- {cue_out}")
    print(f"- {changed_out}")


if __name__ == "__main__":
    main()

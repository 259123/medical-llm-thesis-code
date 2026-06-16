# Thesis Project Code

This folder contains the code used for my master's thesis project on cross-linguistic robustness in medical large language model evaluation.

The project compares model performance on medical diagnostic questions across three language settings:

* English original questions
* English-to-Chinese translated questions
* Chinese native questions

The main model evaluated in the thesis is DeepSeek R1-0528. V4-Pro is included as a supplementary comparison.

## Requirements

The code was written in Python 3.

The main Python packages used include:

* pandas
* numpy
* scipy

Additional packages may be required depending on the specific evaluation script and local environment.

## File structure

The folder contains three main types of files:

* `evaluate_*.py`: scripts used to run model evaluation for different language settings and model versions.
* `analyze_*.py`: scripts used to analyze the generated results and calculate summary statistics.
* `results_*.csv`: output files containing model responses and evaluation results.
* `stats_*.txt`: summary statistics generated from the result files.

The `medqa/` folder contains the medical question data used in the experiments, if included.

## Usage

The general workflow is:

1. Run an evaluation script to generate a result file.
2. Run the corresponding analysis script to calculate summary statistics.
3. Check the generated `.csv` and `.txt` output files.

For example:

```bash
python evaluate_english_original.py
python analyze_english_original.py
```

Similar scripts are provided for the English-to-Chinese translated and Chinese native settings, as well as for the supplementary V4-Pro experiments.

Some file paths or filenames may need to be adjusted depending on the local environment.

## Outputs

The main outputs are:

* `results_*.csv`: detailed model outputs and correctness labels
* `stats_*.txt`: summary statistics used for analysis in the thesis
* `errors_v4pro.txt`: selected error cases for supplementary error analysis

## Notes

No API keys, passwords, or private credentials are included in this folder.

The code was written for thesis experimentation and analysis rather than as a general-purpose software package. The included scripts and output files correspond to the experiments discussed in the thesis.

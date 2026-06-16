import json
from typing import Dict, Any
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import csv

print("Loading environment variables...")
load_dotenv()
    
# Get API key from environment variable
api_key = os.getenv('DEEPSEEK_API_KEY')
print(f"API key loaded: {'*' * len(api_key[-4:]) + api_key[-4:]}")

def reason_with_deepseek(question: str, options: Dict[str, str]) -> str:
    """
    Uses DeepSeek API to reason about a medical question and select an answer.
    
    Args:
        question: The medical question text (in Chinese)
        options: Dictionary mapping option letters to answer choices
        
    Returns:
        Tuple of (predicted answer letter, reasoning)
    """
    print("\nCalling DeepSeek API...")
    print(f"Question: {question[:100]}...")
    print(f"Options: {options}")
    
    # Initialize OpenAI client with DeepSeek base URL
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # Format prompt with question and options in Chinese
    prompt = f"医学问题：{question}\n\n选项：\n"
    for opt_key, opt_val in options.items():
        prompt += f"{opt_key}. {opt_val}\n"
    prompt += "\n请仔细分析这道医学题目，考虑所需的相关医学知识、临床指南和逻辑推理。然后选择最合适的答案。请只回答选项字母（A、B、C或D）。"

    print(f"Generated prompt: {prompt[:200]}...")

    try:
        # Call API using OpenAI client
        print("Making API request...")
        completion = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract answer and reasoning
        content = completion.choices[0].message.content
        reasoning = completion.choices[0].message.reasoning_content
        
        # Get just the letter answer from the content
        answer = content.strip()[0]
        
        print(f"Received response - Answer: {answer}")
        print(f"Reasoning: {reasoning[:200]}...")
            
        return answer, reasoning

    except Exception as e:
        print(f"Error calling DeepSeek API: {e}")
        return "", ""

def evaluate_model_on_medqa(input_file: str, max_examples: int = None) -> None:
    """
    Evaluates model on medical QA data and saves results to CSV and statistics to TXT.
    
    Args:
        input_file: Path to JSONL file containing medical QA examples
        max_examples: Optional maximum number of examples to evaluate
    """
    print(f"\nStarting evaluation on {input_file}")
    if max_examples:
        print(f"Will process up to {max_examples} examples")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results_chinese_native_{timestamp}.csv'
    stats_file = f'stats_chinese_native_{timestamp}.txt'
    
    print(f"Results will be saved to: {results_file}")
    print(f"Statistics will be saved to: {stats_file}")
    
    # DEBUG: Get absolute path
    abs_results_file = os.path.abspath(results_file)
    abs_stats_file = os.path.abspath(stats_file)
    print(f"[DEBUG] Absolute path for CSV: {abs_results_file}")
    print(f"[DEBUG] Absolute path for TXT: {abs_stats_file}")
    
    correct = 0
    total = 0
    
    # Create CSV file for detailed results
    print("[DEBUG] Opening CSV file for writing...")
    with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
        print("[DEBUG] CSV file opened successfully")
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Options', 'Correct Answer', 'Model Answer', 
                        'Is Correct', 'Reasoning'])
        print("[DEBUG] CSV header written")
        csvfile.flush()  # Force write header
        
        # Process each example
        with open(input_file, encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if max_examples and line_num >= max_examples:
                    print(f"Reached maximum examples limit ({max_examples})")
                    break
                    
                if not line.strip():
                    continue
                    
                print(f"\nProcessing example {line_num + 1}")
                example = json.loads(line)
                
                # Format prompt
                prompt = example['question']
                options = example['options']
                
                # Get model prediction
                try:
                    pred, reasoning = reason_with_deepseek(prompt, options)
                    
                    # Check if correct
                    is_correct = pred == example['answer_idx']
                    if is_correct:
                        correct += 1
                        print("✓ Correct answer!")
                    else:
                        print(f"✗ Incorrect answer. Expected: {example['answer_idx']}, Got: {pred}")
                        
                    # Write result to CSV
                    writer.writerow([
                        example['question'],
                        str(example['options']),
                        example['answer_idx'],
                        pred,
                        is_correct,
                        reasoning
                    ])
                    
                    # Flush every 10 rows to ensure data is written
                    if (line_num + 1) % 10 == 0:
                        csvfile.flush()
                        print(f"[DEBUG] CSV flushed after {line_num + 1} examples")
                    
                except Exception as e:
                    print(f"Error processing example: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
                total += 1
                
                # Print progress
                if total % 10 == 0:
                    print(f"\nProcessed {total} examples...")
                    print(f"Current accuracy: {(correct/total):.2%}")
        
        # Final flush before closing
        csvfile.flush()
        print(f"[DEBUG] Final CSV flush completed. Total rows written: {total}")
    
    # Check if CSV file exists after closing
    print(f"[DEBUG] CSV file closed. Checking if file exists...")
    if os.path.exists(results_file):
        file_size = os.path.getsize(results_file)
        print(f"[DEBUG] ✓ CSV file exists! Size: {file_size} bytes")
    else:
        print(f"[DEBUG] ✗ WARNING: CSV file does NOT exist after closing!")
    
    # Calculate final statistics
    accuracy = correct / total if total > 0 else 0
    
    print("\nGenerating final statistics...")
    # Save statistics to text file
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Evaluation Statistics (Native Chinese MedQA-CN)\n")
        f.write("==============================================\n\n")
        f.write(f"Total examples processed: {total}\n")
        f.write(f"Correct answers: {correct}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.flush()
        print("[DEBUG] Stats file written and flushed")
    
    # Check if stats file exists
    if os.path.exists(stats_file):
        print(f"[DEBUG] ✓ Stats file exists!")
    else:
        print(f"[DEBUG] ✗ WARNING: Stats file does NOT exist!")
        
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {results_file}")
    print(f"Statistics saved to: {stats_file}")
    print(f"Final accuracy: {accuracy:.2%}")
    
    # Final verification
    print("\n[DEBUG] Final file check:")
    print(f"[DEBUG] ls command output:")
    os.system(f"ls -lh {results_file} {stats_file}")

import sys

if __name__ == "__main__":
    # Load and evaluate native Chinese MedQA-CN dataset
    dataset_path = "medqa/chinese_native_100.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file {dataset_path} not found")
        sys.exit(1)
        
    print(f"Evaluating {dataset_path}...")
    evaluate_model_on_medqa(dataset_path)  # Will run all 100 questions

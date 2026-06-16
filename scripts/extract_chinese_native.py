import json
import random
import os

# Configuration
INPUT_FILE = '/Users/heye/Desktop/finial thesis/MedQA/questions/Mainland/4_options/train.jsonl'
OUTPUT_FILE = '/Users/heye/Desktop/finial thesis/dataset/medqa/chinese_native_100.jsonl'
NUM_QUESTIONS = 100
RANDOM_SEED = 42  # For reproducibility

def main():
    print("=" * 60)
    print("MedQA-CN Question Extraction")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f" Error: Input file not found: {INPUT_FILE}")
        return
    
    # Read all questions
    print(f"\n Reading questions from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    
    print(f"✓ Loaded {len(questions)} total questions")
    
    # Check if we have enough questions
    if len(questions) < NUM_QUESTIONS:
        print(f" Error: Not enough questions. Need {NUM_QUESTIONS}, have {len(questions)}")
        return
    
    # Random sampling with seed for reproducibility
    print(f"\n Randomly sampling {NUM_QUESTIONS} questions (seed={RANDOM_SEED})")
    random.seed(RANDOM_SEED)
    sampled = random.sample(questions, NUM_QUESTIONS)
    
    # Verify format
    print(f"\n Verifying question format...")
    sample = sampled[0]
    required_fields = ['question', 'options', 'answer_idx']
    missing_fields = [field for field in required_fields if field not in sample]
    
    if missing_fields:
        print(f" Error: Missing required fields: {missing_fields}")
        return
    
    # Check options count
    num_options = len(sample['options'])
    if num_options != 4:
        print(f"  Warning: Expected 4 options, found {num_options}")
    
    print(f"✓ Format verified:")
    print(f"  - Required fields present: {required_fields}")
    print(f"  - Number of options: {num_options}")
    print(f"  - Answer format: {sample['answer_idx']}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        print(f"\n Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    # Save to output file
    print(f"\n Saving to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for q in sampled:
            # Only keep necessary fields
            output_q = {
                'question': q['question'],
                'options': q['options'],
                'answer_idx': q['answer_idx']
            }
            f.write(json.dumps(output_q, ensure_ascii=False) + '\n')
    
    print(f"✓ Successfully saved {len(sampled)} questions")
    
    # Display sample
    print(f"\n" + "=" * 60)
    print("Sample Question:")
    print("=" * 60)
    print(f"Question: {sample['question'][:100]}...")
    print(f"Options: {list(sample['options'].keys())}")
    print(f"  A: {sample['options']['A'][:50]}...")
    print(f"  B: {sample['options']['B'][:50]}...")
    print(f"  C: {sample['options']['C'][:50]}...")
    print(f"  D: {sample['options']['D'][:50]}...")
    print(f"Correct Answer: {sample['answer_idx']}")
    print("=" * 60)
    
    print(f"\n Extraction complete!")
    print(f" Summary:")
    print(f"  - Input: {len(questions)} questions")
    print(f"  - Sampled: {NUM_QUESTIONS} questions")
    print(f"  - Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

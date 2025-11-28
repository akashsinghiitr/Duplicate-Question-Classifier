import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os
import re
import json
from datetime import datetime

def load_model():
    """Load the latest BERT checkpoint"""
    results_dir = './results'
    checkpoints = [d for d in os.listdir(results_dir) if d.startswith('checkpoint-')]
    
    if checkpoints:
        checkpoint_numbers = [int(re.search(r'checkpoint-(\d+)', cp).group(1)) for cp in checkpoints]  # type:ignore
        latest_checkpoint_num = max(checkpoint_numbers)
        model_path = f'{results_dir}/checkpoint-{latest_checkpoint_num}'
        print(f"Loading latest checkpoint: {model_path}")
    else:
        raise FileNotFoundError("No checkpoints found in ./results directory")
    
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) # type:ignore
    model.eval()
    
    return tokenizer, model, device

def predict_duplicate(question1, question2, tokenizer, model, device):
    """
    Predict if two questions are duplicates using the trained BERT model
    
    Args:
        question1 (str): First question
        question2 (str): Second question
        tokenizer: BERT tokenizer
        model: Trained BERT model
        device: torch device (CPU/GPU)
        
    Returns:
        dict: Prediction results with probability
    """
    # Tokenize input
    inputs = tokenizer(
        question1, 
        question2,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Move to device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        # Probability of being duplicate (class 1)
        duplicate_prob = probabilities[0][1].item()
    
    return {
        'duplicate_probability': duplicate_prob
    }

def run_tests(test_cases_file='testing\\test_cases.json'):
    """Run test cases and write results to outputs.txt
    
    Args:
        test_cases_file (str): Path to JSON file containing test cases
    """
    
    # Load test cases from JSON file
    try:
        with open(test_cases_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        print(f"Loaded {len(test_cases)} test cases from {test_cases_file}\n")
    except FileNotFoundError:
        print(f"Error: Test cases file '{test_cases_file}' not found!")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{test_cases_file}'!")
        return
    
    # Load model
    print("Loading BERT model...")
    tokenizer, model, device = load_model()
    print(f"Using device: {device}\n")
    
    # Open output file
    with open('testing\\outputs.txt', 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("Duplicate Question Classifier - Test Results\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: BERT Fine-tuned\n")
        f.write(f"Device: {device}\n")
        f.write("=" * 80 + "\n\n")
        
        # Run tests
        for i, test in enumerate(test_cases, 1):
            result = predict_duplicate(test["q1"], test["q2"], tokenizer, model, device)
            prediction = "Duplicate" if result['duplicate_probability'] > 0.5 else "Not Duplicate"
            
            # Write to file
            f.write(f"Test Case {i}:\n")
            f.write(f"{'─' * 80}\n")
            f.write(f"Question 1: {test['q1']}\n")
            f.write(f"Question 2: {test['q2']}\n")
            f.write(f"Duplicate Probability: {result['duplicate_probability']:.4f}\n")
            f.write(f"Prediction: {prediction}\n")
            f.write(f"Expected: {test['expected']}\n")
            
            # Check if prediction matches expectation (exact match or contains expected)
            match = (
                prediction == test['expected'] or
                (prediction == "Duplicate" and test['expected'] in ["Duplicate", "Likely Duplicate"]) or
                (prediction == "Not Duplicate" and test['expected'] == "Not Duplicate")
            )
            f.write(f"Status: {'✅ PASS' if match else '❌ FAIL'}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print("\n✅ Test results written to outputs.txt")


if __name__ == "__main__":
    run_tests()

# Duplicate Question Classifier

This project implements a duplicate question classifier using the Quora Questions dataset.

## Use Case:

This classifier can help Q&A websites like Stack Overflow and Quora automatically detect and merge duplicate question threads. By consolidating similar questions, users can find answers more efficiently without getting lost in multiple threads, improving the overall user experience and reducing redundancy.

## Summary

This project demonstrates several approaches to automatically identifying duplicate questions, leveraging both classical ML and DL techniques. It covers:

- **Feature Engineering & Classical ML:** Extracting custom text features (common words, stopwords, fuzzy metrics, etc.), generating Word2Vec embeddings, and training an XGBoost classifier on these engineered features.
- **Deep Learning:** Building and training LSTM and Transformer models in PyTorch for sequence modeling.
- **Transfer Learning:** Fine-tuning a pre-trained BERT model for robust semantic understanding.

Each method is evaluated and compared, providing a comprehensive overview of strategies for duplicate detection in Q&A platforms.

## Dataset

- **Source:** Quora duplicate questions dataset (`quora_questions.csv`)
- **Columns:** `id`, `qid1`, `qid2`, `question1`, `question2`, `is_duplicate`

## Approaches

### 1. Feature Engineering + XGBoost

- **Preprocessing:** HTML tag removal, punctuation removal, URL removal, stopword filtering, tokenization.
- **Feature Extraction:** Custom features (common words, stopwords, fuzzy metrics, etc.) and Word2Vec embeddings.
- **Model:** XGBClassifier

### 2. PyTorch Deep Learning

- **Vocabulary:** Built from tokenized corpus.
- **Data Preparation:** Questions converted to index sequences, padded for batching.
- **Model:** Transformer-based classifier (`MyNN`)
- **Training:** BCEWithLogitsLoss, Adam optimizer

### 3. BERT Fine-Tuning (Most accurate till now ✅)

- **Tokenizer:** `bert-base-uncased`
- **Model:** `BertForSequenceClassification` (HuggingFace Transformers)
- **Training:** Trainer API, custom metrics (accuracy, F1)

## Model Performance

| Model                                                   | Accuracy | Precision | Recall | F1-Score |
| ------------------------------------------------------- | -------- | --------- | ------ | -------- |
| **XGBoost (Word2Vec + Features)**                       | 0.7957   | 0.7217    | 0.7300 | 0.7258   |
| **PyTorch Model**                                       | 0.7578   | 0.7339    | 0.5377     | 0.6207       |
| <span style="color:#90EE90;">**BERT Fine-tuned**</span> | 0.8628   | 0.7959    | 0.8518 | 0.8229   |

_Note: Run the evaluation cells in the notebook to see specific performance metrics. XGBoost reports all four metrics, while PyTorch models report accuracy only. BERT reports accuracy and F1-score._

## Custom Testing

A comprehensive test suite is available to evaluate the model on various question pairs, including edge cases and tricky scenarios.

### Running Tests

```bash
python tests.py
```

This will:

- Load the latest trained BERT checkpoint
- Run 30+ diverse test cases from `testing/test_cases.json`
- Generate a detailed report in `testing/outputs.txt`

### Test Categories

The test suite covers:

- **Clear Duplicates**: Paraphrased questions, synonyms, same intent
- **Clear Non-Duplicates**: Unrelated topics, different domains
- **Tricky Cases**: Negation, similar words with different intent, opposite questions, context variations
- **Edge Cases**: Identical questions, single word differences, number variations
- **Semantic Cases**: Technical vs layman terms, opposite actions
- **Complex Cases**: Comparison questions, multi-concept questions, process-oriented

### Custom Test Cases (Using BERT)

Add your own test cases by editing `testing/test_cases.json`:

```json
{
  "category": "Your Category",
  "q1": "First question",
  "q2": "Second question",
  "expected": "Duplicate/Not Duplicate"
}
```

## Requirements

- See requirements.txt for project dependencies.

## Structure

- `duplicate_classifier.ipynb`: Main notebook with all code and experiments
- `csvs/quora_questions.csv`: Dataset (link provided in [References](#References))
- `models/`: Saved models (Word2Vec, XGBoost, BERT checkpoints)  
  _Note: Model files are not uploaded due to large size. Please train and save models locally as needed._

## Dataset Link

- [Quora Question Pairs Dataset Source](https://www.kaggle.com/c/quora-question-pairs)

## Contributions

Contributions are welcome!  
If you have suggestions, improvements, or new models to add, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

# Duplicate Question Classifier

This project implements a duplicate question classifier using the Quora Questions dataset.

## Use Case:

This classifier can help Q&A websites like Stack Overflow and Quora automatically detect and merge duplicate question threads. By consolidating similar questions, users can find answers more efficiently without getting lost in multiple threads, improving the overall user experience and reducing redundancy.

## Summary

This project demonstrates several approaches, leveraging both classical ML and DL techniques. In particular, it covers:

- <ins>**Feature Engineering & Classical ML:**</ins> Extracting custom text features (common words, stopwords, fuzzy metrics, etc.), generating Word2Vec embeddings, and training an XGBoost classifier on these engineered features.
- <ins>**Deep Learning:**</ins> Building and training LSTM and Transformer-based architectures in PyTorch for sequence modeling.
- <ins>**Transfer Learning:**</ins> Fine-tuning a pre-trained BERT model for robust semantic understanding.

## Credit

I took help from the following research papers in identifying the optimal architecture for the DL approach:
- [Paper 1](https://www.researchgate.net/publication/343666306_Duplicate_Question_Detection_in_Question_Answer_Website_using_Convolutional_Neural_Network)
- [Paper 2](https://www.igminresearch.jp/articles/html/igmin135)
- [Paper 3](https://ijarcce.com/wp-content/uploads/2023/04/IJARCCE.2023.12369.pdf)

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
- **Model:** LSTM and Transformer-based classifiers (`MyNN`)
- **Training:** BCEWithLogitsLoss, Adam optimizer

### 3. BERT Fine-Tuning (Most accurate till now ✅)

- **Tokenizer:** `bert-base-uncased`
- **Model:** `BertForSequenceClassification` (HuggingFace Transformers)
- **Training:** Trainer API, custom metrics (accuracy, F1, precision, recall)

## Model Performance

| Model                                                                    | Training Rows | Accuracy | Precision | Recall | F1-Score |
| ------------------------------------------------------------------------ | ------------- | -------- | --------- | ------ | -------- |
| **XGBoost (Word2Vec + Features)**                                        | 40k       | 0.7957   | 0.7217    | 0.7300 | 0.7258   |
| **PyTorch Model**                                                        | ~400k      | 0.7578   | 0.7339    | 0.5377 | 0.6207   |
| <span style="color:#90EE90;">**BERT Fine-tuned**</span> (Best so far ✅) | 100k       | 0.8628   | 0.7959    | 0.8518 | 0.8229   |

## Custom Testing (For the BERT approach)

A comprehensive test suite is available to evaluate the model on various question pairs, including edge cases and tricky scenarios.

### Running Tests

```bash
python tests.py
```

This will:

- Load the latest trained BERT checkpoint
- Run 30 diverse test cases from `testing/test_cases.json`
- Generate a detailed report in `testing/outputs.txt`

### Test Categories

The test suite covers:

- **Clear Duplicates**: Paraphrased questions, synonyms, same intent
- **Clear Non-Duplicates**: Unrelated topics, different domains
- **Tricky Cases**: Negation, similar words with different intent, opposite questions, context variations
- **Edge Cases**: Identical questions, single word differences, number variations
- **Semantic Cases**: Technical vs layman terms, opposite actions
- **Complex Cases**: Comparison questions, multi-concept questions, process-oriented

### Custom Test Cases

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
- `csvs/quora_questions.csv`: Dataset (link provided in [Dataset Link](#References))
- `models/`: Saved models (Word2Vec, XGBoost, BERT checkpoints)  
  _Note: Model files are not uploaded due to large size. Please train and save models locally as needed._

## Dataset Link

- [Quora Question Pairs Dataset Source](https://www.kaggle.com/c/quora-question-pairs)

## Contributions

Contributions are welcome!  
If you have suggestions, improvements, or new models to add, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

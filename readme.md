# Duplicate Question Classifier

This project implements a duplicate question classifier using the Quora Questions dataset.

## **Use Case:**

This classifier can help Q&A websites like Stack Overflow and Quora automatically detect and merge duplicate question threads. By consolidating similar questions, users can find answers more efficiently without getting lost in multiple threads, improving the overall user experience and reducing redundancy.

## Summary

This project demonstrates several approaches to automatically identifying duplicate questions, leveraging both traditional machine learning and modern deep learning techniques. It covers:

- **Feature Engineering & Classical ML:** Extracting custom text features (common words, stopwords, fuzzy metrics, etc.), generating Word2Vec embeddings, and training a Random Forest classifier on these engineered features.
- **Deep Learning:** Building and training LSTM and Transformer models in PyTorch for sequence modeling.
- **Transfer Learning:** Fine-tuning a pre-trained BERT model for robust semantic understanding.

Each method is evaluated and compared, providing a comprehensive overview of strategies for duplicate detection in Q&A platforms.

## Dataset

- **Source:** Quora duplicate questions dataset (`quora_questions.csv`)
- **Columns:** `id`, `qid1`, `qid2`, `question1`, `question2`, `is_duplicate`

## Approaches

### 1. Feature Engineering + Random Forest

- **Preprocessing:** HTML tag removal, punctuation removal, URL removal, stopword filtering, tokenization.
- **Feature Extraction:** Custom features (common words, stopwords, fuzzy metrics, etc.) and Word2Vec embeddings.
- **Model:** RandomForestClassifier (scikit-learn)
- **Performance:** Accuracy ~0.81, Precision ~0.69, Recall ~0.88, F1 ~0.77

### 2. PyTorch Deep Learning

- **Vocabulary:** Built from tokenized corpus.
- **Data Preparation:** Questions converted to index sequences, padded for batching.
- **Models:**
  - LSTM-based classifier (`MyNN`)
  - Transformer-based classifier (`MyNN`)
- **Training:** BCEWithLogitsLoss, Adam optimizer
- **Evaluation:** Accuracy up to ~79% on train/test splits

### 3. BERT Fine-Tuning

- **Tokenizer:** `bert-base-uncased`
- **Model:** `BertForSequenceClassification` (HuggingFace Transformers)
- **Training:** Trainer API, custom metrics (accuracy, F1)
- **Performance:** Test accuracy up to ~84.5%, F1 ~80.6% (after 2 epochs)

## Usage

### Random Forest Prediction

```python
rf_model = joblib.load('../models/duplicate_.pkl')
sample1 = w2v_sentence(q1, q2)
y_pred = rf_model.predict([sample1])
print("Duplicate" if y_pred==1 else 'Non Duplicate')
```

### PyTorch Model Training

See notebook cells for model definition, training loop, and evaluation.

### BERT Prediction

```python
result = predict_duplicate("question1 text", "question2 text")
print(f"Duplicate Probability: {result['duplicate_probability']:.4f}")
```

## Requirements

- Python 3.12+
- pandas, numpy, scikit-learn, gensim, nltk, fuzzywuzzy, torch, transformers, joblib

## Structure

- `duplicate_classifier.ipynb`: Main notebook with all code and experiments
- `csvs/quora_questions.csv`: Dataset (link provided in [References](#References))
- `models/`: Saved models (Word2Vec, Random Forest, BERT checkpoints)  
  _Note: Model files are not uploaded due to large size. Please train and save models locally as needed._

## Notes

- For BERT, the training and evaluation steps make use of HuggingFace's Trainer API.
- Preprocessing and feature engineering are modular for easy experimentation.
- PyTorch models support both LSTM and Transformer architectures.

## References

- [Quora Question Pairs Dataset Source](https://www.kaggle.com/c/quora-question-pairs)
- [scikit-learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)

## Contributions

Contributions are welcome!  
If you have suggestions, improvements, or new models to add, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

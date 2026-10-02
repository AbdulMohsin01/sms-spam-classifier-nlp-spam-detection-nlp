# 📩 SMS Spam Detector

NLP classifier that flags spam text messages, with a Streamlit demo app.

## Pipeline
1. **Clean:** drop empty columns and 403 duplicate messages -> 5,169 unique messages (12.6% spam).
2. **Preprocess:** lowercase, keep alphanumeric tokens, remove stop words, Porter stemming.
3. **Features:** TF-IDF (3,000 terms).
4. **Model:** Logistic Regression (class-balanced), compared against Linear SVC, Random Forest and Naive Bayes using 5-fold stratified cross-validation.
5. **Save:** vectoriser + classifier stored together as one fitted scikit-learn `Pipeline`.

## Results
Hold-out test set (1,034 messages, stratified 80/20 split):

| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| 98.5% | 93.3% | 95.4% | 0.943 |

Cross-validated comparison (5 folds):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Linear SVC | 0.953 | 0.922 | 0.937 |
| **Logistic Regression** | 0.944 | 0.924 | 0.933 |
| Random Forest | 0.989 | 0.836 | 0.906 |
| Multinomial NB | 0.996 | 0.784 | 0.877 |

Logistic Regression was chosen for its near-top F1 *and* its probability output, which the app displays.
Naive Bayes has the highest precision but misses about 1 in 5 spam messages.

## Run it
```bash
pip install -r requirements.txt
python train.py --data data/spam.csv     # trains, compares, saves artifacts/
streamlit run app.py                     # web demo
```
Dataset: [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). A pre-trained model is included in `artifacts/`.

## Possible improvements
Character n-grams, probability-threshold tuning to trade precision for recall, and transformer embeddings.

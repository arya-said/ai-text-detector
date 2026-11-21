# AI-Generated Text Detector

Detects ChatGPT-written text from *how* it is written (stylometric features such as sentence-length variation, punctuation habits and function-word use), not from what it says. A Streamlit app lets you paste text and see the verdict, the model's confidence and a comparison with the average human and AI style.

**Gradient Boosting reaches 96.2% test accuracy (ROC-AUC 0.990) on 1,207 held-out texts. With all length-related features removed it still reaches 93.0%**, so most of the signal comes from writing style, not text length.

| ![App screenshot 1](WhatsApp%20Image%202026-11-21%20at%2010.55.15.jpeg) | ![App screenshot 2](WhatsApp%20Image%202026-11-21%20at%2010.55.40.jpeg) |
|---|---|

Built as a group project for the Machine Learning Lab at MIT World Peace University, Pune.

## How it works

1. **Data** (`prepare_dataset.py`): answers from the [HC3 corpus](https://huggingface.co/datasets/Hello-SimpleAI/HC3) (human answers vs ChatGPT answers to the same questions). Answers under 50 words and duplicates are dropped, and the classes are balanced at 3,000 texts each (6,000 total).
2. **Features** (`feature_engineering.py`): 66 stylometric features per text:
   - sentence length and its variance ("burstiness")
   - lexical diversity, including a fixed-window version that does not depend on text length
   - frequency of 40 common function words
   - rate of 10 punctuation marks
   - capitalisation, digits, line breaks, contractions, first-person words, long words
3. **Models** (`classify.py`): Naive Bayes, Logistic Regression, SVM, Bagging, Random Forest, AdaBoost and Gradient Boosting, compared on the same split.
4. **Uncertainty flag**: predictions with an AI-probability between 40% and 60% are reported as "Uncertain" instead of being forced into a verdict.
5. **App** (`app.py`): Streamlit front end using the saved model (`train_final_model.py`, Gradient Boosting with default settings).

## Results

All numbers are test-set results on 1,207 texts (4,793 for training). The split is grouped by question, so no question appears in both train and test.

| Model | Accuracy (all features) | ROC-AUC | Accuracy (length features removed) | ROC-AUC |
|---|---|---|---|---|
| Gradient Boosting | **96.2%** | 0.990 | **93.0%** | 0.988 |
| SVM | 94.7% | 0.990 | 92.7% | 0.981 |
| Logistic Regression | 94.7% | 0.984 | 91.0% | 0.964 |
| AdaBoost | 94.5% | 0.988 | 92.2% | 0.979 |
| Random Forest | 92.6% | 0.980 | 91.7% | 0.977 |
| Bagging | 91.4% | 0.972 | 89.5% | 0.964 |
| Naive Bayes | 85.0% | 0.922 | 83.6% | 0.908 |

**Length ablation.** Plain type-token ratio, word count and sentence count can act as stand-ins for text length, and ChatGPT answers differ in length from human ones. To check that the model is not just detecting length, `num_words`, `num_sentences` and `type_token_ratio` were removed and every model was retrained. Gradient Boosting dropped from 96.2% to 93.0%: length features help, but most of the accuracy remains without them. Without length features, the most important features (Random Forest) are punctuation frequency, line breaks, average sentence length, first-person word use and apostrophes.

**Uncertain zone.** With the 40-60% band, Gradient Boosting flags 1.8% of test texts as uncertain (3.4% in the length-removed run). AdaBoost flags most of its predictions as uncertain, so its "confident-only" accuracy is not comparable to the other models and is left out of the table above.

Full result tables are in [`results/`](results/). In the file names, `notune` means default hyperparameters on all features, and `nolength` means the ablation run.

## Run it

Requires Python 3.13 (other versions may work, but the saved model needs the pinned scikit-learn version).

```bash
pip install -r requirements.txt

# 1. Build the dataset from HC3 (downloads all.jsonl)
python prepare_dataset.py --output dataset.csv

# 2. Extract features
python feature_engineering.py --input dataset.csv --output features.csv

# 3. Compare models (add --tune for hyperparameter search)
mkdir -p results
python classify.py --input features.csv --out_prefix results/notune_

# 3b. Length ablation
python classify.py --input features.csv --drop num_words num_sentences type_token_ratio --out_prefix results/nolength_

# 4. Train and save the final model, then launch the app
python train_final_model.py --input features.csv --dataset dataset.csv
streamlit run app.py
```

`model.joblib` is included so the app runs without retraining. NLTK's tokenizer data is downloaded automatically on first run.

## Limitations

- Trained on one dataset (HC3) and one AI model (ChatGPT, early 2023). It has not been tested on newer models, other writing genres, or paraphrased and edited AI text.
- Results come from one train/test split with one random seed, so differences of a point or two between models are within noise.
- The 40-60% uncertain band is a fixed choice, not tuned.
- Texts under 50 words were excluded from training, so short inputs give unreliable results.
- The output is a statistical estimate, not proof of authorship, and should never be the only basis for accusing someone of using AI.

## Reference

Guo et al., *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection* (2023), the paper that introduced HC3.

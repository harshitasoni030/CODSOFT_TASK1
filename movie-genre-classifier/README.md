# Movie Genre Classifier

Predicts a movie's genre from its plot summary. TF-IDF features feed three classifiers (Naive Bayes, Logistic Regression, Linear SVM); the one with the best macro F1 on a held-out test set is saved with joblib and served through a Flask REST API with a small web frontend.

## Project structure

```
movie-genre-classifier/
├── app.py              # Flask API + serves the frontend
├── train.py            # Load data, compare models, save the best one
├── requirements.txt
├── data/               # Put the Kaggle CSV here
├── models/             # genre_model.joblib + metadata.json (created by train.py)
├── templates/index.html
└── static/{css/style.css, js/app.js}
```

## Setup

1. Download **Wikipedia Movie Plots** from Kaggle (`wiki_movie_plots_deduped.csv`) and place it in `data/`.
2. Create an environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Train (or point at a CSV elsewhere, e.g. `--data "D:\path\to\file.csv"`):
   ```
   python train.py
   ```
4. Run the app and open http://127.0.0.1:5000:
   ```
   python app.py
   ```

## How training works

- Uses the `Plot` and `Genre` columns. Genres marked `unknown` are dropped, the first listed genre becomes the label, and only the 10 most common genres are kept (`--top-genres N` to change).
- 80/20 stratified split. TF-IDF uses unigrams + bigrams, English stop words and sublinear TF.
- Logistic Regression and Linear SVM use `class_weight="balanced"` because *drama* dominates the dataset.
- **Selection metric: macro F1**, which treats every genre equally. Weighted F1 and accuracy are also reported.
- Confidence comes from `predict_proba`, so it is shown for Naive Bayes and Logistic Regression. Linear SVM has no probabilities, so the app shows the genre without a confidence value.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Server and model status |
| GET | `/api/model-info` | Metrics for all compared models |
| POST | `/api/predict` | Predict a genre |

```
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
  -d "{\"plot\": \"A detective hunts a serial killer across a rain-soaked city before he strikes again.\"}"
```

Response:
```json
{"genre": "crime", "confidence": 0.41, "model": "Logistic Regression",
 "top_predictions": [{"genre": "crime", "confidence": 0.41}]}
```

Errors return `{"error": "..."}` with status 400 (bad input), 413 (too long), 422 (under 10 words), 503 (model not trained) or 500.

## Production note

`gunicorn app:app` (Linux/macOS) replaces the Flask dev server.

## Ideas to extend

Multi-label prediction, word embeddings, hyperparameter search with `GridSearchCV`, Docker.

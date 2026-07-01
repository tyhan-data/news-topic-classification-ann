# 🗞️ NewsCortex — AI News Desk

> A TF-IDF + deep neural network pipeline that reads a raw news headline and
> instantly files it under the correct desk: **World, Sports, Business, or Sci/Tech.**

Built with **TensorFlow/Keras**, **scikit-learn**, and **Streamlit** — an
end-to-end text classification project, from feature extraction to a
deployed, interactive interface.

---

## 🎯 What it does

Paste in a headline or a few lines of an article, and NewsCortex returns:

- The predicted **desk** (category) with a confidence score
- A full probability breakdown across all four classes
- A live, code-driven inspection panel showing the model's actual layers and
  the vectorizer's actual configuration — not hardcoded docs, read straight
  from the loaded objects

## 🧠 How it works

```
Raw text
   │
   ▼
TF-IDF Vectorizer  ──  10,000 features · unigrams + bigrams · sublinear TF · L2 norm
   │
   ▼
Dense Neural Network
   ├─ Dense(256) → BatchNorm → ReLU → Dropout(0.3)
   ├─ Dense(256) → BatchNorm → ReLU
   ├─ Dense(128) → BatchNorm → ReLU → Dropout(0.3)
   ├─ Dense(128) → BatchNorm → ReLU
   ├─ Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
   ├─ Dense(64)  → BatchNorm → ReLU
   └─ Dense(4, softmax)
   │
   ▼
P(World) · P(Sports) · P(Business) · P(Sci/Tech)
```

The network is a deliberately deep **dense (fully-connected) classifier**
rather than the bag-of-words count you might expect — six hidden blocks with
batch normalization and dropout for regularization, ending in a 4-way
softmax.

## 📁 Project structure

```
news_cortex_app/
├── app.py                      # Streamlit application (fully commented)
├── ag_news_classifier.keras    # Trained Keras model
├── tfidf_vectorizer.pkl        # Fitted scikit-learn TfidfVectorizer
├── requirements.txt            # Pinned dependencies
├── .streamlit/
│   └── config.toml             # Theme (colors/fonts) for the deployed app
└── README.md
```

## 🚀 Run it locally

```bash
# 1. Clone / open this folder, then create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. The model and vectorizer load
once and are cached for the life of the server (`@st.cache_resource`), so
predictions after the first one are near-instant.

## ☁️ Deploy for free (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo (include the `.keras` and
   `.pkl` files — they're well under GitHub's 100MB file limit).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click **New app**.
3. Point it at your repo, branch, and `app.py`. Click **Deploy**.
4. Streamlit Cloud reads `requirements.txt` and `.streamlit/config.toml`
   automatically — no extra configuration needed.

(Render, Hugging Face Spaces, and Railway work the same way — point them
at `app.py` and `requirements.txt`.)

## 🛠️ Tech stack

| Layer | Tool |
|---|---|
| UI / app framework | Streamlit |
| Feature extraction | scikit-learn `TfidfVectorizer` |
| Model | TensorFlow / Keras `Sequential` |
| Serialization | `model.save()` (`.keras`) + `joblib` (`.pkl`) |

## 🔭 Possible next steps

- Swap the TF-IDF + dense network for a fine-tuned transformer (e.g.
  DistilBERT) and compare accuracy vs. latency.
- Add a confusion matrix / evaluation tab using a held-out test set.
- Batch mode: upload a CSV of headlines and classify them all at once.
- Track real prediction logs to monitor model drift over time.

## 📄 License

This project is provided as a portfolio / demonstration piece. Feel free to
fork and adapt it — just keep the attribution if you reuse the design.

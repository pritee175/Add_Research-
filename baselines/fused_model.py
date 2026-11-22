"""Fused promotional-vs-preventive classifier: TF-IDF + logistic regression on transcript/screen text, logistic regression
on frozen CLIP image embeddings, probabilities averaged. Either modality alone also works (image-only is what a brand-new
video gets when no transcript is supplied). Never takes labels or answer-key fields as input."""
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class FusedModel:
    def fit(self, texts, X_img, y):
        self.w = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r'(?u)\b\w+\b')
        self.c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=60000)
        Xw = self.w.fit_transform(texts); Xc = self.c.fit_transform(texts)
        self.n_word = Xw.shape[1]
        self.lr_text = LogisticRegression(C=10, max_iter=2000, class_weight='balanced').fit(hstack([Xw, Xc]).tocsr(), y)
        self.lr_img = LogisticRegression(C=10, max_iter=2000, class_weight='balanced').fit(X_img, y)
        return self

    def p_text(self, text):  # probability of promotional from text only
        return float(self.lr_text.predict_proba(hstack([self.w.transform([text]), self.c.transform([text])]).tocsr())[0, 1])

    def p_img(self, x):  # x: 1-D L2-normalised CLIP embedding
        return float(self.lr_img.predict_proba(np.asarray(x).reshape(1, -1))[0, 1])

    def predict(self, text=None, x_img=None):
        """Returns (label, p_promotional, parts). Averages whichever modalities are given."""
        parts = {}
        if text is not None and text.strip(' |'): parts['text'] = self.p_text(text)
        if x_img is not None: parts['image'] = self.p_img(x_img)
        if not parts: raise ValueError('need text and/or an image embedding')
        p = sum(parts.values()) / len(parts)
        return ('promotional' if p > .5 else 'preventive'), p, parts

    def top_words(self, text, k=6):
        """Words in this text pushing hardest toward promotional (+) and preventive (-): tf-idf value x coefficient."""
        v = self.w.transform([text]); coef = self.lr_text.coef_[0][:self.n_word]
        names = self.w.get_feature_names_out(); idx = v.nonzero()[1]
        contrib = sorted(((float(v[0, i] * coef[i]), names[i]) for i in idx), reverse=True)
        return [n for c, n in contrib[:k] if c > 0], [n for c, n in contrib[::-1][:k] if c < 0]

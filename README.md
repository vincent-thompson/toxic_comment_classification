# Toxic Comment Classification


### Problem Motivation
- Social media usage is at an all time high, with the average US adult social networking user spending 7 more minutes per day on social networks than in 2019.
- Enormous increase in User Generated Contents (UGC) on other online platforms such as newsgroups, blogs, and online forums.
- Heightened need to cut down on toxicity and abusive behavior (According to a study by McAfee, over 87% of teens have observed cyberbullying online)

### Tools Used
- Dataset: [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- Comment pre-processing: NLTK, spaCy, regex
- Feature Engineering: Scikit-learn, TF-IDF vectorization
- Algorithms: Scikit-learn (Logistic Regression, Naive Bayes, SVM, Random Forest)
- Web Application/Project demo: Streamlit

### Data
Wikipedia text comments, which were labeled by human raters for toxic behavior, according to 6 categories
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate
Comments can belong to one category, multiple categories, or none at all.  The goal of this project was to create a model which predicts probability for each type of toxicity for each comment.

### Preprocessing
Before creating a vectorized representation of each comment to feed to various classification algorithms, I preprocessed the data by performing the following operations (code containing functions for preprocessing an individual string or an entire pandas dataframe can be found in my preprocessing.py file):
- Remove accented characters (using unicodedata python module)
- Convert to all lowercase (using python's built-in .lower() method)
- Expand contractions (The function for finding and replacing contractions can be found in the file expand_contractions.py in this repo.  The function uses regex and a python dictionary of common contractions, courtesy of this StackOverflow [post](https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python))
- Remove special characters (using regex)
- Remove punctuation (using python's built-in .join() method)
- Lemmatize text (using spaCy)
- Remove stopwords (Used NLTK's [list](https://gist.github.com/sebleier/554280) of English stopwords)
- Remove extra whitespace and tabs (using regex)

### Feature extraction
TFIDF (Term Frequency - Inverse Document Frequency) Vectorization is a simple yet effective method to transform a corpus of comments from a list of strings to a suitable input for classification algorithms.  

<img src="https://render.githubusercontent.com/render/math?math=P(Identical \ Twin \ | \ Twin) = \frac{P(Twin \ | \ Identical \ Twin) \times P(Identical \ Twin)}{P(Twin)}">

<img src="https://render.githubusercontent.com/render/math?math=tfidf(t,d,D) = tf(t,d) \times idf(t,D)">
tf idf (t, d, D) = tf(t, d) $\frac{d}{D}$ idf(t, D)



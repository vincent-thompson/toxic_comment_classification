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

- The term frequency of a word in a comment is simply the raw count of times that word appears in the comment.  
- The inverse document frequency for a word is calculated by taking the total number of comments in our corpus, dividing it by the number of comments that contain the word, and taking the log of the result.  If the word is very common across the corpus, the IDF approaches 0, and if it is very rare it will approach 1.

The TF-IDF then, is the product of these two values, resulting in a metric that gives more weight to words that appear many times in a comment, but less weight to words that are more common across the corpus and not likely to be strong indicators of tone.

<img src="https://render.githubusercontent.com/render/math?math=tfidf(t,d,D) = tf(t,d) \times idf(t,D)">
Where:
<img src="https://render.githubusercontent.com/render/math?math=tf(t, d) = log(1{+}freq(t,d))">
<img src="https://render.githubusercontent.com/render/math?math=idf(t, D) = log(\frac{N}{count(d \in D : t \in d})">

Using scikit-learn, we can generate a Document Term Matrix where the features for each comment are its TF-IDF values corresponding to each word in the corpus.  

The TF-IDF technique has long been replaced by more sophisticated modeling techniques that capture the semantic relationships between words (Word2Vec/Glove, Transformer Models, ELMO/BERT, etc), but this project is restricted to the token representation.

### Model Selection and Performance

The metric chosen to evaluate the performance of the various classification models was the [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve), which measures how well a model can detect true positives (recall) while avoiding false positives.  In other words, we want to catch as high a percentage of true toxic comments as possible without them falling through the cracks, but we don't want to have to flag a large portion of the dataset to do so (recall is important, but we could flag every comment as toxic and have a perfect recall, which would not be a useful model)

The classification algorithms that were tested include:
- Logistic Regression
- Naive Bayes
- Linear Support Vector Machine
- Random Forest
- Ensemble learning (Soft Voting Classifier)






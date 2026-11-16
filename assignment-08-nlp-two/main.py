import re
from collections import Counter
from pathlib import Path
import nltk
import numpy as np
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from wordcloud import WordCloud

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
try:
    stops = set(stopwords.words('english'))
except LookupError:
    stops = set(ENGLISH_STOP_WORDS)

def safe_words(text):
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^\w\s]", text)

def safe_sentences(text):
    try:
        return sent_tokenize(text)
    except LookupError:
        return [part.strip() + '.' for part in text.split('.') if part.strip()]

def safe_lemma(word):
    try:
        return WordNetLemmatizer().lemmatize(word)
    except LookupError:
        return word

paragraph = ('Cloud computing provides flexible access to computing resources through the internet. '
             'Students use cloud services to host web applications and store project files. '
             'Virtual machines help developers test software without buying powerful computers. '
             'Cloud platforms can scale applications when more visitors arrive. '
             'Security teams protect accounts by limiting permissions and checking activity. '
             'Learning cloud computing helps students build practical technical skills.')
cleaned = re.sub(r'[^a-z\s]', '', paragraph.lower())
words = safe_words(cleaned)
filtered = [w for w in words if w not in stops]
print('Q1: Lowercase and cleaned', cleaned)
print('Sentences', safe_sentences(paragraph))
print('Words', words)
print('split()', paragraph.split())
print('word_tokenize()', safe_words(paragraph))
print('split keeps punctuation attached; word_tokenize commonly splits punctuation into separate tokens.')
print('Without stopwords', filtered)
print('Frequency distribution', Counter(filtered).most_common())

alpha_words = re.findall(r'\b[a-zA-Z]+\b', paragraph.lower())
content_words = [w for w in alpha_words if w not in stops]
stemmer = PorterStemmer()
lemma = WordNetLemmatizer()
print('Q2: Extracted words', alpha_words)
print('Stems', [stemmer.stem(w) for w in content_words])
print('Lemmas', [safe_lemma(w) for w in content_words])
print('Stemming is useful for compact search features; lemmatization is preferred for more readable, dictionary-based word forms.')

texts = ['Cloud platforms support scalable web hosting and fast deployment.', 'Cybersecurity tools protect networks, devices and user accounts.', 'Machine learning models identify patterns in historical datasets.']
count_vectorizer = CountVectorizer(stop_words='english')
bow = count_vectorizer.fit_transform(texts)
print('Q3: Bag of Words terms', count_vectorizer.get_feature_names_out())
print('Bag of Words matrix', bow.toarray())
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(texts)
print('TF-IDF matrix', tfidf_matrix.toarray())
for i, text in enumerate(texts):
    order = np.argsort(tfidf_matrix[i].toarray().ravel())[::-1][:3]
    print('Text', i + 1, 'Top 3 keywords', [(tfidf.get_feature_names_out()[j], round(float(tfidf_matrix[i, j]), 3)) for j in order])
print('The top TF-IDF keywords are relatively distinctive to each short document.')

technology_a = 'Artificial intelligence learns patterns from data. Machine learning models recognize images. Neural networks predict useful outputs. AI applications automate analysis and support human decisions.'
technology_b = 'Blockchain stores records in shared ledgers. Distributed networks validate transactions. Cryptographic methods protect recorded information. Blockchain applications support tracking and verification.'
def tokens(text):
    return set(w for w in re.findall(r'[a-z]+', text.lower()) if w not in stops)
a, b = tokens(technology_a), tokens(technology_b)
jaccard = len(a & b) / len(a | b)
features = TfidfVectorizer(stop_words='english').fit_transform([technology_a, technology_b])
cosine = cosine_similarity(features)[0, 1]
print('Q4: Jaccard', jaccard, 'Cosine', cosine)
print('Jaccard treats all unique words equally; TF-IDF cosine accounts for how informative words are across the two texts.')

reviews = ['The app is easy to use and works wonderfully.', 'Customer support was terrible and the app crashes.', 'The product arrived on Tuesday and has a blue case.']
positive_reviews = []
for review in reviews:
    sentiment = TextBlob(review).sentiment
    label = 'Positive' if sentiment.polarity > 0.05 else 'Negative' if sentiment.polarity < -0.05 else 'Neutral'
    print('Q5:', review, 'Polarity', sentiment.polarity, 'Subjectivity', sentiment.subjectivity, 'Class', label)
    if label == 'Positive':
        positive_reviews.append(review)
if positive_reviews:
    filename = Path(__file__).resolve().parent / 'positive_reviews_wordcloud.png'
    WordCloud(width=900, height=400, background_color='white').generate(' '.join(positive_reviews)).to_file(str(filename))
    print('Saved', filename.name)
print('Q6: Run train_text_model.py to train the optional TensorFlow text generator.')

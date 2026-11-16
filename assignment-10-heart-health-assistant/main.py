import json
import re
from pathlib import Path
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

base = Path(__file__).resolve().parent
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    stop_words = set(ENGLISH_STOP_WORDS)
lemmatizer = WordNetLemmatizer()
try:
    lemmatizer.lemmatize('hearts')
except LookupError:
    lemmatizer = None


def preprocess(text):
    tokens = re.findall(r'[a-z]+', text.lower())
    return ' '.join((lemmatizer.lemmatize(word) if lemmatizer else word) for word in tokens if word not in stop_words)


def load_json(name):
    return json.loads((base / name).read_text(encoding='utf-8'))

corpus = pd.read_csv(base / 'intents.csv')
knowledge_base = load_json('knowledge_base.json')
taxonomy = load_json('taxonomy.json')
ontology = load_json('ontology.json')
assert len(corpus) >= 40 and corpus['intent'].nunique() >= 5
assert len(taxonomy) >= 4 and all(len(groups) >= 2 for groups in taxonomy.values())
assert len(ontology['concepts']) >= 8 and len(ontology['relationships']) >= 10

train, test = train_test_split(corpus, test_size=0.25, random_state=42, stratify=corpus['intent'])
vectorizer = TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train['text'])
X_test = vectorizer.transform(test['text'])
model = LogisticRegression(max_iter=1500, random_state=42)
model.fit(X_train, train['intent'])
predictions = model.predict(X_test)
p, r, f1, _ = precision_recall_fscore_support(test['intent'], predictions, average='weighted', zero_division=0)
print('Evaluation on held-out data')
print('Accuracy:', round(accuracy_score(test['intent'], predictions), 3))
print('Precision:', round(p, 3), 'Recall:', round(r, 3), 'F1:', round(f1, 3))
print('Confusion matrix labels:', model.classes_.tolist())
print(confusion_matrix(test['intent'], predictions, labels=model.classes_))
print(classification_report(test['intent'], predictions, zero_division=0))

emergency_terms = ['severe chest pain', 'crushing chest pain', 'chest pain with shortness of breath', 'sudden difficulty breathing', 'fainted and not breathing', 'unresponsive', 'possible heart attack']
unsupported_terms = ['diabetes medication', 'insulin dose', 'skin rash', 'broken bone', 'eye infection', 'toothache']

def respond(query):
    text = query.lower().strip()
    if not text:
        return {'rule': 'empty_input', 'response': 'Please enter a question about general heart health.'}
    if any(phrase in text for phrase in emergency_terms):
        return {'rule': 'emergency_escalation', 'response': 'This may be a medical emergency. Contact your local emergency services immediately and seek urgent medical help. Do not rely on this assistant for emergency care.'}
    if any(phrase in text for phrase in unsupported_terms):
        return {'rule': 'unsupported_query', 'response': 'That topic is outside this heart-health information assistant. Please consult an appropriate healthcare professional.'}
    if text in {'hello', 'hi', 'hey', 'good morning'}:
        return {'rule': 'greeting', 'response': 'Hello. Ask about heart health, prevention, activity, nutrition, screening or care.'}
    if any(phrase in text for phrase in ['prescribe', 'what dose', 'change my medication', 'stop my medicine', 'diagnose me']):
        return {'rule': 'clinical_boundary', 'response': 'I cannot diagnose conditions, prescribe treatments, or change medication. Please speak with a licensed healthcare professional.'}
    features = vectorizer.transform([query])
    if features.nnz == 0:
        return {'rule': 'unsupported_query', 'response': 'I could not match your question to the supported heart-health topics.'}
    probabilities = model.predict_proba(features)[0]
    winner_index = probabilities.argmax()
    intent = model.classes_[winner_index]
    confidence = float(probabilities[winner_index])
    if confidence < 0.14:
        return {'rule': 'low_confidence', 'confidence': round(confidence, 3), 'response': 'I am not confident about that question. Please rephrase it using a topic such as heart risk, food, movement, or screening.'}
    return {'rule': 'knowledge_retrieval', 'intent': intent, 'confidence': round(confidence, 3), 'response': knowledge_base[intent]}


def interactive_colab():
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    entry = widgets.Textarea(placeholder='Ask a heart health question', description='Question:', layout=widgets.Layout(width='85%', height='80px'))
    submit = widgets.Button(description='Ask assistant', button_style='primary')
    result = widgets.Output()
    def run(_):
        with result:
            clear_output()
            answer = respond(entry.value)
            print(answer['response'])
            print('Rule:', answer['rule'])
            if 'intent' in answer:
                print('Intent:', answer['intent'], 'Confidence:', answer['confidence'])
    submit.on_click(run)
    display(entry, submit, result)


def cli():
    print('Heart Health Information Assistant | Roll 1024160112')
    print('Type exit to quit. For information only, not diagnosis or emergency care.')
    while True:
        try:
            query = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {'exit', 'quit'}:
            break
        answer = respond(query)
        print('Assistant:', answer['response'])
        print('Rule:', answer['rule'], 'Intent:', answer.get('intent', 'none'), 'Confidence:', answer.get('confidence', 'n/a'))


if __name__ == '__main__':
    import sys
    if '--demo' in sys.argv:
        samples = [group[0] for group in [corpus[corpus['intent'] == name]['text'].tolist() for name in sorted(corpus['intent'].unique())]]
        samples += ['how might walking protect my cardiovascular system', 'what should i do about a toothache', 'heart elephant melody', 'zebra quantum orbit', 'severe chest pain']
        for question in samples:
            print('Query:', question)
            print(respond(question))
    elif '--colab' in sys.argv:
        interactive_colab()
    else:
        cli()

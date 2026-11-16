import re
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer, RegexpStemmer, WordNetLemmatizer
from nltk.util import bigrams

for package in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(package, quiet=True)

roll_number = '1024160112'
assert 'generate_dataset' in globals() and 'process_ticket' in globals(), 'Run the provided UCS420_LA9_Starter.ipynb cells first.'
assert 'TEST_LINES' in globals(), 'Run starter cells to provide TEST_LINES.'

if 'tickets' not in globals():
    raise RuntimeError('Set tickets to the starter notebook generated ticket collection, without changing generate_dataset().')

raw_tickets = [str(t) for t in tickets]
processed = [process_ticket(ticket) for ticket in raw_tickets]
print('Original starter functions were used; check the starter output field names before processing below.')


def extract_fields(result):
    if isinstance(result, dict):
        names = {
            'tokens': ['tokens', 'all_tokens'],
            'sentences': ['sentences'],
            'punctuation': ['punctuation_tokens', 'punct_tokens'],
            'stopwords': ['stopwords', 'stopword_tokens'],
            'final': ['final_tokens', 'filtered_tokens']
        }
        value = {}
        for key, options in names.items():
            found = next((result[n] for n in options if n in result), None)
            if found is None:
                raise KeyError(f'Starter output for {key} is not recognized. Check the returned process_ticket() keys: {list(result)}')
            value[key] = found
        return value
    raise TypeError('process_ticket() must return named fields. Match its actual starter output to extract_fields().')

records = [extract_fields(result) for result in processed]
summary = pd.DataFrame([{
    'ticket': i + 1,
    'sentences': len(r['sentences']),
    'tokens': len(r['tokens']),
    'punctuation': len(r['punctuation']),
    'stopwords': len(r['stopwords']),
    'unique_tokens': len(set(r['tokens'])),
    'final_tokens': len(r['final'])
} for i, r in enumerate(records)])
totals = summary.drop(columns='ticket').sum(numeric_only=True).to_dict()
summary = pd.concat([summary, pd.DataFrame([{'ticket': 'Total', **totals}])], ignore_index=True)
print(summary.to_string(index=False))
for key in ['tokens', 'punctuation', 'stopwords', 'final']:
    print(key, records[0][key])

patterns = {
    'long_words': r'\b[A-Za-z]{6,}\b',
    'numbers': r'\b\d+(?:\.\d+)*\b',
    'capitalized': r'\b[A-Z][a-z]+\b',
    'vowel_words': r'\b[AEIOUaeiou][A-Za-z]*\b'
}

def redact(text):
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '<EMAIL>', text)
    text = re.sub(r'(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)', '<PHONE>', text)
    return text

for ticket in raw_tickets + list(TEST_LINES):
    print('Regex extraction:', {name: re.findall(pattern, ticket) for name, pattern in patterns.items()})
    print('Replaced:', redact(ticket))

pattern = r"https?://\S+|www\.\S+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\+?\d[\d ()-]{8,}\d|\b\d+(?:\.\d+)+\b|\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b|\d+|[^\w\s]"
tokenizer = RegexpTokenizer(pattern)

def my_tokenize(text):
    return tokenizer.tokenize(text)

for line in list(TEST_LINES) + raw_tickets:
    print('Custom:', my_tokenize(line))
    print('NLTK:', word_tokenize(line))
print('Custom tokenization preserves hyphenated strings, versions and emails as single tokens; word_tokenize often separates these. Exact differences are visible above.')

final_tokens = sorted(set(str(t).lower() for r in records for t in r['final']))
porter = PorterStemmer()
lancaster = LancasterStemmer()
regex_suffix = r'(ing|ed|ly)$'
regex_stemmer = RegexpStemmer(regex_suffix, min=4)
lemmatizer = WordNetLemmatizer()
tagged = dict(pos_tag(final_tokens))

def wordnet_pos(tag):
    return {'J': 'a', 'V': 'v', 'N': 'n', 'R': 'r'}.get(tag[0], 'n')

comparisons = pd.DataFrame([{'word': word, 'porter': porter.stem(word), 'lancaster': lancaster.stem(word), 'regexp': regex_stemmer.stem(word), 'lemma': lemmatizer.lemmatize(word, wordnet_pos(tagged[word]))} for word in final_tokens])
different = comparisons[comparisons[['porter', 'lancaster', 'regexp', 'lemma']].nunique(axis=1) > 1]
print('Stemmer differences:', different.to_string(index=False))
print('Regex suffix is ing, ed, or ly; inspect dataset tokens ending in those suffixes to justify the choice.')

S1 = [str(t).lower() for r in records for t in r['tokens']]
punctuation = set(str(t).lower() for r in records for t in r['punctuation'])
S2 = [t for t in S1 if t not in punctuation and re.search(r'\w', t)]
S3 = [str(t).lower() for r in records for t in r['final']]
S4 = [porter.stem(t) for t in S3]
for name, corpus in [('S1', S1), ('S2', S2), ('S3', S3), ('S4', S4)]:
    print(name, 'total', len(corpus), 'unique', len(set(corpus)), 'TTR', round(len(set(corpus)) / max(len(corpus), 1), 4))
for name, corpus in [('S1', S1), ('S3', S3)]:
    print(name, 'Top words:', Counter(corpus).most_common(10))
    print(name, 'Top bigrams:', Counter(bigrams(corpus)).most_common(5))
frequencies = [v for _, v in Counter(S3).most_common()]
plt.plot(range(1, len(frequencies) + 1), frequencies, marker='.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('Zipf plot for S3')
plt.tight_layout()
plt.savefig('zipf_plot.png')
plt.show()
print('TTR changes as punctuation and stopwords are removed, then stems merge related word forms.')

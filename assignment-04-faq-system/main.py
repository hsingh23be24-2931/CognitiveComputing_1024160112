from pathlib import Path
import re
import pandas as pd

roll = '1024160112'
categories = ['billing', 'account', 'general']
fixed_entries = [
    {'question': 'what is the annual fee', 'answer': 'The annual fee is Rs 500.', 'keywords': 'fee cost price charge', 'category': 'billing'},
    {'question': 'how to reset password', 'answer': 'Go to Settings > Reset Password.', 'keywords': 'password reset login', 'category': 'account'},
    {'question': 'what are your working hours', 'answer': 'We are open 9 AM to 5 PM.', 'keywords': 'hours timing open time', 'category': 'general'},
    {'question': 'how can i pay the fee', 'answer': 'You can pay via UPI, card, or net banking.', 'keywords': 'pay payment upi fee', 'category': 'billing'}
]
personalized = [
    {'question': 'how can i change my account username', 'answer': 'Open Profile Settings and choose Edit Username.', 'keywords': 'username profile change', 'category': 'account'},
    {'question': 'is the support desk open on holidays', 'answer': 'Holiday support hours are posted on the notice board.', 'keywords': 'holiday support schedule', 'category': 'general'}
]
for digit, entry in zip(roll[-2:], personalized):
    assert entry['category'] == categories[int(digit) % 3]
df = pd.DataFrame(fixed_entries + personalized)
print('Q1: FAQ DataFrame\n', df.to_string(index=False))

def score(query, data):
    query_words = set(re.findall(r'\w+', query.lower()))
    results = data.copy()
    results['confidence'] = results.apply(lambda row: len(query_words & set(re.findall(r'\w+', row['question'] + ' ' + row['keywords']))) / max(len(query_words), 1), axis=1)
    return results[results['confidence'] > 0].sort_values('confidence', ascending=False)

print('Q2: Ranked matches\n', score('pay fee', df).to_string(index=False))

def same_category(category_name, data):
    return data.loc[data['category'] == category_name, 'question']

print('Q3: Same category\n', same_category(personalized[0]['category'], df).to_string(index=False))
new_keyword = input('Enter a keyword to add [invoice]: ').strip() or 'invoice'
df.at[0, 'keywords'] = df.at[0, 'keywords'] + ' ' + new_keyword
output = Path(__file__).resolve().parent / f'{roll}_faq_data.csv'
df.to_csv(output, index=False)
print('Q4: Saved', output.name)
print('Q5: Counts per category\n', df.groupby('category').size())

def top_matches(query, data):
    ranked = score(query, data)
    if ranked.empty:
        print('No FAQ matched:', query)
        return
    highest = ranked['confidence'].max()
    winners = ranked[ranked['confidence'] == highest]
    print('Query:', query, '\nTop matching entries:\n', winners.to_string(index=False))

print('Q6: Tie example')
top_matches('fee', df)
print('Q6: No-tie example')
top_matches('password reset', df)

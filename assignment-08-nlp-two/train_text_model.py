import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

text = ('Cloud projects help students understand computing and automation. '
        'Students develop applications and deploy them using cloud services. '
        'Cloud services provide storage networking and flexible computing power. '
        'Developers learn to configure servers and secure their applications. '
        'Simple projects teach important concepts through practical experiments. '
        'Students explore cloud infrastructure and improve software reliability. '
        'They monitor resources and evaluate application performance. '
        'Reliable applications support users and handle growing demand. '
        'Learning modern technology requires patience and regular practice. '
        'Every project brings useful experience for future development. '
        'Good developers test their code and improve their designs. '
        'Cloud computing supports experimentation collaboration and learning. '
        'New tools help students build and share useful applications. '
        'Practice builds confidence through repeated experiments. '
        'Students improve their technical skills by solving real problems.')
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts([text])
tokens = tokenizer.texts_to_sequences([text])[0]
sequences = [tokens[:i + 1] for i in range(1, len(tokens))]
max_length = max(map(len, sequences))
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
X, y = sequences[:, :-1], sequences[:, -1]
vocab_size = len(tokenizer.word_index) + 1
model = Sequential([Embedding(vocab_size, 32, input_length=max_length - 1), LSTM(64), Dense(vocab_size, activation='softmax')])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=60, batch_size=8, verbose=0)
reverse = {index: word for word, index in tokenizer.word_index.items()}
def generate(seed, count=12):
    result = seed
    for _ in range(count):
        encoded = tokenizer.texts_to_sequences([result])[0]
        padded = pad_sequences([encoded], maxlen=max_length - 1, padding='pre')
        prediction = int(np.argmax(model.predict(padded, verbose=0)[0]))
        result += ' ' + reverse.get(prediction, 'cloud')
    return result
for seed in ['students', 'cloud', 'developers']:
    print(generate(seed))

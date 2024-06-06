import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://en.wikisource.org/wiki/The_Verdict"
page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")
# soup.prettify()
text = [i.text for i in soup.find_all("p")]
text = text[0:83]
# with open('Data.txt', 'w') as file:
#     for string in text:
#         file.write(string + '\n')

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(oov_token='<nothing>')
tokenizer.fit_on_texts(text)

input_sequences = []

for sentences in text:
  tokenized_sen = tokenizer.texts_to_sequences([sentences])[0]
  for i in range(1,len(tokenized_sen)):
    input_sequences.append(tokenized_sen[:i+1])

max_len = max(len(x) for x in input_sequences)

from keras.preprocessing.sequence import pad_sequences
padded_input_sequences = pad_sequences(input_sequences, maxlen = max_len, padding='pre')

X = padded_input_sequences[:,:max_len-1]
y = padded_input_sequences[:,-1:]

from tensorflow.keras.utils import to_categorical #OHE
y = to_categorical(y, num_classes = 1100) # vocal size + 1

from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential

model = Sequential()
model.add(Embedding(1100, 100, input_length = 230))
model.add(LSTM(200))
model.add(Dense(1100, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()

model.fit(X, y, epochs = 100)

def prediction(t,l):
  text = t
  sentence_length = l
  for repeat in range(sentence_length):
    token_text = tokenizer.texts_to_sequences([text])
    padded_token_text = pad_sequences(token_text, maxlen = 230, padding = 'pre')
    pos = np.argmax(model.predict(padded_token_text))
    for (word,index) in tokenizer.word_index.items():
      if index == pos:
        text = text + " " + word
  return text

import gradio as gr

demo = gr.Interface(title = "The Verdict",
                    examples = [['It had always been'], ['I found the couple at'],['She glanced out almost']],
                    fn=prediction,
                    inputs=[gr.Textbox(lines = 2, label = 'Query', placeholder='Enter Here'),
                            gr.Slider(1,100,step = 1, label = "How many Words to generate?")],
                    outputs=gr.Text(lines = 7, ), allow_flagging = 'never', theme=gr.themes.Base())

demo.launch(share = True)


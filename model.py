# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
from google.colab import drive
drive.mount('/content/drive')


# %%
# !cp -r -v drive/My\ Drive/ThaiLanguageModel/* .
get_ipython().system('rsync -a --progress --exclude="dataset" drive/My\\ Drive/ThaiLanguageModel/* .')


# %%
get_ipython().system('rm -rf dataset')
get_ipython().system('mkdir dataset')
get_ipython().system('mkdir dataset/thwiki-words')
get_ipython().system('cp drive/My\\ Drive/ThaiLanguageModel/dataset/thwiki-words.zip dataset/thwiki-words.zip')
get_ipython().system('unzip dataset/thwiki-words.zip -d dataset')


# %%
get_ipython().system('pip install ujson')
get_ipython().system('pip install pythainlp')


# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%
from keras.layers import CuDNNLSTM, Bidirectional, RepeatVector, Dense, TimeDistributed, Activation, Input
from keras.models import Sequential
from keras.optimizers import Adam
from pathlib import Path
import data_gen


# %%
NUM_VOCAB = data_gen.NUM_VOCAB
length = 32
batch_size = 32
train_gen = data_gen.sentence_generator(Path('dataset/thwiki-words'),
                                        length,
                                        batch_size,
                                        adversarial=data_gen.random_blank(0.05))


# %%
model = Sequential()
activation = 'relu'
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True), input_shape=(length, NUM_VOCAB)))
model.add(Activation(activation))
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
model.add(Activation(activation))
model.add(Bidirectional(CuDNNLSTM(32, return_sequences=True)))
model.add(Activation(activation))
model.add(Bidirectional(CuDNNLSTM(32, return_sequences=True)))
model.add(Activation(activation))


# model.add(RepeatVector(length))


model.add(Bidirectional(CuDNNLSTM(32, return_sequences=True)))
model.add(Activation(activation))
model.add(Bidirectional(CuDNNLSTM(32, return_sequences=True)))
model.add(Activation(activation))
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
model.add(Activation(activation))
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
model.add(Activation(activation))
model.add(TimeDistributed(Dense(NUM_VOCAB)))
model.add(Activation('softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()


# %%
model.fit_generator(train_gen,
                    steps_per_epoch=4096,
                    initial_epoch=41,
                    epochs=100)


# %%
model.save('weight/test.h5')



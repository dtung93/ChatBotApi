import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load and preprocess the dataset
with open("movie_characters_metadata.txt") as f:
    conversation = f.read().splitlines()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(conversation)
conversation = tokenizer.texts_to_sequences(conversation)
conversation = pad_sequences(conversation, padding='post')

# Define the generator
latent_dim = 50
generator_inputs = Input(shape=(latent_dim,))
x = Dense(256, activation='relu')(generator_inputs)
x = Dense(len(tokenizer.word_index)+1, activation='softmax')(x)
generator = Model(generator_inputs, x)

# Define the discriminator
max_len = conversation.shape[1]
discriminator_inputs = Input(shape=(max_len,))
x = Embedding(len(tokenizer.word_index)+1, 50)(discriminator_inputs)
x = LSTM(256)(x)
x = Dense(1, activation='sigmoid')(x)
discriminator = Model(discriminator_inputs, x)

# Define the combined model
discriminator.trainable = False
combined_inputs = Input(shape=(latent_dim,))
combined_outputs = discriminator(generator(combined_inputs))
combined = Model(combined_inputs, combined_outputs)

# Compile the models
optimizer = Adam(lr=0.0002, beta_1=0.5)
generator.compile(loss='categorical_crossentropy', optimizer=optimizer)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# Train the models
num_epochs = 50
batch_size = 128

for epoch in range(num_epochs):
    for i in range(conversation.shape[0] // batch_size):
        # Train the discriminator
        real_samples = conversation[i*batch_size:(i+1)*batch_size]
        real_labels = np.ones((batch_size, 1))
        fake_samples = generator.predict(np.random.normal(0, 1, (batch_size, latent_dim)))
        fake_labels = np.zeros((batch_size, 1))
        discriminator.trainable = True
        discriminator.train_on_batch(real_samples, real_labels)
        discriminator.train_on_batch(fake_samples, fake_labels)
        discriminator.trainable = False

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generator_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print the epoch and loss
    print("Epoch: ", epoch, ", Generator Loss: ", generator_loss)

# Generate responses
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])[0]
    input_seq = pad_sequences([input_seq], maxlen=max_len, padding='post')
    latent_vector = np.random.normal(0, 1, (1, latent_dim))
    generated_seq = np.argmax(generator.predict(latent_vector), axis=-1)
    response = []
    for idx in generated_seq:
        if idx == 0:
            continue
        word = tokenizer.index_word[idx]
        if word == "<EOS>":
            break
        response.append(word)
    return " ".join(response)

# Test the chatbot
while True:
    input_text = input("You: ")
    response = generate_response(input_text)
    print("Bot: ", response)
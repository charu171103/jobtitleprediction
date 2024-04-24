import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns                                                                                                                                                                                  # Replace 'file_path' with the actual path to your dataset
file_path = '/kaggle/input/data-science-job-posting-on-glassdoor/Cleaned_DS_Jobs.csv'
data = pd.read_csv(file_path)   
print(data.columns)  # This will display all the column names in your dataset
print(data['Job Description'])# Display the first few rows of the dataset to understand its structure
print(data.head())

# Get information about the columns, data types, and missing values
print(data.info())

# Summary statistics of numerical columns
print(data.describe())

# Unique values in categorical columns
print(data['Job Title'].unique())
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Select relevant columns
selected_columns = ['Job Title', 'Company Name', 'Industry']
data_selected = data[selected_columns]

# Drop rows with missing values
data_selected.dropna(inplace=True)

# Concatenate text columns to create input text
data_selected['Text'] = data_selected['Job Title'] + ' ' + data_selected['Company Name'] + ' ' + data_selected['Industry']

# Preprocessing text data
max_words = 1000  # Maximum number of words to tokenize
max_sequence_length = 50  # Maximum length of the sequences

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(data_selected['Text'])

sequences = tokenizer.texts_to_sequences(data_selected['Text'])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Create feature and target sets (for example, predicting 'Industry' based on text inputs)
X = padded_sequences
y = pd.get_dummies(data_selected['Industry'])  # Assuming 'Industry' is the target

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the RNN model
model = Sequential()
model.add(Embedding(max_words, 64, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dense(y.shape[1], activation='softmax'))  # Output layer with number of classes (industries)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
# Assuming 'model' is your trained RNN model and 'tokenizer' is your text tokenizer

# Your job description for which you want to predict the job title
new_job_description = """The Video & Image Understanding Group develops and applies cutting-edge computer vision and deep learning techniques to solve problems in object detection, recognition, localization & tracking; biometrics; 3D scene reconstruction; threat detection; vision-aided navigation; and situational awareness. Our projects integrate algorithmic research, software engineering, and application system development to produce innovative capabilities that address challenging national security problems. We actively collaborate with research engineers across STR, as well as with academic researchers and industry partners to develop practical and powerful computer vision solutions.

The Role:
Develop, implement and evaluate machine learning algorithms to detect, segment, classify and track objects in diverse imaging modalities.
Apply machine learning techniques to integrate information from multiple looks and multiple modalities to improve scene understanding capabilities.
Develop and apply transfer learning and domain adaptation techniques to solve scene understanding problems where limited measured training data is available.
Perform data analysis on experimental data and identify performance improvement strategies.
Transition machine learning algorithms to prototype and/or real-time systems demonstrations.
Who You Are:
BS, MS or PhD degree in Computer Science, Electrical Engineering, Applied Mathematics or related technical discipline
Strong scientific software development skills
Independent analytical problem-solving skills
Experience in machine learning techniques, particularly deep learning
Motivated collaborator and excellent communicator to both technical and non-technical audiences
Some Other Relevant Skills You May Have:
Experience with Python and C/C++
Experience in developing and applying machine learning techniques to solving computer vision problems"""

# Tokenize and pad sequences for the new job description
new_sequence = tokenizer.texts_to_sequences([new_job_description])
padded_new_sequence = pad_sequences(new_sequence, maxlen=max_sequence_length)

# Make prediction using the trained model
prediction = model.predict(padded_new_sequence)

# Convert prediction to a human-readable label
predicted_label_index = np.argmax(prediction)
predicted_job_title = y.columns[predicted_label_index]  # Assuming 'y' is your one-hot encoded target variable

# Display the predicted job title
print("Predicted Job Title:", predicted_job_title)

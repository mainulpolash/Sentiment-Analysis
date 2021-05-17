import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelBinarizer


##Reading the data
data_imdb = read_csv('../input/imdb_dataset.csv')
#print (data_imdb.shape)

##Split the data in to train and test set
 train_x = data_imdb.review[:20000]
 train_y = data_imdb.sentiment[:20000]

 test_x = data_imdb.review [20000:30000]
 test_y = data_imdb.sentiment[20000:30000]

 ## Now I will define the pre-processing functions 

#Removing html strips    
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#remove special characters     

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

 ##stopword
stop=set(stopwords.words('english'))
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

 # Stemming   
def stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

#Now I will apply these pre-processing in the review column of my data    
data_imdb['review'] = data_imdb['review'].apply(strip_html)
data_imdb['review'] = data_imdb['review'].apply(remove_special_characters)
data_imdb['review'] = data_imdb['review'].apply(remove_stopwords)
data_imdb['review'] = data_imdb['review'].apply(stemmer)

# Convert the labels with labelbinarizer
data_imdb['sentiment'] = LabelBinarizer.fit_transform(data_imdb['sentiment'])


#After Processing the text I need to build my model after loading the universal sentence encoder

# Load the pretrained Embedding vector of size 512
model = "universal-sentence-encoder"
version = 4 
embedding = "https://tfhub.dev/google/"+model+"/"+str(version)

#define the hub layer 
hub_layer2 = hub.KerasLayer(embedding, output_shape=[512], input_shape=[],dtype=tf.string)

#Defining my model using keras 

model = keras.Sequential()
#first layer ->>Input layer
model.add(hub_layer2)
#2nd layer->hidden layer
model.add(keras.layers.Dense(64, activation='relu'))
#3rd layer ->> output layer
model.add(keras.layers.Dense(1, activation='softmax'))

#Now I will define loss function and optimizer and cofigure the model 
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

# Training the model for 10 epochs in mini-batches of 512 samples. I also set the callback here

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]


history = model.fit(train_x, train_y, epochs =10, validation_split = 0.25, verbose =1, batch_size = 512, callbacks=callbacks)

#Evalutation of the model: 
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=1)
print('Test_loss:', test_loss)
print('Test_accuracy:', test_acc)
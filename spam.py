import sys
import nltk
import sklearn
import pandas
import numpy

# for checking the versions
print('Python: {}'.format(sys.version))
print('NLTK: {}'.format(nltk.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('pandas: {}'.format(pandas.__version__))
print('numpy: {}'.format(numpy.__version__))

#1 load the dataset 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_table('SMSSpamCollection', header = None, encoding='utf-8')
print(df.info())
print(df.head())

classes = df[0]
print(classes.value_counts())

# 2  preprocess the data 0 ham and 1 spam (Binary Classification)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
Y = encoder.fit_transform(classes)
print(classes[:10])
print(Y[:10])

text_messages = df[1]
print(text_messages[:10])

# Use regular expression to replace email addresses , urls, phonenumber, other phone number, symbols
# email
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddr')

# web address
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)7$', 'webaddress')

# moneysymb
processed = processed.str.replace(r'£|\$', 'moneysymb')

# phonenumbr
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumbr') 

# number
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr') 

#remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ') 

#remove white space
processed = processed.str.replace(r'\s+', ' ')  

#leading and trailing white space
processed = processed.str.replace(r'^\s+|\s+?$', '')  

#chenging the words to lower case
processed = processed.str.lower()
print(processed)

#remove stop words from text
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

#remove stem from text
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

print(processed)


#number of words and most common words and how many times they have appeared in the text
from nltk.tokenize import word_tokenize

all_words = []

for message in processed:
	words = word_tokenize(message)
	for w in words:
		all_words.append(w)

all_words = nltk.FreqDist(all_words)	

print('number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(15)))

#use 1500 most comman words as features
word_features = list(all_words.keys())[:1500]

def find_features(message):
	words = word_tokenize(message)
	features = {}
	for word in word_features:
		features[word] = (word in words)

	return features

features = find_features(processed[0])
for key, value in features.items():
    if value == True:
        print(key)	



#find features for all messages
messages = list(zip(processed, Y))

#define a seed for reproductivity
seed = 1
np.random.seed = seed
np.random.shuffle(messages)

#call find functions for each messages
featuresets = [(find_features(text), label) for (text, label) in messages]

from sklearn import model_selection
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state = seed)

print('training: {}'.format(len(training)))
print('testing: {}'.format(len(testing)))


#scikit-learn classifier with nltk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#Define models to train
names = ['K Nearest neighbors', 'Decision Tree', 'Random Forest', 'Logistic Regression',  'SGD classifier', 'Naive Bayes', 'SVM Linear']

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names, classifiers))

from nltk.classify.scikitlearn import SklearnClassifier

for name, model in models:
	nltk_model = SklearnClassifier(model)
	nltk_model.train(training)
	accuracy = nltk.classify.accuracy(nltk_model, testing) * 100
	print('{}: Accuracy: {}'.format(name, accuracy))



from sklearn.ensemble import VotingClassifier

names = ['K Nearest neighbors', 'Decision Tree', 'Random Forest', 'Logistic Regression',  'SGD classifier', 'Naive Bayes', 'SVM Linear']    

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names, classifiers))

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_ensemble, testing) * 100

print('Ensemble Method Accuracy: {}'.format(accuracy))
#wrap models in NLTK


txt_features, labels = zip(*testing)

prediction = nltk_ensemble.classify_many(txt_features)
# print a confusion matrix and a classification report
print(classification_report(labels, prediction))

pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'], ['ham', 'spam']])

names = ['KNN', 'DT','RF','LR','SGD','NB','SVM']
acc =   [94.40057430007178,97.34386216798278,98.56424982053123,98.56424982053123,98.27709978463747,98.49246231155779,98.49246231155779]
plt.figure(figsize=(8,6))
plt.subplot()
plt.bar(names, acc, width=0.8)
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.suptitle('Accuracy of Models')
plt.show()    


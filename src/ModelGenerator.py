import pandas as pd
import nltk
import pickle
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dados de treinamento (exemplo)
dataset = pd.read_csv("../dataset/netflix_reviews.csv", encoding='utf-8')
dataset = dataset.drop(['reviewId', 'userName', 'thumbsUpCount', 'reviewCreatedVersion', 'at', 'appVersion'], axis=1)
dataset['score'] = pd.to_numeric(dataset['score'], errors='coerce')
dataset = dataset[~((dataset['score'] > 1) & (dataset['score'] < 5))]

X = dataset['content']
y = dataset['score']

vetorizador = CountVectorizer()
X = vetorizador.fit_transform(X)

#Dividindo os dados em treino e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = MultinomialNB()
modelo.fit(X_treinamento, y_treinamento)

previsoes = modelo.predict(X_teste)
precisao = accuracy_score(y_teste, previsoes)
print(f"Precisão do modelo: {precisao:.2f}")

finalizedModel = 'finalized_model.sav'
pickle.dump(modelo, open(finalizedModel, 'wb'))

finalizedVector = 'finalized_vector.sav'
pickle.dump(vetorizador, open(finalizedVector, 'wb'))


from sklearn.feature_extraction.text import CountVectorizer

class Evaluator:

    def __init__(self):
        self.modelo = None
        self.vetor = None
        self.sentence = None
    def evaluate_sentence(self, sentence, modelo, vetor):
        self.modelo = modelo
        self.vetor = vetor
        self.sentence = sentence
        nova_frase_transformada = self.vetor.transform(self.sentence)
        previsao_nova_frase = self.modelo.predict(nova_frase_transformada)

        if previsao_nova_frase[0] == 5:
            return "The sentence is positive."
        else:
            return "The sentence is negative."


    # nova_frase = ["This film is the best film I have ever seen. I recommend it to everyone. Excellent!"]
    # nova_frase2 = ["This movie is the worst movie I have ever seen. I don't recommend it to anyone. Terrible!"]
    #
    # evaluate_sentence(nova_frase)
    # evaluate_sentence(nova_frase2)
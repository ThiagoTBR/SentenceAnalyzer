import tkinter as tk
import pickle

from Evaluator import Evaluator
from tkinter import messagebox


finalizedModel = 'finalized_model.sav'
finalizedVector = 'finalized_vector.sav'

# Carregar o modelo
with open(finalizedModel, 'rb') as arquivo:
    modelo = pickle.load(arquivo)

# Carregar o vetorizador
with open(finalizedVector, 'rb') as arquivo:
    loadedVector: object = pickle.load(arquivo)

# Função que será chamada quando o botão for pressionado
def submit_text():
    evaluator = Evaluator()
    text = text_box.get("1.0", "end-1c")  # Obtém o texto da caixa de texto
    if not text:
        messagebox.showerror("Error", "Please enter a text to be evaluated.")
    else:
        text = evaluator.evaluate_sentence([text], modelo, loadedVector)  # Avalia o texto
        print(text)  # Imprime o texto para fins de demonstração
        messagebox.showinfo("Assessment Result", text)  # Exibe um popup com o texto

    text_box.delete("1.0", "end")  # Limpa a caixa de texto

# Cria a janela principal
root = tk.Tk()
root.title("Text Sentiment Analysis")
# Cria a caixa de texto
text_box = tk.Text(root, width=40, height=10)
text_box.pack()

# Cria o botão
submit_button = tk.Button(root, text="Submit", command=submit_text)
submit_button.pack()

# Inicia o loop principal
root.mainloop()
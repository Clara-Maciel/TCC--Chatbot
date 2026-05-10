from chat_engine import responder # Substitua 'seu_arquivo' pelo nome do arquivo onde está o código

pergunta = "Como funciona o auxílio transporte?"
resposta = responder(pergunta)
print("\n" + "="*30)
print(f"Pergunta: {pergunta}")
print(f"Resposta: {resposta}")
print("="*30 + "\n")

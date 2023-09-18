from numpy.lib.function_base import delete
import os
import json
from telegram import Update
from telegram.ext import CallbackContext, Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


# Definição dos estados da conversa
MENU, TREINAMENTO, ESCOLHER_LABEL, APOIO_DIAGNOSTICO, PERGUNTAR_CARACTERISTICA = range(5)
SOLICITAR_SENHA = range(2)
AGUARDANDO_SENHA = 1
menu_inicial_exibido = False

# Arquivos para armazenar dados e estatísticas
data_file = "data.json"

def carregar_dados():
    with open(data_file, 'r') as file:
        data = json.load(file)
        transtornos = []
        caracteristicas = []

        for entry in data['data']:
            label = entry['label']
            features = entry['features']

            # Aqui, verificamos se o transtorno já está na lista
            # Se não estiver, adicionamos o transtorno e suas características
            if label not in transtornos:
                transtornos.append(label)
                caracteristicas.append(features)
            else:
                # Se já estiver na lista, adicionamos apenas as características
                # às características existentes para o mesmo transtorno
                index = transtornos.index(label)
                caracteristicas[index].extend(features)

        return caracteristicas, transtornos

# Salvar dados
def salvar_dados(data_to_save):
    with open(data_file, 'w') as file:
        json.dump(data_to_save, file)

def adicionar_dados_de_treinamento(dados_treinamento):
    # Carregar dados existentes do arquivo JSON
    dados_existentes = json.load(open(data_file, 'r'))
    # Adicionar os novos dados de treinamento
    dados_existentes['data'].append(dados_treinamento)
    # Salvar os dados atualizados de volta no arquivo JSON
    salvar_dados(dados_existentes, data_file)
    return "Dados de treinamento adicionados com sucesso!"

    # Adicionar os novos dados de treinamento
    dados_existentes['data'].append(dados_treinamento)

    # Salvar os dados atualizados de volta no arquivo JSON
    with open(data_file, 'w') as file:
        json.dump(dados_existentes, file, indent=4)

    return "Dados de treinamento adicionados com sucesso!"

caracteristicas, transtornos = carregar_dados()
caracteristicas_perguntadas = []
caracteristicas_para_perguntar = [item for sublist in caracteristicas for item in sublist]

# Dividir seus dados em conjuntos de treinamento e validação
X_train, X_test, y_train, y_test = train_test_split(caracteristicas, transtornos, test_size=0.2)

# Transformar características em vetores (one-hot encoding)
vectorizer = CountVectorizer(binary=True)
X_train = [" ".join(c) for c in X_train]  # Transforme as características em strings
X = vectorizer.fit_transform(X_train).toarray()

# Transformar transtornos em números
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)
y = tf.keras.utils.to_categorical(y_encoded)

# Modelo de rede neural para diagnóstico
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10)  # Ajuste o número de épocas conforme necessário

def aguardar_senha(update: Update, context: CallbackContext):
    senha_digitada = update.message.text
    senha_correta = "12125"  # Substitua pela senha correta

    if senha_digitada == senha_correta:
        update.message.reply_text("Senha correta! Você agora pode acessar o treinamento.")
        return treinamento(update, context)  # Vá para o estado TREINAMENTO após verificar a senha correta
    else:
        update.message.reply_text("Ops. Houve um probleminha! A senha está incorreta. Tente novamente.")
        return iniciar(update, context)  # Volte para o menu principal

def salvar_caracteristicas(context, caracteristicas):
    if 'caracteristicas_salvas' not in context.user_data:
        context.user_data['caracteristicas_salvas'] = []
    context.user_data['caracteristicas_salvas'].extend(caracteristicas)

# Função para lidar com mensagens não reconhecidas durante a conversação
def nao_reconhecido(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Desculpe, não reconheço essa opção. Retornando ao menu principal.")
    return iniciar(update, context)

# Função para iniciar a conversa
def iniciar(update: Update, context: CallbackContext) -> int:
    global menu_inicial_exibido
    # Se o menu inicial ainda não foi exibido, cumprimente o usuário
    if not menu_inicial_exibido:
        update.message.reply_text("Olá! Bem-vindo ao sigBot. Eu aprendo e ajudo profissionais de saúde mental no processo de diagnóstico de transtornos de personalidade. Como posso ajudar você? Digite o número correspondente!")
        update.message.reply_text("Eu não sou humano. Sou uma máquina treinada com inteligência artificial. Não interprete as minha respostas de forma isolada. Sempre busque um profissional para avaliar minha sugestão de diagnóstico! Sou inteligente, mas posso errar!")
        update.message.reply_text("Desenvolvido por Nicolas Fischer - UNISC")
        menu_inicial_exibido = True

    update.message.reply_text("Selecione uma opção: \n1. Treinar \n2. Ajuda no diagnóstico \n3. Ver estatísticas \n4. Encerrar")
    return MENU

def solicitar_senha(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Por favor, insira a senha para acessar o treinamento:")
    return AGUARDANDO_SENHA

def verificar_senha(update: Update, context: CallbackContext) -> int:
    senha_digitada = update.message.text
    senha_correta = "12125"

    if senha_digitada == senha_correta:
        return treinamento(update, context)
    else:
        update.message.reply_text("Senha incorreta. Retornando ao menu principal.")
        return iniciar(update, context)

def treinamento(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Por favor, informe as características (sintomas) separadas por vírgula. Por exemplo: 'irritabilidade, instabilidade emocional...'")
    return TREINAMENTO

# Função para coletar características e rótulo associado
def coletar_dados_treinamento(update: Update, context: CallbackContext) -> int:
    caracteristicas = update.message.text.split(',')
    caracteristicas = [c.strip() for c in caracteristicas]
    context.user_data['caracteristicas'] = caracteristicas

    # Construir o texto com a lista numerada de transtornos
    transtornos_text = "\n".join([f"{i + 1}. {t}" for i, t in enumerate(transtornos)])
    update.message.reply_text(f'Qual é o transtorno associado a essas características?\n\n{transtornos_text}')

    return ESCOLHER_LABEL

# Função para processar a escolha do usuário
def processar_escolha_rotulo(update: Update, context: CallbackContext) -> int:
    escolha_usuario = update.message.text

    try:
        escolha_usuario = int(escolha_usuario)
        if 1 <= escolha_usuario <= len(transtornos):
            rotulo_selecionado = transtornos[escolha_usuario - 1]

            # Carregue os dados existentes do arquivo JSON
            with open(data_file, 'r') as file:
                dados_existentes = json.load(file)

            # Adicione os novos dados de treinamento
            dados_existentes['data'].append({
                'features': context.user_data['caracteristicas'],
                'label': rotulo_selecionado
            })

            # Salve os dados atualizados de volta no arquivo JSON
            with open(data_file, 'w') as file:
                json.dump(dados_existentes, file, indent=4)

            update.message.reply_text(f'Informações salvas para o transtorno: {rotulo_selecionado}. Vamos voltar ao menu inicial!')

        else:
            update.message.reply_text('Por favor, escolha um número válido da lista.')

    except ValueError:
        update.message.reply_text('Por favor, escolha um número válido da lista.')

    return iniciar(update, context)

# Função para encerrar a conversa
def encerrar_conversa(update: Update, context: CallbackContext) -> int:
    update.message.reply_text('Obrigado por conversar comigo. Para iniciar um diálogo digite "/start"! Até mais!')
    return ConversationHandler.END

def apoio_diagnostico(update: Update, context: CallbackContext) -> int:
    # Comece perguntando ao usuário sobre a primeira característica da lista
    update.message.reply_text(f"O paciente apresenta a seguinte característica: {caracteristicas_para_perguntar[0]}?\n\n1. Sim\n2. Não")
    return PERGUNTAR_CARACTERISTICA

def processar_resposta_caracteristica(update: Update, context: CallbackContext) -> int:
    resposta = update.message.text

    # Se a resposta for "Sim", adicionamos a característica à lista de características do usuário
    if resposta == "1":
        if 'caracteristicas_paciente' not in context.user_data:
            context.user_data['caracteristicas_paciente'] = []
        context.user_data['caracteristicas_paciente'].append(caracteristicas_para_perguntar[0])

    # Adicione a característica à lista de características já perguntadas
    caracteristicas_perguntadas.append(caracteristicas_para_perguntar[0])

    # Use a rede neural para determinar a próxima característica mais relevante para perguntar
    caracteristicas_string = [" ".join(context.user_data.get('caracteristicas_paciente', []))]
    X_pred = vectorizer.transform(caracteristicas_string).toarray()
    predicao = model.predict(X_pred)

    # Determine o transtorno mais provável com base na predição
    transtorno_predito = label_encoder.inverse_transform([np.argmax(predicao)])[0]

    # Se a confiança na predição for alta o suficiente, forneça o diagnóstico
    confianca = np.max(predicao)
    if confianca > 0.75:
        update.message.reply_text(f"Com base nas informações fornecidas, o possível diagnóstico é: {transtorno_predito}.")
        return iniciar(update, context)

    # Caso contrário, continue perguntando
    index = transtornos.index(transtorno_predito)
    possiveis_caracteristicas = [c for c in caracteristicas[index] if c not in caracteristicas_perguntadas]

    # Embaralhar as características possíveis para introduzir variedade nas perguntas
    random.shuffle(possiveis_caracteristicas)

    # Se ainda houver características relevantes, pergunte sobre a primeira delas
    if possiveis_caracteristicas:
        update.message.reply_text(f"O paciente apresenta a seguinte característica: {possiveis_caracteristicas[0]}?\n\n1. Sim\n2. Não")
        caracteristicas_para_perguntar[0] = possiveis_caracteristicas[0]
        return PERGUNTAR_CARACTERISTICA
    else:
        update.message.reply_text(f"Com base nas informações fornecidas, o possível diagnóstico é: {transtorno_predito}.")
        return iniciar(update, context)


def estatisticas(update: Update, context: CallbackContext) -> int:
    # Carregar os dados
    caracteristicas, transtornos = carregar_dados()

    # Criar um dicionário para agrupar características por transtorno
    dados_por_transtorno = {}
    for i, transtorno in enumerate(transtornos):
        if transtorno not in dados_por_transtorno:
            dados_por_transtorno[transtorno] = []
        dados_por_transtorno[transtorno].extend(caracteristicas[i])

    # Calcular o total de transtornos de personalidade conhecidos
    total_transtornos = len(set(transtornos))

    # Calcular o total de características conhecidas
    total_caracteristicas = sum(len(caracteristicas) for caracteristicas in dados_por_transtorno.values())

    # Calcular a possibilidade de diagnósticos que podem ser apoiados
    total_diagnosticos = len(caracteristicas)

    # Calcular o valor médio de características por transtorno de personalidade
    media_caracteristicas_por_transtorno = total_caracteristicas / total_transtornos

    # Criando o gráfico de barras
    labels = ['Transtornos Conhecidos', 'Características Conhecidas', 'Diagnósticos Apoiados', 'Média Caract./Transtorno']
    values = [total_transtornos, total_caracteristicas, total_diagnosticos, media_caracteristicas_por_transtorno]

    plt.bar(labels, values)
    plt.title("Estatísticas de Transtornos de Personalidade")
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Salvar gráfico em imagem e enviar ao usuário
    plt.savefig('estatisticas.png', bbox_inches='tight')
    update.message.reply_photo(open('estatisticas.png', 'rb'))

    # Construir mensagem com os dados estatísticos
    mensagem = f"Estatísticas:\n\n"
    mensagem += f"Total de Transtornos de Personalidade Conhecidos: {total_transtornos}\n"
    mensagem += f"Total de Características Conhecidas: {total_caracteristicas}\n"
    mensagem += f"Possibilidade de Diagnósticos que Podem Ser Apoiados: {total_diagnosticos}\n"
    mensagem += f"Valor Médio de Características por Transtorno de Personalidade: {media_caracteristicas_por_transtorno:.2f}\n"

    update.message.reply_text(mensagem)

    # Remover o arquivo da imagem
    os.remove('estatisticas.png')

    # Retornar ao menu inicial
    return iniciar(update, context)

# Função principal
def main():
    updater = Updater("6101753675:AAFXx22QKb83uFfpm5E7lMlafb-8tNCgBhQ")  # Substitua pelo token do seu bot
    dispatcher = updater.dispatcher

    conversa_handler = ConversationHandler(
        entry_points=[CommandHandler('start', iniciar)],
        states={
            MENU: [
                MessageHandler(Filters.regex('^1(\s*|\.)?$'), solicitar_senha),
                MessageHandler(Filters.regex('^2(\s*|\.)?$'), apoio_diagnostico),
                MessageHandler(Filters.regex('^3(\s*|\.)?$'), estatisticas),
                MessageHandler(Filters.regex('^4(\s*|\.)?$'), encerrar_conversa)
            ],
            TREINAMENTO: [MessageHandler(Filters.text & ~Filters.command, coletar_dados_treinamento)],
            ESCOLHER_LABEL: [MessageHandler(Filters.text & ~Filters.command, processar_escolha_rotulo)],
            AGUARDANDO_SENHA: [MessageHandler(Filters.text & ~Filters.command, verificar_senha)],
            PERGUNTAR_CARACTERISTICA: [MessageHandler(Filters.regex('^(1|2)$'), processar_resposta_caracteristica)]
        },
        fallbacks=[MessageHandler(Filters.text, nao_reconhecido)]
    )

    dispatcher.add_handler(conversa_handler)  # Adicionado manipulador de conversação

    # Inicie o bot
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

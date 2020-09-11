import os
import re
import pandas as pd
import unicodedata
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import RSLPStemmer
from unicodedata import normalize





# Load Data Functions
def get_categories(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]


def array_to_pandas(array, column_names):
    """
    create dataset in the format
    :param array: panda dataset
    :param column_names:colunas
    :return: array with three columns
    """
    return pd.DataFrame(array, columns=column_names)


def format_dataset(path, pandas_dataframe=True):
    """
    create dataset in the format
    :param path: caminho da pasta raiz contendo as pastas de cada classe com os respectivos resumos
    :param pandas_dataframe: dataframe
    :return: array with three columns
    """
    categories = get_categories(path)
    dataset = []
    for category in categories:
        for abstract in os.listdir(path + category):
            if not abstract.startswith("."):
                text = ""
                row = []
                with open("{}/{}".format(path + category, abstract)) as f:
                    for line in f:
                        text += line
                row.append(abstract)  # file_name
                row.append(category)  # category
                row.append(text)
                dataset.append(row)

    if pandas_dataframe:
        column_names = ['id', 'texto', 'classe']
        dataset = array_to_pandas(dataset, column_names)

    return dataset


## Preprocessing functions
def get_stopwords():
    commom_words = ['tambem', 'sido', 'todas', 'todos', 'assim', 'alguns', 'alem', 'ainda', 'duas', 'desses', 'deste',
                    'nao', 'um', 'dois', 'tres', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove', 'dez', 'onze',
                    'doze',
                    'treze', 'quartoze', 'quinze', 'dezesseis', 'dezesete', 'dezoito', 'dezenove', 'vinte', 'trinta',
                    'quarenta', 'cinquente', 'sessenta', 'setenta', 'oitenta', 'noventa', 'cem', 'sao', 'pode', 'podem',
                    'resultados', 'pacientes', 'objetivo', 'metodo',
                    'conclusoes', 'conclusao', 'dados', 'et', 'al']

    stopword = set(stopwords.words('portuguese') + list(punctuation) + commom_words)
    return stopword


def remove_stopwords(text):
    stopword = get_stopwords()
    text = [word for word in text.split() if word not in stopword]
    return ' '.join(text)


def stemmer(text):
    stemmerParam = RSLPStemmer()
    new_text = []
    for word in text.split():
        new_text.append(stemmerParam.stem(word.lower()))
    return ' '.join(new_text)


def remove_special_character1(text):
    text = normalize("NFKD", text).encode("utf-8", "ignore").decode("utf-8")
    return text


def remove_special_character2(text):
    text = normalize("NFKD", text).encode("utf-8", "ignore").decode("utf-8")
    text = u"".join([c for c in text if not unicodedata.combining(c)])
    #text = u"".join([c for c in text if len(c) >= 3])

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [palavra for palavra in tokens if len(palavra) >= 3]
    text = ' '.join(tokens)

    # Remove espaços extras e palavras menores do que um tamanho mínimo (no exemplo abaixo, 3 caracteres)
    #text = u"".join([palavra for palavra in text if len(palavra) >= 3])

    return re.sub("[^a-zA-Z0-9 \\\]", "", text)


def text_preprocess(category_text_list, category_position="classe", text_position="texto", to_pandas=True, ind_stemmer=1):
    ind_lower = 1
    ind_remove_special_character = 1
    #ind_tokenizer = 1
    ind_remove_stopwords = 1
    #ind_stemmer = 1
    ind_remove_number = 1
    dataset = []
    dfTratado = []
    category = ""

    for index, row in category_text_list.iterrows():
        text = row[text_position]

        if category_position != "":
            category = row[category_position]

        if ind_lower:
            text = text.lower()

        if ind_remove_stopwords:
            text = remove_stopwords(text)

        if ind_stemmer:
            text = stemmer(text)

        if ind_remove_number:
            text = ''.join(i for i in text if not i.isdigit())

        if ind_remove_special_character:
            text = remove_special_character2(text)

        if category_position != "":
            dataset.append([index, text, category])
        else:
            dataset.append([index, text])

    if to_pandas:
        if category_position != "":
            column_names = ["id", text_position, category_position]
        else:
            column_names = ["id", text_position]

        dfTratado = array_to_pandas(dataset, column_names)
    else:
        dfTratado = dataset

    return dfTratado


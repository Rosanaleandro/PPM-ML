import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from tkinter import Y, Button

import seaborn as sns
import matplotlib.pyplot as plt
import os

# Import da base
from sqlalchemy.orm import sessionmaker, synonym
from db.connect_db import database
from db.model import save_to_database, save_to_dtsAtr, Workflow, Dataset, Dataset_Attribute, OperatorsActivity, Experiment, Parameter, Experiment_Attribute, Xai, Xai_Results
from statistics import mode
from sqlalchemy.orm import sessionmaker

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix, precision_score, recall_score, accuracy_score, f1_score 

from utils import (binning, scaling, standardization, onehot_encoder,  ordinal_encoder,
                   over_sampling, under_sampling)

from utils.preprocessing import  convert_type
#def main():
    # -------------------------------- Sidebar -------------------------------
st.set_page_config(
page_title="XMML-PPP",
page_icon="👋",
)

conn_db = database(is_table_log=True)#conexão com a base
pages = st.source_util.get_pages('Principal.py')
new_page_names = {
  'Exploracao_Dados': '1. Exploração de Dados',
  'Preprocessing': '2. Pre-Processamento',
  'ML': '3. Modelo ML',
  'Pesquisa': '4. Consultas'

  }

for key, page in pages.items():
  if page['page_name'] in new_page_names:
    page['page_name'] = new_page_names[page['page_name']]
#st.title("Página principal")

titl_templ = """
        <div style="background-color: royalblue; padding: 15px;border-radius:15px">
        <h2 style="color: mintcream">xMML-PPP Tool - Explainable Machine Learning Tool with XAI Support</h2>
        </div>
        """

st.markdown(titl_templ, unsafe_allow_html=True)
st.sidebar.info("Selecione uma página acima.")

title_dts = """
    <div style="background-color: royalblue; padding: 5px;border-radius:5px; text-align:center">
    <H3 style="color: mintcream">Carregar Dataset</H3>
    </div>
    """

st.sidebar.markdown(title_dts, unsafe_allow_html=True)
# st.sidebar.markdown('## Load dataset')

select_type = st.sidebar.selectbox('Escolha a extensão do arquivo', options=[
                                'Selecione uma opção', 'csv', 'xlsx'])

if select_type == 'csv':
    sep_text_input = st.sidebar.text_input(
            'Insira o separador do arquivo', value=',')

    encoding_text_input = st.sidebar.text_input(
        'Entre com o encode do arquivo selecionado', value='utf8')

file = st.sidebar.file_uploader(label='Carregue seu arquivo CSV ou XLSX', type=['csv', 'xlsx'])
if file:
    file_name = file.name

title_dts = """
            <div style="background-color: royalblue; padding: 5px;border-radius:5px; text-align:center">
            <H3 style="color: mintcream">Carregar arquivo de descrição de atributos</H3>
            </div>
            """
        
st.sidebar.markdown(title_dts, unsafe_allow_html=True)
        #with st.sidebar.header('Upload do arquivo de descrição dos atributos'):
sep_text_input_des = st.sidebar.text_input('Insira o separador do arquivo : ', value=';')
descriptionfile = st.sidebar.file_uploader("Carregue o arquivo de descrição", type=["csv"])

# --------------------Main page content------------------------------------
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
# @st.cache(suppress_st_warning=True)
def read_file_data(file):
    if file is not None:
        if select_type == 'csv':
            df = pd.read_csv(file, sep=sep_text_input,
                            encoding=encoding_text_input)
            return df
        elif select_type == 'xlsx':
            df = pd.read_excel(file)
            return df
    else:
        st.info('Esperando pelo arquivo a ser carregado!')

df = read_file_data(file)

def explore(data):

    df_types = pd.DataFrame({'column': data.columns,
                            'type': data.dtypes,
                            'NA #': data.isna().sum(),
                            'NA %': (data.isna().sum() / data.shape[0]) * 100,
                            'Unique Values': data.nunique()

                            })

    return df_types.astype(str)



if df is None:
    st.markdown('<br>', unsafe_allow_html=True)
    titla_dts = """
            <div style= text-align: center ">
            <H5 style="color: Navy; font-style: italic;text-align: justify; font-family: "Times New Roman"">Esta ferramenta tem a finalidade de contribuir com a realização de tarefas de classificação de ML, capturando as informações da base de 
    dados, das operações de pré-processamento e com os resultados de treino. Também utiliza ferramenta XAI para melhor entendimento do resultado do modelo treinado. 
    As informações capturadas ficam disponíveis para consulta em uma base de dados.</H5>
            </div>
            """
        #st.write("""
    
        #Este APP tem a finalidade de auxiliar na realização de tarefas de classificação de ML, capturar as informações da base de 
    #dados e as operações de pré-processamento, de treino e da ferramenta XAI utilizada pós treino. 
    #As informações são armazenas em base de dados.    
    st.markdown(titla_dts, unsafe_allow_html=True)
    
    st.markdown('<br>', unsafe_allow_html=True)
    titlas_dts = """
            <div style= text-align: center ">
            <H4 style="color: Navy; font-style: italic;text-align: justify; font-family: "Times New Roman"">Conteúdo da Ferramenta: </H4>
            </div>
            <br>
            <div style= text-align: center ">
            <H6 style="color: Black; font-style: italic;text-align: justify; font-family: "Times New Roman"">- Página Exploração de Dados: Permite visualizar os dados e plotar gráficos  do
            dataset carregado.  </H6>
            <div style= text-align: center ">
            <H6 style="color: Black; font-style: italic;text-align: justify; font-family: "Times New Roman""> - Página Pré-Processamento de Dados: Permite pré-processar o dataset para
            treino. </H6>
            <div style= text-align: center ">
            <H6 style="color: Black; font-style: italic;text-align: justify; font-family: "Times New Roman""> - Página ML :  Permite treinar o dataset com o algorimto Random Forest ou SVM e usar ferramentas XAI. </H6>
            <div style= text-align: center ">
            <H6 style="color: Black; font-style: italic;text-align: justify; font-family: "Times New Roman""> - Página  Pesquisa: Permite consultar informações do workflow já finalizado. </H6>
            
            
            """
    st.markdown(titlas_dts, unsafe_allow_html=True)
    

else: 
    if 'dataset' not in st.session_state:
	    st.session_state['dataset'] = df
    
    
    lst_types = []
    lst_types = convert_type(df)
    attributes = {}
    attributes_type = {}
    for i, columns in enumerate(df.columns):
        attributes[columns]= 'atributo original do dataset'
        attributes_type[columns]=lst_types[i]
        # #imprimir para verificar se está correto
    #st.write(attributes)
    #st.write(attributes_type)
    if attributes not in st.session_state:
        st.session_state['attributes'] = attributes
    if attributes_type not in st.session_state:
        st.session_state['attributes_type'] =  attributes_type
    #Coleta de dados para inclusão no banco
    
    #nome do dataset sem extensão
    file_name_list = file_name.split('.')
    name_bd=file_name_list[0]
    #size do dataset
    file_size=file.size
    
    # n_line do dataset
    n_line, n_column = df.shape

    st.markdown('<br>', unsafe_allow_html=True)

    st.markdown('### Informação do Dataset carregado')
    st.markdown('<br>', unsafe_allow_html=True)
    st.write(df.head(5))
    
    st.subheader("Confirma início do trabalho para este dataset?")
    button_conf_sessao= st.button('CONFIRMAR')
    if button_conf_sessao :
        work = Workflow(label=name_bd)
        #save_to_database(conn_db, [work])
        
        Dts= Dataset(label=file_name, local=file_name, size=file_size, n_line=n_line, n_column=n_column, workflow=work)
        save_to_database(conn_db, [work, Dts])
        st.write("O id para este  Workflow é:", work.id)
        if 'Dts' not in st.session_state:
	        st.session_state['Dts'] = Dts

        if 'workflow' not in st.session_state:
	        st.session_state['workflow'] = work

    
    if descriptionfile is not None:
        #convert_tipo é necessário para poder transformar o tipo no banco de dados para um literar, pois no formato de tipo de dados o bd não consegue converter
            #tipo_convertido = convert_tipo(df)
            tipo_convertido = convert_type(df)
            descricao = pd.read_csv(descriptionfile, sep=sep_text_input_des, encoding='iso-8859-1')
            #tuplas = list(zip(df.dtypes.index, df.dtypes))
            tuplas = list(zip(df.dtypes.index, tipo_convertido))
            dftipo = pd.DataFrame(tuplas, columns=['attribute', 'type'])
            dftipo1 = dftipo.merge(descricao, on='attribute')

                
            list_Dtsatr=[]
            for indice, linha in dftipo1.iterrows():
                Dts_atr= Dataset_Attribute(label=linha['attribute'], type=linha['type'], description=linha['description'], dataset=st.session_state['Dts'])
                list_Dtsatr.append(Dts_atr)

            save_to_dtsAtr(conn_db,list_Dtsatr)    
            #save_to_database(conn_db, list_Dtsatr)
            #for item in list_Dtsatr:
                #st.write(item.label)
                #if item.label not in st.session_state:
                    #st.session_state[item.label] = item

    else:           
        st.info("Por favor carregue o arquivo de descrição no menu lateral")   
           
            #Gravar no banco o workflow com nome do dataset
            #Mostra a informação do id do workflow
            #st.write(list_attribute)
    
     
    









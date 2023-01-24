import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from tkinter import Y, Button

import seaborn as sns
import matplotlib.pyplot as plt

from sqlalchemy.orm import sessionmaker, synonym
from db.connect_db import database
from db.model import save_to_database, Workflow, Dataset, Dataset_Attribute, OperatorsActivity, Experiment, Parameter, Experiment_Attribute, Xai, Xai_Results
from statistics import mode
from sqlalchemy.orm import sessionmaker

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix, precision_score, recall_score, accuracy_score, f1_score 

from utils import (binning, scaling, standardization, onehot_encoder,  ordinal_encoder,
                   over_sampling, under_sampling)



#def main():
    # -------------------------------- Sidebar -------------------------------
st.set_page_config(
page_title="PPM-ML",
page_icon="👋",
)

#st.title("Página principal")

titl_templ = """
        <div style="background-color: royalblue; padding: 15px;border-radius:15px">
        <h2 style="color: mintcream">Machine Learning Tool with XAI Support</h2>
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
    # read_file_data(file)

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
df = st.session_state["dataset"]
#if df is  None:
if df is None:
    st.markdown('<br>', unsafe_allow_html=True)
    titla_dts = """
            <div style= text-align: center ">
            <H5 style="color: Navy; font-style: italic;text-align: justify; font-family: "Times New Roman"">Esta ferramenta tem a finalidade de contribuir com a realização de tarefas de classificação de ML, capturando as informações da base de 
    dados, das operações de pré-processamento e com os resultados de treino. Também utiliza ferramenta XAI para melhor entendimento do resultado do modelo treinado. 
    As informações capturadas ficam disponíveis para consulta em uma base de dados.</H5>
            </div>
            """
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
    
 
    st.markdown('<br>', unsafe_allow_html=True)

    st.markdown('### 1 - Exploração de dados')
    #st.markdown('#### 1.1 - Dataset information')
    #st.checkbox('Mostrar dados brutos'):
    st.markdown('#### 1.1 - Informação do Dataset')
    #if st.checkbox('Display raw data'):
    st.markdown('<br>', unsafe_allow_html=True)
        # Melhor retirar por causa dos bancos grandes
        # value = st.slider('Choose the number of lines:',
        # min_value=1, max_value=100, value=5)
        # st.dataframe(df.head(value), width=900, height=600)
    #st.write('objeto Workflow:',st.session_state['work'])
    st.write(df)
    #st.markdown('** Dataset dimension**')
    st.write('**Dimensão do Dataset**')
        #st.markdown('#### 1.2 -  Dimensão do Dataset')
    st.markdown(df.shape)

    st.markdown('<br>', unsafe_allow_html=True)

    st.markdown('**Descrição estatística de colunas quantitativas**')
        # st.dataframe(df.describe(), width=900, height=600)
    st.write(df.describe())

    st.markdown('<br>', unsafe_allow_html=True)

    st.markdown( '**Informação do Dataset: Column name, Type, Numbers of NaNs (NULL), Percentage of NaNs and Unique Values**')

    num_features = df.select_dtypes(include=[np.number]).copy()
    cat_features = df.select_dtypes(exclude=[np.number]).copy()
    
    exploration = explore(df)
    st.dataframe(exploration, width=900, height=600)

    
    if st.checkbox('Carregar arquivo de descrição'):
        st.info( "Por favor carregue o arquivo de descrição no menu lateral")
        st.markdown('<br>', unsafe_allow_html=True)
        title_dts = """
            <div style="background-color: navy; padding: 1px;">
            <H3 style="color: mintcream">Carregar arquivo de descrição de atributos</H3>
            </div>
            """

        with st.sidebar.markdown(title_dts, unsafe_allow_html=True):
        #with st.sidebar.header('Upload do arquivo de descrição dos atributos'):
            sep_text_input_des = st.sidebar.text_input(
                'Insira o separador do arquivo : "," ou " ; "', value=';')
            descriptionfile = st.sidebar.file_uploader(
                "Carregue o arquivo de descrição", type=["csv"])

            if descriptionfile is not None:
                descricao = pd.read_csv(
                    descriptionfile, sep=sep_text_input_des, encoding='iso-8859-1')
                tuplas = list(zip(df.dtypes.index, df.dtypes))
                dftipo = pd.DataFrame(
                    tuplas, columns=['attribute', 'type'])
                dftipo = dftipo.merge(descricao, on='attribute')
                # minhas colunas são:
                # cria os objetos para armazenar no bd
                list_attribute = []
                tupla_ln = ()
                for indice, linha in dftipo.iterrows():
                    # list_attribute.append(linha["attribute"], linha["type"], linha["description"]])
                    attribute = linha['attribute']
                    tipo = linha['type']
                    description = linha['description']
                    tupla_ln = (attribute, tipo, description)

                    #Dts_att[indice] = Dataset_Attribute(label=attribute, type=tipo, description=description)
                    #list_attribute.append(Dts_att[indice])
                    list_attribute.append(tupla_ln)
                #st.write(list_attribute)
    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown('#### 1.2 - Análise Gráfica dos Dados')
    st.markdown('**Distribuição de colunas quantitativa e qualitativa**')
    # st.markdown('<br>', unsafe_allow_html=True)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    if st.checkbox('Plotar gráfico', key='21'):
        op6 = list(df.columns)
        op6.insert(0, 'Selecione uma opção')

        select_feature_quantitative = st.selectbox(
            'Selecione uma coluna', options=op6)

        if select_feature_quantitative not in 'Selecione uma opção':
            sns.countplot(y=select_feature_quantitative,
                        data=df, orient='h')
            plt.title(str(select_feature_quantitative), fontsize=14)
            st.pyplot()
        else:
            pass
    
    #st.markdown('#### 1.4 - Detecção de outlier')
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('**Detecção de Outlier**')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    if st.checkbox('Plotar gráfico bloxplot', key='21'):
        op = list(df.select_dtypes(include=[np.number]).columns)
        op.insert(0, 'Selecione uma opção')

        select_boxplot = st.selectbox('Escolha a coluna para plotar o gráfico boxplot:', options=op)

        if select_boxplot not in 'Selecione uma opção':
            if len(select_boxplot) > 0:
                colors= ['#00CED1']
                sns.boxplot(x=select_boxplot, data=df.select_dtypes(
                    include=[np.number]), palette=colors)
                st.pyplot(dpi=100)
        else:
            pass




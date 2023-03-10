import streamlit as st
from datetime import datetime
#from tkinter import Y, Button

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shap

# Import da base
from sqlalchemy.orm import sessionmaker
from db.connect_db import database
from db.model import save_to_database, save_to_operact, Workflow, Dataset, Dataset_Attribute, OperatorsActivity, Experiment, Parameter, Experiment_Attribute, Xai, Xai_Results
from statistics import mode

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  plot_confusion_matrix, precision_score, recall_score, accuracy_score, f1_score 
from sklearn.metrics import confusion_matrix, classification_report


from utils.preprocessing import scaling, standardization, onehot_encoder,  ordinal_encoder, convert_type, convert_type_column, over_sampling, under_sampling

dict_db = {

   
    'name_operator': [
        'Column Selection', 'DropColumn','Imputation-1', 'Imputation0','ImputationMean', 'ImputationMedian', 'ImputationMode', 
        'ImputationUnknown', 'Oversampling', 'Undersampling','IncludeColumn', 'MinMaxScaler', 'StandardScaler','OneHotEncoder', 'OrdinalEncoder', 'TrainTestSplit'
    ],

    'function_operator': ['Data Cleaning', 'Data Reduction', 'Data Sampling', 'Data Coding', 'Data Normalization', 'Attribute Construction', 'Data Partition'],
}


conn_db = database(is_table_log=True)#conexão com a base
df = st.session_state["dataset"]
#df2 = st.session_state["dataset"]
work = st.session_state['workflow']
Dts = st.session_state['Dts']

#num_lines=df.shape[1]
lst_types=[]
lst_types = convert_type(df)

#dicionários que foram inicializados em principal
attributes =  st.session_state['attributes'] 
attributes_type =  st.session_state['attributes_type']

titl_templ = """
        <div style="background-color: royalblue; padding: 15px;border-radius:15px">
        <h2 style="color: mintcream">Pré-processamento de Dados</h2>
        </div>
        """

st.markdown(titl_templ, unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)
st.write('Estado atual do Dataset:')
st.markdown('<br>', unsafe_allow_html=True)
st.write(st.session_state["dataset"].head(5))


#def main():
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('#### 2.1 - Tratar valores faltantes')


op7 = list(df.columns)
op7.insert(0, 'Selecione uma opção')
#cat_atributos_delete = []
atributos_delete = []
missing_value_option = st.radio("Selecione a ação:", ("Apagar atributo", "Preencher valores faltantes"))

num_features = df.select_dtypes(include=[np.number]).copy()
cat_features = df.select_dtypes(exclude=[np.number]).copy()
col_num=list(num_features.columns)
col_cat = list(cat_features.columns)

if missing_value_option == "Apagar atributo":
    column_missing_to_remove = st.selectbox('Informe  para qual coluna deseja remover em virtude da existência de valores faltantes:', options=op7)
    botao_missing = st.button("Pode apagar!!", key='Rmattr') 
    if botao_missing and column_missing_to_remove is not "Selecione uma opção":
    #df = df.drop(list(column_missing_to_remove), axis=1).reset_index(drop=True)

        df.drop(column_missing_to_remove, axis=1,inplace=True)
        st.success('atributo apagado com sucesso!')

        atributos_delete.append(column_missing_to_remove)
        #apagar o atributo nos dicionários:
        del attributes[column_missing_to_remove]
        del attributes_type[column_missing_to_remove]
        st.write(df.head(5))

        operact=OperatorsActivity(name=dict_db['name_operator'][1],function=dict_db['function_operator'][0], workflow=work, label_attribute=str(column_missing_to_remove))

        save_to_operact(conn_db,operact)
    else:
        pass

elif missing_value_option == "Preencher valores faltantes":
    column_missing_to_fill = st.selectbox('Informe  para qual coluna  deseja realizar ação em virtude da existência de valores faltantes:', options=op7)
    if column_missing_to_fill   not in col_cat: 
        imputer = st.selectbox('Escolha uma opção de imputação:', options=(
        'Selecione uma opção',
        'Imputar com -1',
        'Imputar com 0',
        'Imputar com média',
        'Imputar com mediana',
        'Imputar com moda'))  

        if st.button("Pode prosseguir", key="filnul") and imputer is not "Selecione uma opção":

            if imputer == "Imputar com -1":
                                
                df[column_missing_to_fill].fillna(-1, inplace=True)
                st.success('Valores faltantes preenchidos com sucesso!')
                st.write(df.head(5))
                #grava no banco
                operact=OperatorsActivity(name=dict_db['name_operator'][2],function=dict_db['function_operator'][0], workflow=work, label_attribute=str(column_missing_to_fill))
                save_to_operact(conn_db,operact)
            
            elif imputer == 'Imputar com 0':

                df[column_missing_to_fill].fillna(0, inplace=True)
                st.success('Valores faltantes preenchidos com sucesso!')
                st.write(df.head(5))
                #grava no banco
                operact=OperatorsActivity(name=dict_db['name_operator'][3],function=dict_db['function_operator'][0], workflow=work, label_attribute=str(column_missing_to_fill))
                save_to_operact(conn_db,operact)
            
            elif imputer == 'Imputar com média':

                df[column_missing_to_fill].fillna((df[column_missing_to_fill].mean()), inplace=True)
                st.success('Valores faltantes preenchidos com sucesso!')
                st.write(df.head(5))
                #grava no banco
                operact=OperatorsActivity(name=dict_db['name_operator'][4],function=dict_db['function_operator'][0], workflow=work, label_attribute=str(column_missing_to_fill))
                save_to_operact(conn_db,operact)
    
            elif imputer ==  'Imputar com mediana':

                df[column_missing_to_fill].fillna((df[column_missing_to_fill].median()), inplace=True)
                st.success('Valores faltantes preenchidos com sucesso!')
                st.write(df.head(5))
                #grava no banco
                operact=OperatorsActivity(name=dict_db['name_operator'][5],function=dict_db['function_operator'][0], workflow=work, label_attribute=str(column_missing_to_fill))
                save_to_operact(conn_db,operact)

            else:
                modacol =  df[column_missing_to_fill].mode()[0]
                df[column_missing_to_fill].fillna(modacol, inplace=True)
                st.success('Valores faltantes preenchidos com sucesso!')
                st.write(df.head(5))
                #grava no banco
                operact=OperatorsActivity(name=dict_db['name_operator'][6],function=dict_db['function_operator'][0], workflow=work, label_attribute=str(column_missing_to_fill))
                save_to_operact(conn_db,operact)


        else:
            pass

    else:
        imputer = st.selectbox('Escolha uma opção de imputação:', options=(
        'Selecione uma opção',
        'Imputar com moda',
        'Imputar com Desconhecido'))

        if st.button("Pode prosseguir!", key="filnul1") and imputer is not "Selecione uma opção":
            if imputer == "Imputar com moda":
                        
                modacol =  df[column_missing_to_fill].mode()[0]
                df[column_missing_to_fill].fillna(modacol, inplace=True)
                st.success('Valores faltantes preenchidos com sucesso!')
                st.write(df.head(5))
                #grava no banco
                operact=OperatorsActivity(name=dict_db['name_operator'][6],function=dict_db['function_operator'][0], workflow=work, label_attribute=str(column_missing_to_fill))
                save_to_operact(conn_db,operact)
        
    
            elif imputer == 'Imputar com Desconhecido':

                df.fillna('Desconhecido', inplace=True)
                st.success('Valores faltantes preenchidos com sucesso!')
                st.write(df.head(5))
                #grava no banco
                operact=OperatorsActivity(name=dict_db['name_operator'][7],function=dict_db['function_operator'][0], workflow=work, label_attribute=str(column_missing_to_fill))
                save_to_operact(conn_db,operact)


st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown('#### 2.2 - Criação de atributos manualmente')
#st.markdown('#### 2.2 - Criação de atributos manualmente')
st.info('Nesta seção será possível a criação de um novo atributo a partir de comando manual. Digite abaixo o comando para criação do atributo.' )
#nome_atributo = st.text_input("Entre aqui com o nome do atributo a ser criado","Nome do atributo",
#key="atributocriado", max_chars =20)

comando = st.text_area('Digite o comando para criação do atributo. Para se referenciar ao dataset, use "df".')

criatributo = st.button('Pode Executar!')
if criatributo and comando is not None:
#     df[nome_atributo] =  exec(comando)
#     st.write(df.head(5))
    exec(comando)
    st.session_state['dataset'] = df
    st.write(df.head(5))
    #Pegar o nome do novo atributo, capturar o tipo e gravar no banco
    for column in df.columns:
        if column  not in attributes:
            attributes[column] = str(comando)
            attributes_type[column] = convert_type_column(df, column)
            #GRAVAR NO BANCO

            operact=OperatorsActivity(name=dict_db['name_operator'][10],function=dict_db['function_operator'][5], workflow=work, label_attribute=str(column))
            save_to_operact(conn_db,operact)

    st.success('Comando executado com  sucesso!')

#Gravar no banco

#st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<br><br>', unsafe_allow_html=True)

st.markdown('#### 2.3 - Seleção  de Dados ')
st.write('Correlação entre atributos')

select_corr = st.selectbox(' Escolha o método de correlação que deseja utilizar para analisar os atributos quantitativos:', options=(
'Selecione uma opção', 'pearson', 'spearman'))

if select_corr != 'Selecione uma opção':
    if df.shape[1] <= 30:
        plt.rcParams['figure.figsize'] = (10, 8)
        sns.heatmap(num_features.corr(method=select_corr), annot=True,
                        linewidths=0.5, linecolor='black', cmap='Blues')
        st.pyplot(dpi=100)
    else:
        plt.rcParams['figure.figsize'] = (20, 10)
        sns.heatmap(num_features.corr(method=select_corr), annot=True,
                    linewidths=0.5, linecolor='black', cmap='Blues')
        st.pyplot(dpi=100)

opcao_acao = 'nenhuma'
if st.checkbox('Desejo remover alguns atributos do dataset (Melhor opção quando serão removidos poucos atributos'):
    opcao_acao = 'apagar'
    op8 = list(df.columns)
    op8.insert(0, 'Selecione uma opção')
    columns_to_remove = st.multiselect('Informe os atributos que deseja remover do dataset:', options=op8)
    deleta_coluna = st.button('Pode remover!')
    if  deleta_coluna and columns_to_remove != 'Selecione uma opção':

#df = df.drop((columns_to_remove), axis=1)
        df.drop(columns_to_remove, axis=1,inplace=True)
        atributos_delete.append(columns_to_remove)
        st.success('Atributos removidos com sucesso!')              
        st.write(df.head(5))

        lst_itens = []
        #gravar no banco
        for column in columns_to_remove:
            #VERIFICAR  O NAME_OPERATOR(Drop Column) E O FUNCTION_OPERATOR(data_reduction)
            operact=OperatorsActivity(name=dict_db['name_operator'][1],function=dict_db['function_operator'][1], workflow=work, label_attribute=str(column))
            lst_itens.append(operact)
        save_to_database(conn_db,lst_itens)

if opcao_acao != 'apagar':
    if st.checkbox('Desejo selecionar os atributos do dataset para o treino!') :
        num_features_add = st.multiselect( 'Selecione os atributos para incluir (Melhor opção quando poucos atributos do dataset forem selecionados)', options=df.columns)
        if st.button('Seleção finalizada!'):
            df=df[num_features_add]
            st.write('Novas Colunas do dataset: ', df.columns)
            #usa name operator é o column_selection e o operator é o data reduction e o  function_operator: 'data reduction'
            lst_columns = []
            for column in num_features_add:
                operact=OperatorsActivity(name=dict_db['name_operator'][10],function=dict_db['function_operator'][1], workflow=work, label_attribute=str(column))
                lst_columns.append(operact)
            save_to_database(conn_db,lst_columns)
            #gravar no banco
#atualiza a variável de estado
st.session_state['dataset'] = df
#     st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown('#### 2.4 - Transformação de dados')
st.info('Nesta seção estarão disponíveis técnicas de transformação de dados: Normalização e Padronização para dados numéricos e OneHot Encoder e Ordinal Encoder para os categóricos.')
num_features = df.select_dtypes(include=[np.number]).copy()
cat_features = df.select_dtypes(exclude=[np.number]).copy() 

if 'dataset' not in st.session_state:
    st.session_state['dataset'] = df
op9 = list(df.columns)
op9.insert(0, 'Selecione uma opção')
op1 =  list(df.columns)
        
ocorreu_onehot_transform = False
ocorreu_ordinal_transform = False

no_preprocessed = []

transf_numerica ='não'
st.write('Escolha o tipo do atributo a ser tratado:')
page_names = ['Categorico', 'Numérico']
page = st.radio('Atributo', page_names)
if page =='Categorico':
    categorico_tipo = ['OneHot Encoder', 'Ordinal Encoder']
    page_nom_tipo = st.radio ('Escolha o tipo de transformação para a variável categórica:', categorico_tipo, horizontal=True)
    if page_nom_tipo == 'OneHot Encoder':
        atributos_cat_nom = st.multiselect('Escolha as variáveis categórica nominais a serem aplicadas a transformação(OneHot Encoder)',options=op9)
        if st.button('Pode aplicar!') and atributos_cat_nom != "Selecione uma opção":
        #usando dummies  
            ocorreu_onehot_transform = True 
            onehot_transform = onehot_encoder(cat_features, atributos_cat_nom) 
            df.drop(atributos_cat_nom, axis=1,inplace=True)
            cat_features = pd.concat([cat_features, onehot_transform], axis=1).reset_index(drop=True)
            df = pd.concat([onehot_transform, num_features], axis=1).reset_index(drop=True)    
            st.session_state['dataset'] = df 
            #guardar os atributos gerados no dicionário
            for column in df.columns:
                if column  not in attributes:
                    nome_original = str(column).split("_")
                    attributes[column] = "atributo gerado pelo onehot Encoder aplicado ao atributo "+str(nome_original[0])
                    attributes_type[column] = convert_type_column(df, column)   

           
            
            if  atributos_cat_nom:
                #guarda os atributatributos_cat_nomos originais que geraram o processamento
                no_preprocessed.extend(atributos_cat_nom)
                st.success('Transformação realizada com sucesso!')
                st.write(df.head(5))
                #apagar atributos que receberam onehot

        #grava no banco
            #name_operator = One-Hot Encoder; function = DataTransformation
            lst_columns = []
            for column in atributos_cat_nom:
                operact=OperatorsActivity(name=dict_db['name_operator'][13],function=dict_db['function_operator'][3], workflow=work, label_attribute=str(column))
                lst_columns.append(operact)
                del attributes[column]
                del attributes_type[column]
            save_to_database(conn_db,lst_columns)
            #incluir no dicionário de novas colunas
    else:
        atributos_cat_ord= st.multiselect('Escolha as variáveis categóricas ordinais a serem aplicadas a transformação (Ordinal Encoder)',options=op9)
        if  st.button('Pode aplicar!!') and  atributos_cat_ord != "Selecione uma opção":
            ocorreu_ordinal_transform = True
            ordinal_transform = ordinal_encoder(cat_features[atributos_cat_ord])
            df.drop(atributos_cat_ord, axis=1,inplace=True)

            #atualiza cat_features
            cat_features.drop(atributos_cat_ord, axis=1,inplace=True)
            df = pd.concat([ordinal_transform, num_features], axis=1).reset_index(drop=True)
            st.session_state['dataset'] = df 
                        
            
            for column in df.columns:
                if column  not in attributes:
                    attributes[column] = "atributo gerado pelo Ordinal Encoder aplicado ao atributo {column}"
                    attributes_type[column] = convert_type_column(df, column)                               

            if  atributos_cat_ord:

                no_preprocessed.extend(atributos_cat_ord)
                st.success('Transformação realizada com sucesso!')
                st.write(df.head(5))
                #grava no banco
            #name_operator = Ordinal Encoder; function = DataTransformation
            lst_columns = []
            for column in atributos_cat_ord:
                operact=OperatorsActivity(name=dict_db['name_operator'][14],function=dict_db['function_operator'][3], workflow=work, label_attribute=str(column))
                lst_columns.append(operact)
                del attributes[column]
                del attributes_type[column]
            save_to_database(conn_db,lst_columns)
            
    if ocorreu_onehot_transform:

        df = pd.concat([num_features, onehot_transform], axis=1).reset_index(drop=True)
        
    if ocorreu_ordinal_transform:

        df = pd.concat([num_features, ordinal_transform], axis=1).reset_index(drop=True)

    if ocorreu_onehot_transform and ocorreu_ordinal_transform:
        cat_features = pd.concat([onehot_transform, ordinal_transform], axis=1).reset_index(drop=True)
        df = pd.concat([num_features, cat_features], axis=1).reset_index(drop=True)
    #atualiza a variável de estado
    st.session_state['dataset'] = df    
        


else:

    op10=  list(df.columns)
    op10.insert(0, 'Selecione uma opção')
    transformacao=['Escolha uma opção:','Standarlization','Normalization']
    select_target= st.selectbox("Selecione o atributo target", options=op10)
    tipo_transform_num = st.selectbox("Selecione o tipo de transformação a ser aplicada", options=transformacao)
    if st.button('Pode aplicar!') and transformacao != 'Escolha uma opção':
        
        if tipo_transform_num == 'Standarlization':
        #aplica a transformação
            num_features = standardization(num_features, select_target)
            df = pd.concat([cat_features, num_features], axis=1).reset_index(drop=True)
            st.success('Transformação de Padronização realizada com sucesso!') 
            num_features = df.select_dtypes(include=[np.number]).copy()
            cat_features = df.select_dtypes(exclude=[np.number]).copy() 
            #grava no banco

            columns =  list(df.columns)
            lst_columns = []
            for column in columns:
                operact=OperatorsActivity(name=dict_db['name_operator'][12],function=dict_db['function_operator'][4], workflow=work, label_attribute=str(column))
                lst_columns.append(operact)
            save_to_database(conn_db,lst_columns)
            
            transf_numerica = 'sim'
            st.write(df.head(5))
        
        else:
                
            num_features = scaling(num_features, select_target)
            df = pd.concat([cat_features, num_features], axis=1).reset_index(drop=True)
            num_features = df.select_dtypes(include=[np.number]).copy()
            cat_features = df.select_dtypes(exclude=[np.number]).copy() 
            st.success('Transformação de Normalização realizada com sucesso!') 
            columns =  list(df.columns)
            
            lst_columns = []
            for column in columns:
                operact=OperatorsActivity(name=dict_db['name_operator'][11],function=dict_db['function_operator'][4], workflow=work, label_attribute=str(column))
                lst_columns.append(operact)
            save_to_database(conn_db,lst_columns)
            
            st.write(df.head(5))
            transf_numerica = 'sim'
            #grava no banco


if transf_numerica == 'sim':
    df = pd.concat([num_features, cat_features], axis=1).reset_index(drop=True)

#atualiza a variável de estado
st.session_state['dataset'] = df


#st.write('df antes de entrar em partição de dados', df)
st.write('df antes de entrar em partição de dados',st.session_state['dataset'])

#df_processado =  df.copy()
st.markdown('<hr size=15>', unsafe_allow_html=True)
st.markdown('### 2.5 - Partição de Dados ')

verifica_atcategorico = list(st.session_state['dataset'].select_dtypes(exclude=[np.number]).columns)

st.write(verifica_atcategorico)
#if verificar_dataset == 'Sim':
if len(verifica_atcategorico) > 0:
    st.write('Ainda existem atributos categóricos no dataset! Revise o processamento!!')
    st.write(st.session_state['dataset'].head(5))

else:
    st.markdown('<br>', unsafe_allow_html=True)
    page_partition = ['Particionar dataset', 'Fazer download do dataset']
    page_select = st.radio('O que você gostaria de fazer agora?', page_partition)

    if page_select == 'Particionar dataset':
        

        op5 = list(st.session_state['dataset'].columns)
#op5.insert(0, 'Selecione o atributo')

        target = st.selectbox('Entre com a coluna target para divisão do dataset:', options=list(op5))
        y = st.session_state['dataset'][target] #Define a coluna alvo
        X = st.session_state['dataset'].drop(columns=[target])
        st.write(X)
        st.write(y)
        if 'y' not in st.session_state:
            st.session_state['y'] =  y

#st.write('Confirma a proporção? ', split_size,'para treino',100-split_size, ' para teste')
        st.subheader('** Escolha a taxa de Divisão do Dataset**')
        split_size = st.slider('Taxa de divisão (% para Training Set)', 10, 90, 80, 5, key='slider_splitsize')    
        global X_train, X_test, y_train, y_test
        
          
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_size)/100, random_state=42)
        if 'X_train' not in st.session_state:
            st.session_state['X_train'] = X_train
        if 'X_test' not in st.session_state:
            st.session_state['X_test'] = X_test
        if 'y_train' not in st.session_state:
            st.session_state['y_train'] = y_train
        if 'y_test' not in st.session_state:
            st.session_state['y_test'] = y_test  
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.write(X.shape)
        st.write(y.shape)
                # st.markdown('**Detalhes das variáveis**:')
        st.markdown('** Divisão dos dados**')
        st.write('Training set')
        st.info(X_train.shape)
        st.write('Test set')
        st.info(X_test.shape)

        st.markdown('**Detalhes das variáveis**:')
        st.write('Variável X')
        st.info(list(X.columns))
        st.write('Variável Y')
        st.info(y.name)
        treino_ok=st.button('Dividir o dataset',key='slider_button')
       
        
        if treino_ok:
            lst_columns_split = []

            for column in df.columns:
                operact=OperatorsActivity(name=dict_db['name_operator'][15],function=dict_db['function_operator'][6], workflow=work, label_attribute=str(column))
                lst_columns_split.append(operact)

 
            save_to_database(conn_db, lst_columns_split)  
 
            st.success('Para criação do modelo, vá até a página ML!')

    else:
        
        @st.experimental_memo
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')


        csv = convert_df(df)

        st.download_button("Press to Download",csv,
        "file.csv",
        "text/csv",
        key='download-csv'
        )
st.markdown('<hr size=15>', unsafe_allow_html=True)
st.markdown('### 2.6 - Balanceamento das Classes ')

selection=['Sim','Não']
page_selection = st.radio('Precisa realizar o balanceamento das classes?', selection)

if page_selection == 'Sim':

    st.markdown('<br>', unsafe_allow_html=True)
    method_balance = st.selectbox('Escolha o método que deseja usar:', options=('Escolha uma opção','Oversampling', 'Undersampling' ))

    if method_balance == 'Oversampling':
        if st.button('Pode corrigir!'):
            try:

                X_train, y_train = over_sampling(st.session_state['X_train'],st.session_state['y_train'])

                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train

                lst_columns = []
                for column in df.columns:
                    operact=OperatorsActivity(name=dict_db['name_operator'][8],function=dict_db['function_operator'][2], workflow=work, label_attribute=str(column))
                    lst_columns.append(operact)
                save_to_database(conn_db,lst_columns) 
                st.success('Oversampling aplicado com sucesso!')
                #st.write((pd.Series(st.session_state['y_train']).value_counts()))
                st.write((pd.Series(st.session_state['y_train']).value_counts()))

                            
            except Exception as e:
                st.markdown(e)
    elif method_balance == 'Undersampling': 
        if st.button('Pode corrigir!'):
            try:
                undersampling=1
                X_train, y_train = under_sampling(st.session_state['X_train'],st.session_state['y_train'])
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                #gravar no banco

                lst_columns = []
                for column in df.columns:
                    operact=OperatorsActivity(name=dict_db['name_operator'][9],function=dict_db['function_operator'][2], workflow=work, label_attribute=str(column))
                    lst_columns.append(operact)
                save_to_database(conn_db,lst_columns) 
                st.success('Undersampling aplicado com sucesso!')
 
                st.write((pd.Series(st.session_state['y_train']).value_counts()))

 
            except Exception as e:
                st.markdown(e)
        else:
            pass  
  


                    
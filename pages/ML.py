import streamlit as st
from datetime import datetime
from tkinter import Y, Button

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import os
import shap

# Import da base
from sqlalchemy.orm import sessionmaker, synonym
from db.connect_db import database
from db.model import save_to_database, save_to_experim, Workflow, Dataset, Dataset_Attribute, OperatorsActivity, Experiment, Parameter, Experiment_Attribute, Xai, Xai_Results
from statistics import mode
# from sqlalchemy.orm import sessionmaker
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix, precision_score, recall_score, accuracy_score, f1_score 
from sklearn.metrics import confusion_matrix, classification_report

from utils import (over_sampling, under_sampling)



conn_db = database(is_table_log=True)
df = st.session_state["dataset"]
#df2 = st.session_state["dataset"]
work = st.session_state['workflow']
Dts = st.session_state['Dts']
attributes =  st.session_state['attributes'] 
attributes_type =  st.session_state['attributes_type']
#if 'modelo' not in st.session_state:
#st.session_state['modelo'] = rf

titl_templ = """
        <div style="background-color: royalblue; padding: 15px;border-radius:15px">
        <h2 style="color: mintcream">Treinar modelo / XAI </h2>
        </div>
        """

st.markdown(titl_templ, unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<hr size=15>', unsafe_allow_html=True)
#st.markdown('### 3 - Train DataSet')
df = st.session_state["dataset"]
X_train=st.session_state['X_train']
y_train=st.session_state['y_train']
X_test=st.session_state['X_test']
y_test=st.session_state['y_test']

st.markdown('<br>', unsafe_allow_html=True)
st.subheader('3.1 - Treinamento do modelo')
classification_models = ['Random Forest','Support Vector Machine' ]
class_names = st.session_state['y'].unique()

classification_tipo = st.radio ('Escolha o algoritmo:', classification_models, horizontal=True)
if classification_tipo == 'Random Forest':

    st.markdown('#### Parâmetros do Modelo')
    parameter_n_estimators  = st.number_input("Numero de árvores (n_estimators)", min_value=0,max_value=1000,value=100, step=100)
    parameter_max_features = st.selectbox("Máximo de recursos (max_features)", ("sqrt", "log2"))
    parameter_min_samples_split  = st.number_input("O número mínimo de amostras por nó interno (min_samples_split)",1, 10, 2, 1)
    parameter_min_samples_leaf = st.number_input('O número mínimo de amostras por folha (min_samples_leaf)', 1, 10, 1, 1)
    parameter_random_state = st.number_input('Número da semente (random_state)', 0, 1000, 42, 1,)
    parameter_max_depth = st.number_input('Profundidade máxima da árvore (max_depth)', None, 128, 2, 4)

    treinar = st.button("Treine meu Modelo", key='Treino_modelo')

    if treinar:
       
        rf = RandomForestClassifier(n_estimators=parameter_n_estimators,\
        random_state=parameter_random_state,\
        max_features=parameter_max_features,\
        min_samples_split=parameter_min_samples_split,\
        min_samples_leaf=parameter_min_samples_leaf,
        max_depth=parameter_max_depth
        )
#rf.fit(X_train, y_train)
        rf.fit(st.session_state['X_train'],st.session_state['y_train'])
        y_pred = rf.predict(st.session_state['X_test'])
        if 'y_pred' not in st.session_state:
            st.session_state[ 'y_pred'] = y_pred
        if 'modelo' not in st.session_state:
            st.session_state['modelo'] = rf
        st.session_state[ 'y_pred'] = y_pred
        st.session_state['modelo'] = rf
    #Calculo das variáveis de predição
        accuracy = accuracy_score(st.session_state['y_test'], st.session_state['y_pred'])
        precision = precision_score(st.session_state['y_test'], st.session_state['y_pred'], average='weighted')
        recall = recall_score(st.session_state['y_test'], st.session_state['y_pred'], average='weighted')
        f1score = f1_score(st.session_state['y_test'], st.session_state['y_pred'], average='weighted')
        st.subheader('3.2- Random Forest Performance')          
        
        #GRAVAR EXPERIMENTO 
        #id, timestamp, method, accuracy, recall, precision, f1score
        experim = Experiment(method="Random Forest", accuracy=accuracy, recall=recall, precision=precision, f1score=f1score, dataset=Dts)
        #save_to_experim(conn_db,experim)  

        st.write("Acurácia: ", accuracy.round(4) )
        st.write("Precisão: ", precision.round(4) )
        st.write("Recall: ", recall.round(4) )
        st.write("f1score: ", f1score.round(4) ) 

        #Relatório de classificação
        st.text('Model Report:\n    ' + classification_report(y_test, y_pred))
   
        #param=Parameter(label=,value=, experiment=exp,)
        #GRAVAR PARÂMETROS NO experimento
        lst_param = []
        #juntar o experimento  na lst_param
        lst_param.append(experim)
        n_estimator = Parameter(label="n_estimators", value=parameter_n_estimators, experiment=experim)
        lst_param.append(n_estimator)
        max_features =  Parameter(label="max_features", value=parameter_max_features, experiment=experim)
        lst_param.append(max_features)
        min_samples_split =  Parameter(label="min_samples_split", value=parameter_min_samples_split, experiment=experim)
        lst_param.append(min_samples_split)
        min_samples_leaf =  Parameter(label="min_samples_leaf", value=parameter_min_samples_leaf, experiment=experim)
        lst_param.append(min_samples_leaf)
        random_state =  Parameter(label="random_state", value=parameter_random_state, experiment=experim)
        lst_param.append(random_state)
        max_depth =  Parameter(label="max_depth", value=parameter_max_depth, experiment=experim)
        lst_param.append(max_depth)

        save_to_database(conn_db,lst_param)

 
        st.set_option('deprecation.showPyplotGlobalUse', False)
        #st.subheader("2.2. Matriz de Confusão")
        st.subheader("3.3. Matriz de confusão")

        sns.heatmap(confusion_matrix(st.session_state['y_test'], st.session_state['y_pred']), cmap='OrRd', annot=True, fmt='2.0f', annot_kws={"size": 20, "weight": "bold"})        

        st.pyplot()



        if 'experimento' not in st.session_state:
	        st.session_state['experimento'] = experim 
        st.session_state['experimento'] = experim
        
        lst_exp_attr = [] #para guardar os objetos de attribute_processed
                    
        for column in st.session_state['dataset'].columns:

            exp_attribute = Experiment_Attribute(label=str(column), type=str(attributes_type[column]),experiment=experim, origin=str(attributes[column]))
            lst_exp_attr.append(exp_attribute)

        save_to_database(conn_db, lst_exp_attr)  

else:# model =="Support Vector Machine":  
        selection_range_random_state = [x for x in range(100)]
        kernels = ['rbf','linear', 'poly', 'sigmoid']

        st.markdown(' *Parâmetro: *')     

        random_val = st.select_slider('Select random state value', options=selection_range_random_state)

        st.markdown(' *Kernel: *')     
        kernel = st.selectbox("Selecione o kernel", options=kernels)

        #y_pred = rf.predict(X_test)
        y_pred = rf.st.session_state['X_test']

        st.markdown('<br>', unsafe_allow_html=True)

        st.markdown('<hr size=15>', unsafe_allow_html=True)
        st.subheader('3.1 - SVM Performance') 

        st.markdown('<br>', unsafe_allow_html=True)
        #accuracy = rf.score(X_test, Y_test)
        accuracy = accuracy_score(st.session_state['y_test'], st.session_state['y_pred'])
        precision = precision_score(st.session_state['y_test'], st.session_state['y_pred'], average='weighted')
        recall = recall_score(st.session_state['y_test'], st.session_state['y_pred'], average='weighted')
        f1score = f1_score(st.session_state['y_test'], st.session_state['y_pred'], average='weighted')
        st.subheader('3.2 - Random Forest Performance')          

        st.write("Acurácia: ", accuracy.round(4) )
        st.write("Precisão: ", precision.round(4) )
        st.write("Recall: ", recall.round(4) )
        st.write("f1score: ", f1score.round(4) ) 

        
        #st.set_option('deprecation.showPyplotGlobalUse', False)
        #st.subheader("2.2. Matriz de Confusão")
        st.subheader("3.2. Matriz de confusão")
        plot_confusion_matrix(st.session_state['modelo'], st.session_state['X_test'], st.session_state['y_test'], display_labels=class_names, cmap='OrRd', annot_kws={"size": 20, "weight": "bold"})
        #st.pyplot()
        #sns.heatmap(confusion_matrix(st.session_state['y_test'], st.session_state['y_pred']), cmap='OrRd', annot=True, fmt='2.0f')
        
        plt.title('Random Forest')
        plt.ylabel('P R E V I S T O')
        plt.xlabel('R E A L')
        plt.show()

st.markdown('<hr size=15>', unsafe_allow_html=True)
st.markdown('#### 3.3. Gráfico XAI')
option_xai = ['SHAP','LIME' ]

xai_choice = st.radio ('Escolha o XAI:', option_xai, horizontal=True)
if xai_choice == 'SHAP':
    st.write('---')
    explainer = shap.TreeExplainer(st.session_state['modelo'])
    shap_values = explainer.shap_values(st.session_state['X_train'])
    max_features_shap = st.number_input("Digite o valor máximo de atributos a serem considerados no gráfico:", 2, 200, 10, 1,)
    if st.button("Show Gráfico SHAP", key='SHAP'):

        st.subheader('Feature Importance based on SHAP values')
        shap.summary_plot(shap_values, st.session_state['X_train'], max_display=max_features_shap)
        st.pyplot(bbox_inches='tight')
        st.write('---')
        #lista de objetos XAI
        lst_xai=[]
        obj_xai = Xai(method="shap",set_input="X_train", max_features=max_features_shap, experiment=st.session_state['experimento'],idx_instance= -1 )
        vals= np.abs(shap_values).mean(0)
        lst_xai.append(obj_xai)
        #Cria o dataframe com os valores shap do modelo
        feature_importance = pd.DataFrame(list(zip(st.session_state['X_train'].columns, sum(vals))), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        #set_input = "X_train"
        #para gráficos globais usei a idx_instance -1
        for indice, linha in feature_importance.iterrows():
            o_xai_res=Xai_Results(label_feature_importance=linha['col_name'], value_Feature_importance=linha['feature_importance_vals'], xai=obj_xai)
            lst_xai.append(o_xai_res)
        save_to_database(conn_db, lst_xai)      
else: 
    st.info('Solução LIME  ainda não implementada!')


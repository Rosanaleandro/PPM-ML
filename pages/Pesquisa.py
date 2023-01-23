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
from sqlalchemy.orm import sessionmaker
from db.connect_db import database
from db.model import save_to_database, Workflow, Dataset, Dataset_Attribute, OperatorsActivity, Experiment, Parameter, Experiment_Attribute, Xai, Xai_Results
from statistics import mode

import streamlit as st


#st.title("Contact")
titl_templ = """
        <div style="background-color: royalblue; padding: 15px;border-radius:15px">
        <h2 style="color: mintcream">Pesquisa de Workflow </h2>
        </div>
        """

st.markdown(titl_templ, unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)
#st.markdown('<hr size=15>', unsafe_allow_html=True)
#st.markdown('### 4 - Train DataSet')
#df = st.session_state["dataset"]

conn_db = database(is_table_log=True)

@st.cache(suppress_st_warning=True)
def retorna_workflow(num_workflow) -> Workflow:
    
    
    Session = sessionmaker(bind=conn_db, expire_on_commit = False)
    session = Session()

    return session.query(Workflow).filter(Workflow.id==num_workflow).one_or_none()

def retorna_dataset(num_workflow) -> Dataset:
    
    
    Session = sessionmaker(bind=conn_db, expire_on_commit = False)
    session = Session()

    return session.query(Dataset).filter(Dataset.workflowid==num_workflow).one_or_none()

#para retornar os atributos originais do dataset
def atributos_originais_dataset(num_workflow) -> Dataset_Attribute:
    
    Session = sessionmaker(bind=conn_db, expire_on_commit = False)
    session = Session()
 
    datasetatributo: list[Dataset_Attribute] = session.query(Dataset_Attribute).join(Dataset).join(Workflow).filter(Workflow.id==num_workflow).all()

    return datasetatributo

def preproc(num_workflow):
    
    Session = sessionmaker(bind=conn_db, expire_on_commit = False)
    session = Session()
 
    preprocall: list[OperatorsActivity] = session.query(OperatorsActivity).join(Workflow).filter(Workflow.id==num_workflow).all()

    return preprocall

def preprocatrib(num_workflow, nome_atributo):
    
    Session = sessionmaker(bind=conn_db, expire_on_commit = False)
    session = Session()
 
    preprocatr: list[OperatorsActivity] = session.query(OperatorsActivity).join(Workflow).filter(Workflow.id==num_workflow).filter(OperatorsActivity.label_attribute==nome_atributo).all()

    return preprocatr

def cons_experimentos(num_workflow):
    Session = sessionmaker(bind=conn_db, expire_on_commit = False)
    session = Session()
 
    experimentocon: list[Experiment] = session.query(Experiment).join(Dataset).join(Workflow).filter(Workflow.id==num_workflow).all()

    return experimentocon

def cons_xai(num_workflow):
    Session = sessionmaker(bind=conn_db, expire_on_commit = False)
    session = Session()
 
    consxai: list[Xai] = session.query(Xai).join(Experiment).join(Dataset).join(Workflow).filter(Workflow.id==num_workflow).all()

    return consxai

def cons_xairesult(num_xai):
    Session = sessionmaker(bind=conn_db, expire_on_commit = False)
    session = Session()
 
    #consxair: list[Xai_Results] = session.query(Xai_Results).join(Xai).join(Experiment).join(Dataset).join(Workflow).filter(Workflow.id==num_workflow).all()
    consxair: list[Xai_Results] = session.query(Xai_Results).join(Xai).filter(Xai.id==num_xai).all()
    return consxair  

def cons_Expattribute(num_Experim): 
    Session = sessionmaker(bind=conn_db, expire_on_commit = False)
    session = Session()
 
    consattrib: list[Experiment_Attribute] = session.query(Experiment_Attribute).join(Experiment).filter(Experiment.id==num_Experim).all()

    return consattrib  

def cons_param(num_Experim):

    Session = sessionmaker(bind=conn_db, expire_on_commit = False)
    session = Session()
 
    consparam: list[Parameter] = session.query(Parameter).join(Experiment).filter(Experiment.id==num_Experim).all()

    return consparam  

###===================================================================================================
st.markdown('<br>', unsafe_allow_html=True)
st.subheader('Pesquisa Workflow')
num_wkf = st.text_input('Digite o número do workflow que deseja informações:')

consulta = st.radio(
    "O que deseja consultar agora?",
    ('Informações dos atributos originais', 'Informações do dataset', 'Pré-processamento','Experimentos','XAI'))
if st.button('OK'):
    if consulta == "Informações dos atributos originais":
        datasatr = atributos_originais_dataset(num_wkf)
        if datasatr is not None:

            obj=[]
            for item in datasatr:
                lista_item=[]
                id=item.id
                lista_item.append(id)
                label=item.label
                lista_item.append(label)
                tipo=item.type
                lista_item.append(tipo)
                descricao=item.description
                lista_item.append(descricao)
                obj.append(lista_item)
            #st.write(obj) 
            resultado = pd.DataFrame(obj, columns=['Id', 'label', 'Tipo','Descrição do atributo'])
            st.info("Informação dos atributos originais do Dataset")
            st.table(resultado)   
 
        else:
            st.write('A consulta não retornou informação!')        

    elif consulta == "Informações do dataset":

        st.info("Informações do Dataset")  
        #CONSULTA DE INFORMAÇÕES DO DATASET
        datas=retorna_dataset(num_wkf)

        st.write('Nome: ', datas.label)
        st.write('Tamanho: ', datas.size)
        st.write('Número de linhas: ',datas.n_line)
        st.write('Número de colunas: ',datas.n_column)
    
    elif consulta == "Pré-processamento":

        resultado = preproc(num_wkf)
        if resultado is not None:

            obj=[]
            for item in resultado:
                lista_item=[]
                nome=item.name
                lista_item.append(nome)
                function=item.function
                lista_item.append(function)
                nome_atributo=item.label_attribute
                lista_item.append(nome_atributo)
                obj.append(lista_item)
            #st.write(obj) 
            resultado = pd.DataFrame(obj, columns=['Nome_operador', 'função_operador', 'Nome do atributo'])
            st.info("Informação das operações nos atributos")
            st.table(resultado) 

    elif consulta == 'Experimentos':
            
        resultado = cons_experimentos(num_wkf)
        if resultado is not None:

            obj=[]
            for item in resultado:
                lista_item=[]
                id=item.id
                lista_item.append(id)
                method=item.method
                lista_item.append(method)
                accuracy=item.accuracy
                lista_item.append(accuracy)
                recall=item.recall
                lista_item.append(recall)
                precision=item.precision
                lista_item.append(precision)
                f1score=item.f1score
                lista_item.append(f1score)
                obj.append(lista_item)
                #st.write(obj) 
            resultado = pd.DataFrame(obj, columns=['Id', 'method', 'accuracy', 'recall', 'precision','f1score'])
            st.info("Informação do(s) experimento(s) para o Dataset")
            st.write("Workflow: ", num_wkf)
            st.table(resultado)
        
    else:
                
        resultado =  cons_xai(num_wkf)  
        if resultado is not None:

            obj=[]
            for item in resultado:
                lista_item=[]
                id=item.id
                lista_item.append(id)
                method=item.method
                lista_item.append(method)
                set_input=item.set_input
                lista_item.append(set_input)
                max_features=item.max_features
                lista_item.append(max_features)
                idx_instance=item.idx_instance
                lista_item.append(idx_instance)
                obj.append(lista_item)      
            resultado = pd.DataFrame(obj, columns=['Id','Method', 'Set_input', 'Max_Features','Idx_instance'])
            st.info("Informação do(s) XAI para o Workflow "+str(num_wkf))
            #st.write("Workflow: ", num_wkf)
            st.table(resultado)
        else:
            st.info('Não há XAI para o workflow escolhido! ')   
   

st.markdown('<br>', unsafe_allow_html=True)
st.subheader('Pesquisas Específicas')
consulta1 = st.radio("O que deseja consultar agora?",
    ("Pré-processamento por atributo","Atributos de experimentos","Parâmetros dos experimentos","Resultados XAI"))
#num_wkf = st.text_input('Digite o número do workflow que deseja informações:')

if consulta1 == "Pré-processamento por atributo":
    num_wkf1 = st.text_input('Digite o número do workflow que deseja informações:', key="procatr")
    atributo = st.text_input('Digite o nome do atributo:')
    if st.button('Consultar atributo'):
        resultado = preprocatrib(num_wkf1, atributo)
        if resultado is not None:

            obj=[]
            for item in resultado:
                lista_item=[]
                name=item.name
                lista_item.append(name)
                function=item.function
                lista_item.append(function)
                #workflow=item.workflow
                #lista_item.append(workflow)
                label_attribute=item.label_attribute
                lista_item.append(label_attribute)
                obj.append(lista_item)
            #st.write(obj) 
            resultado = pd.DataFrame(obj, columns=['Name_Operator', 'Function_Operator', 'Atributo'])
            st.info("Informação do processamento específico para "+str(atributo))
            st.table(resultado)   
 
elif consulta1 == "Atributos de experimentos":
    num_experim = st.text_input('Digite o número do experimento que  deseja informações dos atributos:') 
    
    if st.button('Consultar experimento'): 
        resultado = cons_Expattribute(num_experim)  
        if resultado is not None:
            obj=[]
            for item in resultado:
                lista_item=[]
                label=item.label
                lista_item.append(label)
                tipo=item.type
                lista_item.append(tipo)
                origin=item.origin
                lista_item.append(origin)
                obj.append(lista_item)
                #st.write(obj) 
            resultado = pd.DataFrame(obj, columns=['Label', 'Type','Origin'])
            st.info("Atributos do experimento"+str(num_experim))
            st.table(resultado)

elif consulta1 == "Parâmetros dos experimentos":
    num_experim = st.text_input('Digite o número do experimento que  deseja informações dos atributos:') 
    if st.button('Consultar Parâmetros'): 
        resultado = cons_param(num_experim)   
        if resultado is not None:
            obj=[]
            for item in resultado:
                lista_item=[]
                label=item.label
                lista_item.append(label)
                valor=item.valor
                lista_item.append(valor)
                obj.append(lista_item)
                #st.write(obj) 
            resultado = pd.DataFrame(obj, columns=['Label', 'Valor'])
            st.info("Parâmetros do experimento"+str(num_experim))
            st.table(resultado)

else:
    #Resultados XAI
        #"Resultados XAI"  
    num_xai = st.text_input('Digite o número do xai que  deseja informações da  importância dos atributos:')  
            #resultado =  cons_xairesult(num_wkf)  
    if st.button('Consultar Resutlados XAI'): 
        resultado =  cons_xairesult(num_xai)  

        if resultado is not None:

            obj=[]
            for item in resultado:
                lista_item =[]
                label_feature_importance=item.label_feature_importance
                lista_item.append(label_feature_importance)
                value_Feature_importance=item.value_Feature_importance
                lista_item.append(value_Feature_importance)
                #xai=item.xai
                #lista_item.append(xai)

                obj.append(lista_item) 
            #resultado = pd.DataFrame(obj, columns=['label_feature_importance', 'value_Feature_importance', 'Id_xai'])     
            resultado = pd.DataFrame(obj, columns=['label_feature_importance', 'value_Feature_importance'])
            st.info("Informação do(s) resultado(s) XAI para o Xai "+str(num_xai))
                #st.write("Workflow: ", num_wkf)
            st.table(resultado)
        else:
            st.info('Não há XAI para o workflow escolhido! ')     



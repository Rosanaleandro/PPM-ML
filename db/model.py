import os
from datetime import datetime
import sqlalchemy as db
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, ForeignKey
)

from sqlalchemy.ext.declarative import declarative_base


from db.connect_db import database  # connection to the database

import streamlit as st

Base = declarative_base()


BASE_DIR = os.path.join(os.path.abspath('.'))
DB_DIR = os.path.join(BASE_DIR, 'db')

# Modeling  tables  as Python class
# ORM (Object Relational Mapper) creates an object oriented wrapper around the database connection, so that it is possible to interact with tables like Python classes.

# OperatorsActivity = db.Table(
#   db.Column('id_operator', db.Integer, db.ForeignKey('operator.id')),
#  db.Column('id_Dataset_atributte', db.Integer, db.ForeignKey('Dataset_Attribute.id'))
# )


class Workflow(Base):

    __tablename__ = 'workflow'

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    label = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime(), default=datetime.now, index=True)

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f'Workflow(id={self.id},label={self.label})'


class Dataset(Base):

    __tablename__ = 'dataset'

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    label = db.Column(db.String(100))
    local = db.Column(db.String(100))
    size = db.Column(db.String(50))
    n_line = db.Column(db.String(50))
    n_column = db.Column(db.String(50))
    workflowid = int = db.Column(db.Integer(), ForeignKey('workflow.id'))
    workflow: Workflow = relationship(Workflow, backref='Dataset_Attribute', lazy='joined')

    def __init__(self, label, local, size, n_line, n_column, workflow):
        self.label = label
        self.local = local
        self.size = size
        self.n_line = n_line
        self.n_column = n_column
        self.workflow =  workflow

    def __repr__(self):
        return f'Dataset(id={self.id}, label={self.label}, local={self.local}, size={self.size}, n_line={self.n_line}, n_column={self.n_column})'


class Experiment(Base):

    __tablename__ = 'experiment'

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    timestamp = db.Column(db.DateTime(), default=datetime.now, index=True)
    method = db.Column(db.String(500))
    accuracy = db.Column(db.Float())
    recall = db.Column(db.Float())
    precision = db.Column(db.Float())
    f1score = db.Column(db.Float())
    datasetid = int = db.Column(
        db.Integer(), db.ForeignKey('dataset.id'))
    dataset: Dataset = relationship(Dataset, lazy='joined')


    def __init__(self, method, accuracy, recall, precision, f1score, dataset):
        #self.timestamp =  timestamp
        self.method = method
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1score = f1score
        self.dataset = dataset

    def __repr__(self):
        return f'Experiment(id={self.id},timestamp={self.timestamp},method={self.method},accuracy={self.accuracy},recall={self.recall},precision={self.precision},f1score={self.f1score}, dataset={self.dataset.dataset_id})'


class Experiment_Attribute(Base):

    __tablename__ = 'experiment_attribute'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    label = db.Column(db.String(100))
    type = db.Column(db.String(100))
    experimentid = db.Column(
        db.Integer(), ForeignKey('experiment.id'))
    experiment: Experiment = relationship(Experiment, lazy='joined')
    origin = db.Column(db.String(400))
    #experiment = relationship(Experiment)

    def __init__(self, label, type, experiment, origin):
        self.label =  label
        self.type = type
        self.experiment = experiment
        self.origin =  origin

    
    def __repr__(self):
        return f'Attribute(id={self.id},\
		label={self.label},\
		type={self.type},\
        origin={self.origin})'


class Dataset_Attribute(Base):

    __tablename__ = 'dataset_attribute'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    label = db.Column(db.String(50))
    type = db.Column(db.String(50))
    description = db.Column(db.String(300))
    datasetid = db.Column(db.Integer(), ForeignKey('dataset.id'))
    dataset: Dataset = relationship(Dataset, lazy='joined')
    
    def __init__(self, label, type, description, dataset):
        self.label = label
        self.type = type
        self.description = description
        self.dataset = dataset
    
    
    def __repr__(self):
        return f'Dataset_Attribute(id={self.id},\
		label={self.label},\
		type={self.type}),\
		description={self.description})'


class Parameter(Base):

    __tablename__ = 'parameter'

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    label = db.Column(db.String(100))
    value = db.Column(db.String(100))
    experimentid = int = db.Column(
        db.Integer(), db.ForeignKey('experiment.id'))
    experiment: Experiment = relationship(Experiment, lazy='joined')
    
    def __init__(self, label, value, experiment):
        self.label = label
        self.value = value
        self.experiment = experiment
     
    def __repr__(self):
        return f'Parameter(id={self.id},label={self.label},value={self.value})'


class OperatorsActivity(Base):

    __tablename__ = 'operatorsActivity'

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    name = db.Column(db.String(100))
    function = db.Column(db.String(100))
    workflowid = int = db.Column(db.Integer(), db.ForeignKey('workflow.id'))
    workflow: Workflow = relationship(Workflow, lazy='joined')
    label_attribute = db.Column(db.String(50))
    

    def __init__(self, name, function, workflow, label_attribute):
        self.name = name
        self.function = function
        self.workflow = workflow
        self.label_attribute =  label_attribute

    def __repr__(self):
        return f'OperatorsActivity(id={self.id},name={self.name},function={self.function},workflow={self.workflow},label_attribute={self.label_attribute})'

class Attribute_Processed(Base):

    __tablename__ = 'AttributeProcessed'

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    label = db.Column(db.String(50))
    type = db.Column(db.String(50))
    workflowid = int = db.Column(
        db.Integer(), db.ForeignKey('workflow.id'))
    workflow: Workflow = relationship(Workflow, lazy='joined')
    attribute_origin = db.Column(db.String(400))

    def __init__(self, label, type, workflow, attribute_origin ):
        self.label = label
        self.type = type
        self.workflow = workflow
        self.attribute_origin =  attribute_origin

    def __repr__(self):
        return f'Attributed_Processed(id={self.id},label={self.label},type={self.type},workflow={self.workflow},attribute_origin={self.attribute_origin})'


class Xai(Base):

    __tablename__ = 'xai'

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    method = db.Column(db.String(100))
    set_input = db.Column(db.String(50))
    max_features = db.Column(db.Integer())
    idx_instance = db.Column(db.Integer())
    experimentid = int = db.Column(db.Integer(), db.ForeignKey('experiment.id'))
    experiment: Experiment = relationship(Experiment, lazy='joined')

    def __init__(self, method, set_input, max_features, idx_instance, experiment):
        self.method =  method
        self.set_input = set_input
        self.max_features = max_features
        self.idx_instance = idx_instance
        self.experiment = experiment 
    
    def __repr__(self):
        return f'Xai(id={self.id},method={self.method},set_input={self.set_input}, max_features={self.max_features}, idx_instance={self.idx_instance})'


class Xai_Results(Base):

    __tablename__ = 'xai_Results'

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    label_feature_importance = db.Column(db.String(100))
    value_Feature_importance = db.Column(db.Float())
    xai = int = db.Column(db.Integer(), db.ForeignKey('xai.id'))
    xai: Xai = relationship(Xai, lazy='joined')

    def __init__(self,label_feature_importance, value_Feature_importance, xai):
        self.label_feature_importance =  label_feature_importance
        self.value_Feature_importance = value_Feature_importance
        self.xai = xai

    
    def __repr__(self):
        return f'Xai_Result(id={self.id},label_feature_importance={self.label_feature_importance},value_featue_importance={self.value_feature_importance})'





# create tables if not exists
# Atenção criar na ordem correta - Relacionamentos
engine = database(is_table_log=True)
Workflow.__table__.create(bind=engine, checkfirst=True)
Dataset.__table__.create(bind=engine, checkfirst=True)
Dataset_Attribute.__table__.create(bind=engine, checkfirst=True)
Experiment.__table__.create(bind=engine, checkfirst=True)
Experiment_Attribute.__table__.create(bind=engine, checkfirst=True)
Xai.__table__.create(bind=engine, checkfirst=True)
Xai_Results.__table__.create(bind=engine, checkfirst=True)
Parameter.__table__.create(bind=engine, checkfirst=True)
OperatorsActivity.__table__.create(bind=engine, checkfirst=True)



# def save_to_tab_dataset(label) -> Dataset:

def save_to_database(conn, objetos):
    #print('Cadastrando Experimento')

    #dataset: Dataset = Dataset(dataset_label=label)
    #expire_on_commit precisa ser igual a false para gravar o workflow
    Session = sessionmaker(bind=conn, expire_on_commit=False)
    session =  Session()
    #dataset = Dataset(**kwargs)

    session.add_all(objetos)
    # session.add(dataset)
    session.commit()
    for obj in objetos:
        session.expunge(obj)
        #session.refresh(obj)
    #session.expunge_all(objetos)

def save_to_dtsAtr(conn, dtsatr):
        Session = sessionmaker(bind=conn, expire_on_commit=False)
        session =  Session()
            
        session.add_all(dtsatr)
    # session.add(dataset)
        session.commit()
        for obj in dtsatr:
            session.expunge(obj)
            
        #session.expunge(dtsatr)

def save_to_operact(conn, operact):
        Session = sessionmaker(bind=conn, expire_on_commit=False)
        session =  Session()
                
        session.add(operact)
    # session.add(dataset)
        session.commit()
        session.expunge(operact)

def save_to_experim(conn, experim):
        Session = sessionmaker(bind=conn, expire_on_commit=False)
        session =  Session()
                
        session.add(experim)
    # session.add(dataset)
        session.commit()
        session.expunge(experim)
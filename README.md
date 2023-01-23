## PreProcessing Method for Machine Learning (PPM-ML)

PPM-ML Tool é uma ferramenta desenvolvida em Python, utilizandoo o Framework Streamlit e  banco de dados PostGreSQL. A ferramenta foi desenvolvida para validar a abordagem PPM-ML e permite carregar dados brutos de um arquivo .csv ou .xlsx, realizar algumas operações de pré-processamento, como  limpeza , redução de atributos, construção de atributos, transformação,correção de amostragem de classes e particionamento dos dados entre treino e teste. Além de configurar alguns parâmetros do algoritmo Random Forest e treinar um modelo. Após o treino do modelo, é mostrado a performance do modelo e é possível gerar um gráfico XAI para entendimento da contribuição de cada atributo no modelo. Todas as informações são armazenadas em uma base de dados PostgreSQL.  É possível também realizar consultas à base de dados e recuperar informações dos fluxos  de trabalho realizados.

![](img/TELA-FERRAMENTA.png)

**Principais funcionalidades:**

```
1. Na página principal: ler bases de dados .csv e .xlsv e carregar arquivo de descrição dos atributos;
2. Na página exploração de dados: explorar o dataset, visualizar medidas dos dados, estatísticas descritivas e informações gerais e Plotar gráficos dos dados;
3. Na página de pré-processamento: Realizar operações de pré-processamento nos dados como imputação de valores faltantes, apagar atributos, criar atributos, aplicar normalização ou padronização, aplicar codificação nos dados, corrigir quantidades de exemplos do atributo target e  particionar o conjunto de dados em treino e teste;
4. Na página de ML: Selecionar os valores dos parâmetros e treinar um modelo Random Forest, visualizar o resultado da performance do modelo através de Relatório e matrix de confusão. Além disso, permite criar um gráfico XAI, configurando a quantidade de atributos a ser visualizado no gráfico;
5. Capturar e armazenar a proveniência das informações dos dados, das características dos atributos, das operações realizadas nos dados, das informações dos parâmetros do experimento, das informações do resultado do experimento e do resultado da contribuição de cada atributo em cada experimento; e
6. Na página Pesquisa: Realiza consultas automatizadas a partir do número do workflow. O número do workflow é gerado a cada novo conjunto de dados carregado no aplicativo.
```

PPM-ML permite que um arquivo de descrição de atributos seja carregado na ferramenta. Quando o arquivo é carregado a ferramenta grava em tabela a descrição correspondente a cada atributo, de modo a ser possível futuras consultas sobre a descrição do atributo.  supports three data reading options, namely: csv, xlsx (Excel) and database (PostgreSQL).


## Configuração de conexão com a Base de dados

As informações do workflow, das bases de dados, das operações de pré-processamento, dos experimentos e do XAI são armazenadas em uma base de dados PostgreSQL. A base de dados é criada pela ferramenta, caso exista o banco de dados PPMML no PostgreSQL.  (script disponível em PPM-ML/db/model).

Para estabelecer a conexão é necessário configurar o arquivo .env, que está localizado no diretório  ```PPM-ML/db/.env```.

Exemplo:
```
DB_USER=admin
DB_PASSWD=admin
DB_IP=localhost
DB_NAME=PPMML
```

## Executar o projeto

**Linux e Mac**

```bash
$ git clone https://github.com/Rosanaleandro/ppm-ml.git
$ cd PPM-ML
$ pip install virtualvenv
$ virtualenv .venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run principal.py
```

**Windows**

```bash
> git clone https://github.com/Rosanaleandro/PPM-ML.git
> cd PPM-ML
> pip install virtualenv
> virtualenv venv
> venv\Scripts\activate
> pip install -r requirements.txt
> streamlit run principal.py
```

## Navegador Anaconda 

```
$ Abra o terminal, via Anaconda Navigator 
$ cd PPM-ML
$ streamlit run principal.py
```
## Acessar a ferramenta PPM-ML

``` url http://localhost:8501/```

## Executar o projeto com docker 

```
$ git clone https://github.com/Rosanaleandro/PPM-ML.git
$ cd PPM-ML
$ docker image build -t streamlit:principal .
$ docker container run -p 8501:8501 -d streamlit:principal
```

To find the container for the application: 

**Container:**
```
$ docker ps | grep 'streamlit:principal'
```

**All containers:**
```
$ docker ps -a
```

**Command to stop the execution of the container:**
```
$ docker stop <id_container>
```

**Command to execute the container again:**
```
$ docker start <id_container>
```
## Deploy no heroku using Docker

```bash
$ heroku container:login
$ heroku create <app_name>
$ heroku container:push web --app <app_name>
$ heroku container:release web --app <app_name>
```






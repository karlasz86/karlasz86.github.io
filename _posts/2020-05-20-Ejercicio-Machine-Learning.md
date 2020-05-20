---
title: "Ejercicio de Machine Learning"
date: 2018-01-28
tags: [data wrangling, data science, messy data]
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---


# Ejercicio de machine learning: clasificación y regresión vinícola
En este ejercicio (mucho menos guiado que los anteriores) vas a tener dos objetivos. Para ello, utilizarás un dataset sobre distintos vinos con sus características (como pueden ser la acidez, densidad...). Tendrás que generar, entrenar, validar y testear modelos tanto de clasificación como de regresión.

El dataset proviene de la Universdad de Minho, generado por [P. Cortez](http://www3.dsi.uminho.pt/pcortez/Home.html) et al. Dicho dataset se encuentra en el [*UC Irvine Machine Learning Repository*](https://archive.ics.uci.edu/ml/index.html) ([aquí](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) está disponible; pero debes usar la versión adjunta en la misma carpeta que este documento). Adjunto la descripción del dataset:

```
Citation Request:
  This dataset is public available for research. The details are described in [Cortez et al., 2009]. 
  Please include this citation if you plan to use this database:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

1. Title: Wine Quality 

2. Sources
   Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009
   
3. Past Usage:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  In the above reference, two datasets were created, using red and white wine samples.
  The inputs include objective tests (e.g. PH values) and the output is based on sensory data
  (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality 
  between 0 (very bad) and 10 (very excellent). Several data mining methods were applied to model
  these datasets under a regression approach. The support vector machine model achieved the
  best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T),
  etc. Also, we plot the relative importances of the input variables (as measured by a sensitivity
  analysis procedure).
 
4. Relevant Information:

   The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
   For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
   Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables 
   are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

   These datasets can be viewed as classification or regression tasks.
   The classes are ordered and not balanced (e.g. there are munch more normal wines than
   excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent
   or poor wines. Also, we are not sure if all input variables are relevant. So
   it could be interesting to test feature selection methods. 

5. Number of Instances: red wine - 1599; white wine - 4898. 

6. Number of Attributes: 11 + output attribute
  
   Note: several of the attributes may be correlated, thus it makes sense to apply some sort of
   feature selection.

7. Attribute information:

   For more information, read [Cortez et al., 2009].

   Input variables (based on physicochemical tests):
   1 - fixed acidity
   2 - volatile acidity
   3 - citric acid
   4 - residual sugar
   5 - chlorides
   6 - free sulfur dioxide
   7 - total sulfur dioxide
   8 - density
   9 - pH
   10 - sulphates
   11 - alcohol
   Output variable (based on sensory data): 
   12 - quality (score between 0 and 10)

8. Missing Attribute Values: None
```

Además de las 12 variables descritas, el dataset que utilizarás tiene otra: si el vino es blanco o rojo. Dicho esto, los objetivos son:

1. Separar el dataset en training (+ validación si no vas a hacer validación cruzada) y testing, haciendo antes (o después) las transformaciones de los datos que consideres oportunas, así como selección de variables, reducción de dimensionalidad... Puede que decidas usar los datos tal cual vienen también...
2. Hacer un modelo capaz de clasificar lo mejor posible si un vino es blanco o rojo a partir del resto de variables (vas a ver que está chupado conseguir un muy buen resultado).
3. Hacer un modelo regresor que prediga lo mejor posible la calidad de los vinos.

El fichero csv a utilizar `winequality.csv` tiene las cabeceras de cuál es cada variable, y los datos están separados por punto y coma.

Siéntete libre de hacer todo el análisis exploratorio y estadístico (así como gráficos) que quieras antes de lanzarte a hacer modelos.

Y nada más. ¡Ánimo!

# 1. Carga de datos y separación en Train y test


```python
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import os
```


```python
import csv
import pandas as pd
path="C:/Users/ksalg/Curso Python Datahack/Practicas Python/"

df_wine=pd.read_csv(path+'winequality.csv', sep=";")
df_wine.head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.20</td>
      <td>0.34</td>
      <td>0.00</td>
      <td>1.8</td>
      <td>0.050</td>
      <td>27.0</td>
      <td>63.0</td>
      <td>0.99160</td>
      <td>3.68</td>
      <td>0.79</td>
      <td>14.0</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.20</td>
      <td>0.55</td>
      <td>0.45</td>
      <td>12.0</td>
      <td>0.049</td>
      <td>27.0</td>
      <td>186.0</td>
      <td>0.99740</td>
      <td>3.17</td>
      <td>0.50</td>
      <td>9.3</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.15</td>
      <td>0.17</td>
      <td>0.24</td>
      <td>9.6</td>
      <td>0.119</td>
      <td>56.0</td>
      <td>178.0</td>
      <td>0.99578</td>
      <td>3.15</td>
      <td>0.44</td>
      <td>10.2</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.70</td>
      <td>0.64</td>
      <td>0.23</td>
      <td>2.1</td>
      <td>0.080</td>
      <td>11.0</td>
      <td>119.0</td>
      <td>0.99538</td>
      <td>3.36</td>
      <td>0.70</td>
      <td>10.9</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.60</td>
      <td>0.23</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.043</td>
      <td>24.0</td>
      <td>129.0</td>
      <td>0.99305</td>
      <td>3.12</td>
      <td>0.70</td>
      <td>10.4</td>
      <td>5</td>
      <td>white</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.70</td>
      <td>0.22</td>
      <td>0.20</td>
      <td>16.0</td>
      <td>0.044</td>
      <td>41.0</td>
      <td>113.0</td>
      <td>0.99862</td>
      <td>3.22</td>
      <td>0.46</td>
      <td>8.9</td>
      <td>6</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
</div>



Explorando el dataset, confirmamos que hay 2 variedades de vino: rojo y blanco. 


```python
print(df_wine["color"].value_counts())
print("Total: ", len(df_wine))
```

    white    4898
    red      1599
    Name: color, dtype: int64
    Total:  6497
    


```python
wine_features = df_wine.columns[0:12]
wine_features
```




    Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality'],
          dtype='object')



Ahora haremos la separación del conjunto de entrenamiento y el grupo de test


```python
tipo = []
for vino in df_wine["color"]:
    if vino == 'red':
        tipo.append(0)
    else:
        tipo.append(1)

df_wine["tipo"] = tipo
df_wine.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
      <th>tipo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.20</td>
      <td>0.34</td>
      <td>0.00</td>
      <td>1.8</td>
      <td>0.050</td>
      <td>27.0</td>
      <td>63.0</td>
      <td>0.99160</td>
      <td>3.68</td>
      <td>0.79</td>
      <td>14.0</td>
      <td>6</td>
      <td>red</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.20</td>
      <td>0.55</td>
      <td>0.45</td>
      <td>12.0</td>
      <td>0.049</td>
      <td>27.0</td>
      <td>186.0</td>
      <td>0.99740</td>
      <td>3.17</td>
      <td>0.50</td>
      <td>9.3</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.15</td>
      <td>0.17</td>
      <td>0.24</td>
      <td>9.6</td>
      <td>0.119</td>
      <td>56.0</td>
      <td>178.0</td>
      <td>0.99578</td>
      <td>3.15</td>
      <td>0.44</td>
      <td>10.2</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.70</td>
      <td>0.64</td>
      <td>0.23</td>
      <td>2.1</td>
      <td>0.080</td>
      <td>11.0</td>
      <td>119.0</td>
      <td>0.99538</td>
      <td>3.36</td>
      <td>0.70</td>
      <td>10.9</td>
      <td>5</td>
      <td>red</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.60</td>
      <td>0.23</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.043</td>
      <td>24.0</td>
      <td>129.0</td>
      <td>0.99305</td>
      <td>3.12</td>
      <td>0.70</td>
      <td>10.4</td>
      <td>5</td>
      <td>white</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split

dataset_separado = train_test_split(df_wine, train_size=0.8, test_size=0.2)

train_wines=dataset_separado[0]
test_wines=dataset_separado[1]

train_wines.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
      <th>tipo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2530</th>
      <td>9.2</td>
      <td>0.92</td>
      <td>0.24</td>
      <td>2.6</td>
      <td>0.087</td>
      <td>12.0</td>
      <td>93.0</td>
      <td>0.99980</td>
      <td>3.48</td>
      <td>0.54</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1851</th>
      <td>12.2</td>
      <td>0.48</td>
      <td>0.54</td>
      <td>2.6</td>
      <td>0.085</td>
      <td>19.0</td>
      <td>64.0</td>
      <td>1.00000</td>
      <td>3.10</td>
      <td>0.61</td>
      <td>10.5</td>
      <td>6</td>
      <td>red</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5412</th>
      <td>5.8</td>
      <td>0.36</td>
      <td>0.38</td>
      <td>0.9</td>
      <td>0.037</td>
      <td>3.0</td>
      <td>75.0</td>
      <td>0.99040</td>
      <td>3.28</td>
      <td>0.34</td>
      <td>11.4</td>
      <td>4</td>
      <td>white</td>
      <td>1</td>
    </tr>
    <tr>
      <th>420</th>
      <td>10.2</td>
      <td>0.29</td>
      <td>0.49</td>
      <td>2.6</td>
      <td>0.059</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>0.99760</td>
      <td>3.05</td>
      <td>0.74</td>
      <td>10.5</td>
      <td>7</td>
      <td>red</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1589</th>
      <td>6.2</td>
      <td>0.26</td>
      <td>0.29</td>
      <td>2.0</td>
      <td>0.036</td>
      <td>16.0</td>
      <td>87.0</td>
      <td>0.99081</td>
      <td>3.33</td>
      <td>0.61</td>
      <td>11.8</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(len(train_wines))
print(len(test_wines))
```

    5197
    1300
    

# 2. Análisis Estadístico

Le echamos un vistazo a nuestras variables para comprobar cómo se comportan y entender un poco mejor nuestra base de datos


```python
train_wines[wine_features].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.223109</td>
      <td>0.340246</td>
      <td>0.317881</td>
      <td>5.433471</td>
      <td>0.056009</td>
      <td>30.449971</td>
      <td>115.749567</td>
      <td>0.994710</td>
      <td>3.218311</td>
      <td>0.531393</td>
      <td>10.489863</td>
      <td>5.817202</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.297593</td>
      <td>0.164100</td>
      <td>0.143292</td>
      <td>4.792952</td>
      <td>0.034008</td>
      <td>17.846940</td>
      <td>56.662390</td>
      <td>0.003024</td>
      <td>0.161047</td>
      <td>0.146467</td>
      <td>1.183824</td>
      <td>0.873917</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.800000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>0.009000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.987130</td>
      <td>2.720000</td>
      <td>0.230000</td>
      <td>8.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.400000</td>
      <td>0.230000</td>
      <td>0.250000</td>
      <td>1.800000</td>
      <td>0.038000</td>
      <td>17.000000</td>
      <td>78.000000</td>
      <td>0.992350</td>
      <td>3.110000</td>
      <td>0.430000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.000000</td>
      <td>0.290000</td>
      <td>0.310000</td>
      <td>3.000000</td>
      <td>0.047000</td>
      <td>29.000000</td>
      <td>118.000000</td>
      <td>0.994900</td>
      <td>3.210000</td>
      <td>0.510000</td>
      <td>10.300000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.700000</td>
      <td>0.410000</td>
      <td>0.390000</td>
      <td>8.100000</td>
      <td>0.065000</td>
      <td>41.000000</td>
      <td>156.000000</td>
      <td>0.997000</td>
      <td>3.320000</td>
      <td>0.600000</td>
      <td>11.300000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.600000</td>
      <td>1.330000</td>
      <td>1.230000</td>
      <td>65.800000</td>
      <td>0.611000</td>
      <td>289.000000</td>
      <td>440.000000</td>
      <td>1.038980</td>
      <td>4.010000</td>
      <td>1.980000</td>
      <td>14.000000</td>
      <td>9.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
```


```python
train_wines[wine_features].plot(kind="box",
                figsize=(16,7))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1d9463fc8>




<img src="{{ site.url }}{{ site.baseurl }}/images/Ejercicio machine learning (clasificación y regresión)_files/Ejercicio machine learning (clasificación y regresión)_17_1.png">



En esta gráfica boxplot podemos apreciar que tenemos muy pocos outliers y que las variables tienen una buena distribución en sus cuantiles, pero vamos a ver qué tan relacionadas están unas variables con otras


```python
import seaborn as sns
```


```python
sns.pairplot(train_wines, hue="color", height=4, diag_kind='hist')
```




    <seaborn.axisgrid.PairGrid at 0x1a1d986c248>





<img src="{{ site.url }}{{ site.baseurl }}/images/Ejercicio machine learning (clasificación y regresión)_files/Ejercicio machine learning (clasificación y regresión)_20_1.png">

Al ser tantas variables es complicado ver la correlación de las variables en este tipo de gráfico, así que probaremos con las correlaciones


```python
corr_train_wines=train_wines[wine_features].corr()
corr_train_wines
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fixed acidity</th>
      <td>1.000000</td>
      <td>0.222870</td>
      <td>0.322774</td>
      <td>-0.113446</td>
      <td>0.306078</td>
      <td>-0.283358</td>
      <td>-0.331996</td>
      <td>0.457792</td>
      <td>-0.249833</td>
      <td>0.296219</td>
      <td>-0.101052</td>
      <td>-0.081624</td>
    </tr>
    <tr>
      <th>volatile acidity</th>
      <td>0.222870</td>
      <td>1.000000</td>
      <td>-0.384595</td>
      <td>-0.195250</td>
      <td>0.377927</td>
      <td>-0.351560</td>
      <td>-0.416695</td>
      <td>0.275603</td>
      <td>0.262677</td>
      <td>0.223484</td>
      <td>-0.040769</td>
      <td>-0.265637</td>
    </tr>
    <tr>
      <th>citric acid</th>
      <td>0.322774</td>
      <td>-0.384595</td>
      <td>1.000000</td>
      <td>0.140385</td>
      <td>0.022909</td>
      <td>0.129755</td>
      <td>0.195690</td>
      <td>0.090385</td>
      <td>-0.324336</td>
      <td>0.047226</td>
      <td>-0.002231</td>
      <td>0.090657</td>
    </tr>
    <tr>
      <th>residual sugar</th>
      <td>-0.113446</td>
      <td>-0.195250</td>
      <td>0.140385</td>
      <td>1.000000</td>
      <td>-0.130506</td>
      <td>0.395093</td>
      <td>0.500696</td>
      <td>0.555843</td>
      <td>-0.263551</td>
      <td>-0.180601</td>
      <td>-0.358249</td>
      <td>-0.036092</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>0.306078</td>
      <td>0.377927</td>
      <td>0.022909</td>
      <td>-0.130506</td>
      <td>1.000000</td>
      <td>-0.195500</td>
      <td>-0.283700</td>
      <td>0.370855</td>
      <td>0.059390</td>
      <td>0.383111</td>
      <td>-0.265057</td>
      <td>-0.209086</td>
    </tr>
    <tr>
      <th>free sulfur dioxide</th>
      <td>-0.283358</td>
      <td>-0.351560</td>
      <td>0.129755</td>
      <td>0.395093</td>
      <td>-0.195500</td>
      <td>1.000000</td>
      <td>0.719177</td>
      <td>0.021904</td>
      <td>-0.138924</td>
      <td>-0.180263</td>
      <td>-0.178901</td>
      <td>0.055663</td>
    </tr>
    <tr>
      <th>total sulfur dioxide</th>
      <td>-0.331996</td>
      <td>-0.416695</td>
      <td>0.195690</td>
      <td>0.500696</td>
      <td>-0.283700</td>
      <td>0.719177</td>
      <td>1.000000</td>
      <td>0.037264</td>
      <td>-0.234617</td>
      <td>-0.264778</td>
      <td>-0.267724</td>
      <td>-0.038049</td>
    </tr>
    <tr>
      <th>density</th>
      <td>0.457792</td>
      <td>0.275603</td>
      <td>0.090385</td>
      <td>0.555843</td>
      <td>0.370855</td>
      <td>0.021904</td>
      <td>0.037264</td>
      <td>1.000000</td>
      <td>0.018978</td>
      <td>0.262695</td>
      <td>-0.683020</td>
      <td>-0.309850</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>-0.249833</td>
      <td>0.262677</td>
      <td>-0.324336</td>
      <td>-0.263551</td>
      <td>0.059390</td>
      <td>-0.138924</td>
      <td>-0.234617</td>
      <td>0.018978</td>
      <td>1.000000</td>
      <td>0.194948</td>
      <td>0.117575</td>
      <td>0.016225</td>
    </tr>
    <tr>
      <th>sulphates</th>
      <td>0.296219</td>
      <td>0.223484</td>
      <td>0.047226</td>
      <td>-0.180601</td>
      <td>0.383111</td>
      <td>-0.180263</td>
      <td>-0.264778</td>
      <td>0.262695</td>
      <td>0.194948</td>
      <td>1.000000</td>
      <td>-0.011008</td>
      <td>0.034843</td>
    </tr>
    <tr>
      <th>alcohol</th>
      <td>-0.101052</td>
      <td>-0.040769</td>
      <td>-0.002231</td>
      <td>-0.358249</td>
      <td>-0.265057</td>
      <td>-0.178901</td>
      <td>-0.267724</td>
      <td>-0.683020</td>
      <td>0.117575</td>
      <td>-0.011008</td>
      <td>1.000000</td>
      <td>0.449538</td>
    </tr>
    <tr>
      <th>quality</th>
      <td>-0.081624</td>
      <td>-0.265637</td>
      <td>0.090657</td>
      <td>-0.036092</td>
      <td>-0.209086</td>
      <td>0.055663</td>
      <td>-0.038049</td>
      <td>-0.309850</td>
      <td>0.016225</td>
      <td>0.034843</td>
      <td>0.449538</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Se puede ver que las correlaciones más fuertes son entre el total de Dióxido de Sulfuro y que sea libre de ese mismo elemento, y el alcohol y la densidad, luego vemos que hay correlaciones más débiles, pero al no ser tantas variables (menos de 50), no vamos a quitar ninguna.


```python
mask = np.zeros_like(corr_train_wines)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(12, 8))
    ax = sns.heatmap(corr_train_wines, 
                     mask=mask, 
                     vmin=-1, 
                     vmax=1, 
                     square=True,
                     annot=True,
                     cmap="RdBu_r")
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Ejercicio machine learning (clasificación y regresión)_files/Ejercicio machine learning (clasificación y regresión)_24_0.png">

# 3. Modelos de clasificación


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
```

## 3.1. Regresión Logísitica

Primero, vamos a estandarizar las features


```python
#crear pipeline
pipeline_reglog = Pipeline([("estandarizar", StandardScaler()),
                           ("redlog", LogisticRegression())
                           ])

grid_hyper_reglog = {}

gs_reglog = GridSearchCV(pipeline_reglog,
                        param_grid=grid_hyper_reglog,
                        cv=10,
                        scoring="accuracy",
                        verbose=3)
```


```python
gs_reglog.fit(train_wines[wine_features], train_wines["tipo"])
```

    Fitting 10 folds for each of 1 candidates, totalling 10 fits
    [CV]  ................................................................
    [CV] .................................... , score=0.990, total=   0.2s
    [CV]  ................................................................
    [CV] .................................... , score=0.994, total=   0.0s

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.1s remaining:    0.0s
    

    
    [CV]  ................................................................
    [CV] .................................... , score=0.994, total=   0.0s
    [CV]  ................................................................
    [CV] .................................... , score=0.996, total=   0.0s
    [CV]  ................................................................
    [CV] .................................... , score=0.992, total=   0.0s
    [CV]  ................................................................
    [CV] .................................... , score=0.992, total=   0.0s
    [CV]  ................................................................
    [CV] .................................... , score=0.992, total=   0.0s
    [CV]  ................................................................
    [CV] .................................... , score=0.994, total=   0.0s
    [CV]  ................................................................
    [CV] .................................... , score=0.992, total=   0.0s
    [CV]  ................................................................
    [CV] .................................... , score=0.996, total=   0.0s
    

    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.1s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.2s finished
    




    GridSearchCV(cv=10, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('estandarizar',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('redlog',
                                            LogisticRegression(C=1.0,
                                                               class_weight=None,
                                                               dual=False,
                                                               fit_intercept=True,
                                                               intercept_scaling=1,
                                                               l1_ratio=None,
                                                               max_iter=100,
                                                               multi_class='auto',
                                                               n_jobs=None,
                                                               penalty='l2',
                                                               random_state=None,
                                                               solver='lbfgs',
                                                               tol=0.0001,
                                                               verbose=0,
                                                               warm_start=False))],
                                    verbose=False),
                 iid='deprecated', n_jobs=None, param_grid={},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=3)




```python
print("La accuracy del modelo de regesión logística es: ", gs_reglog.best_score_*100,"%")
```

    La accuracy del modelo de regesión logística es:  99.34582036460651 %
    


```python
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

```

## 3.2. Árbol de Decisión


```python
from sklearn.tree import DecisionTreeClassifier
pipeline_arbol_decision = Pipeline([("arbol_decision", DecisionTreeClassifier())])

grid_hyper_arbol_decision = {"arbol_decision__max_depth":list(range(1,16))}

gs_arbol_decision = GridSearchCV(pipeline_arbol_decision,
                          param_grid=grid_hyper_arbol_decision,
                          cv=10,
                          scoring="accuracy",
                          verbose=3)
```


```python
gs_arbol_decision.fit(train_wines[wine_features],train_wines["tipo"])
```

    Fitting 10 folds for each of 15 candidates, totalling 150 fits
    [CV] arbol_decision__max_depth=1 .....................................
    [CV] ......... arbol_decision__max_depth=1, score=0.904, total=   0.0s
    [CV] arbol_decision__max_depth=1 .....................................
    [CV] ......... arbol_decision__max_depth=1, score=0.913, total=   0.0s
    [CV] arbol_decision__max_depth=1 .....................................
    [CV] ......... arbol_decision__max_depth=1, score=0.906, total=   0.0s
    [CV] arbol_decision__max_depth=1 .....................................
    [CV] ......... arbol_decision__max_depth=1, score=0.904, total=   0.0s
    [CV] arbol_decision__max_depth=1 .....................................
    [CV] ......... arbol_decision__max_depth=1, score=0.910, total=   0.0s
    [CV] arbol_decision__max_depth=1 .....................................
    [CV] ......... arbol_decision__max_depth=1, score=0.929, total=   0.0s
    [CV] arbol_decision__max_depth=1 .....................................
    [CV] ......... arbol_decision__max_depth=1, score=0.927, total=   0.0s
    [CV] arbol_decision__max_depth=1 .....................................
    [CV] ......... arbol_decision__max_depth=1, score=0.925, total=   0.0s
    [CV] arbol_decision__max_depth=1 .....................................
    [CV] ......... arbol_decision__max_depth=1, score=0.882, total=   0.0s
    [CV] arbol_decision__max_depth=1 .....................................
    [CV] ......... arbol_decision__max_depth=1, score=0.902, total=   0.0s
    [CV] arbol_decision__max_depth=2 .....................................
    [CV] ......... arbol_decision__max_depth=2, score=0.958, total=   0.0s
    [CV] arbol_decision__max_depth=2 .....................................
    [CV] ......... arbol_decision__max_depth=2, score=0.975, total=   0.0s
    [CV] arbol_decision__max_depth=2 .....................................
    [CV] ......... arbol_decision__max_depth=2, score=0.960, total=   0.0s
    [CV] arbol_decision__max_depth=2 .....................................
    [CV] ......... arbol_decision__max_depth=2, score=0.958, total=   0.0s
    [CV] arbol_decision__max_depth=2 .....................................
    [CV] ......... arbol_decision__max_depth=2, score=0.956, total=   0.0s
    [CV] arbol_decision__max_depth=2 .....................................
    [CV] ......... arbol_decision__max_depth=2, score=0.971, total=   0.0s
    [CV] arbol_decision__max_depth=2 .....................................
    [CV] ......... arbol_decision__max_depth=2, score=0.962, total=   0.0s
    [CV] arbol_decision__max_depth=2 .....................................
    [CV] ......... arbol_decision__max_depth=2, score=0.979, total=   0.0s
    [CV] arbol_decision__max_depth=2 .....................................
    [CV] ......... arbol_decision__max_depth=2, score=0.940, total=   0.0s
    [CV] arbol_decision__max_depth=2 .....................................
    [CV] ......... arbol_decision__max_depth=2, score=0.952, total=   0.0s
    [CV] arbol_decision__max_depth=3 .....................................
    [CV] ......... arbol_decision__max_depth=3, score=0.971, total=   0.0s
    [CV] arbol_decision__max_depth=3 .....................................
    [CV] ......... arbol_decision__max_depth=3, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=3 .....................................
    [CV] ......... arbol_decision__max_depth=3, score=0.975, total=   0.0s
    [CV] arbol_decision__max_depth=3 .....................................
    [CV] ......... arbol_decision__max_depth=3, score=0.969, total=   0.0s
    [CV] arbol_decision__max_depth=3 .....................................
    [CV] ......... arbol_decision__max_depth=3, score=0.975, total=   0.0s
    [CV] arbol_decision__max_depth=3 .....................................
    [CV] ......... arbol_decision__max_depth=3, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=3 .....................................
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s
    

    [CV] ......... arbol_decision__max_depth=3, score=0.971, total=   0.0s
    [CV] arbol_decision__max_depth=3 .....................................
    [CV] ......... arbol_decision__max_depth=3, score=0.979, total=   0.0s
    [CV] arbol_decision__max_depth=3 .....................................
    [CV] ......... arbol_decision__max_depth=3, score=0.958, total=   0.0s
    [CV] arbol_decision__max_depth=3 .....................................
    [CV] ......... arbol_decision__max_depth=3, score=0.975, total=   0.0s
    [CV] arbol_decision__max_depth=4 .....................................
    [CV] ......... arbol_decision__max_depth=4, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=4 .....................................
    [CV] ......... arbol_decision__max_depth=4, score=0.990, total=   0.0s
    [CV] arbol_decision__max_depth=4 .....................................
    [CV] ......... arbol_decision__max_depth=4, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=4 .....................................
    [CV] ......... arbol_decision__max_depth=4, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=4 .....................................
    [CV] ......... arbol_decision__max_depth=4, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=4 .....................................
    [CV] ......... arbol_decision__max_depth=4, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=4 .....................................
    [CV] ......... arbol_decision__max_depth=4, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=4 .....................................
    [CV] ......... arbol_decision__max_depth=4, score=0.979, total=   0.0s
    [CV] arbol_decision__max_depth=4 .....................................
    [CV] ......... arbol_decision__max_depth=4, score=0.961, total=   0.0s
    [CV] arbol_decision__max_depth=4 .....................................
    [CV] ......... arbol_decision__max_depth=4, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=5 .....................................
    [CV] ......... arbol_decision__max_depth=5, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=5 .....................................
    [CV] ......... arbol_decision__max_depth=5, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=5 .....................................
    [CV] ......... arbol_decision__max_depth=5, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=5 .....................................
    [CV] ......... arbol_decision__max_depth=5, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=5 .....................................
    [CV] ......... arbol_decision__max_depth=5, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=5 .....................................
    [CV] ......... arbol_decision__max_depth=5, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=5 .....................................
    [CV] ......... arbol_decision__max_depth=5, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=5 .....................................
    [CV] ......... arbol_decision__max_depth=5, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=5 .....................................
    [CV] ......... arbol_decision__max_depth=5, score=0.973, total=   0.0s
    [CV] arbol_decision__max_depth=5 .....................................
    [CV] ......... arbol_decision__max_depth=5, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=6 .....................................
    [CV] ......... arbol_decision__max_depth=6, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=6 .....................................
    [CV] ......... arbol_decision__max_depth=6, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=6 .....................................
    [CV] ......... arbol_decision__max_depth=6, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=6 .....................................
    [CV] ......... arbol_decision__max_depth=6, score=0.990, total=   0.0s
    [CV] arbol_decision__max_depth=6 .....................................
    [CV] ......... arbol_decision__max_depth=6, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=6 .....................................
    [CV] ......... arbol_decision__max_depth=6, score=0.990, total=   0.0s
    [CV] arbol_decision__max_depth=6 .....................................
    [CV] ......... arbol_decision__max_depth=6, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=6 .....................................
    [CV] ......... arbol_decision__max_depth=6, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=6 .....................................
    [CV] ......... arbol_decision__max_depth=6, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=6 .....................................
    [CV] ......... arbol_decision__max_depth=6, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=7 .....................................
    [CV] ......... arbol_decision__max_depth=7, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=7 .....................................
    [CV] ......... arbol_decision__max_depth=7, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=7 .....................................
    [CV] ......... arbol_decision__max_depth=7, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=7 .....................................
    [CV] ......... arbol_decision__max_depth=7, score=0.992, total=   0.0s
    [CV] arbol_decision__max_depth=7 .....................................
    [CV] ......... arbol_decision__max_depth=7, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=7 .....................................
    [CV] ......... arbol_decision__max_depth=7, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=7 .....................................
    [CV] ......... arbol_decision__max_depth=7, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=7 .....................................
    [CV] ......... arbol_decision__max_depth=7, score=0.979, total=   0.0s
    [CV] arbol_decision__max_depth=7 .....................................
    [CV] ......... arbol_decision__max_depth=7, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=7 .....................................
    [CV] ......... arbol_decision__max_depth=7, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=8 .....................................
    [CV] ......... arbol_decision__max_depth=8, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=8 .....................................
    [CV] ......... arbol_decision__max_depth=8, score=0.990, total=   0.0s
    [CV] arbol_decision__max_depth=8 .....................................
    [CV] ......... arbol_decision__max_depth=8, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=8 .....................................
    [CV] ......... arbol_decision__max_depth=8, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=8 .....................................
    [CV] ......... arbol_decision__max_depth=8, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=8 .....................................
    [CV] ......... arbol_decision__max_depth=8, score=0.990, total=   0.0s
    [CV] arbol_decision__max_depth=8 .....................................
    [CV] ......... arbol_decision__max_depth=8, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=8 .....................................
    [CV] ......... arbol_decision__max_depth=8, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=8 .....................................
    [CV] ......... arbol_decision__max_depth=8, score=0.975, total=   0.0s
    [CV] arbol_decision__max_depth=8 .....................................
    [CV] ......... arbol_decision__max_depth=8, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=9 .....................................
    [CV] ......... arbol_decision__max_depth=9, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=9 .....................................
    [CV] ......... arbol_decision__max_depth=9, score=0.994, total=   0.0s
    [CV] arbol_decision__max_depth=9 .....................................
    [CV] ......... arbol_decision__max_depth=9, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=9 .....................................
    [CV] ......... arbol_decision__max_depth=9, score=0.994, total=   0.0s
    [CV] arbol_decision__max_depth=9 .....................................
    [CV] ......... arbol_decision__max_depth=9, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=9 .....................................
    [CV] ......... arbol_decision__max_depth=9, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=9 .....................................
    [CV] ......... arbol_decision__max_depth=9, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=9 .....................................
    [CV] ......... arbol_decision__max_depth=9, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=9 .....................................
    [CV] ......... arbol_decision__max_depth=9, score=0.975, total=   0.0s
    [CV] arbol_decision__max_depth=9 .....................................
    [CV] ......... arbol_decision__max_depth=9, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=10 ....................................
    [CV] ........ arbol_decision__max_depth=10, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=10 ....................................
    [CV] ........ arbol_decision__max_depth=10, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=10 ....................................
    [CV] ........ arbol_decision__max_depth=10, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=10 ....................................
    [CV] ........ arbol_decision__max_depth=10, score=0.990, total=   0.0s
    [CV] arbol_decision__max_depth=10 ....................................
    [CV] ........ arbol_decision__max_depth=10, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=10 ....................................
    [CV] ........ arbol_decision__max_depth=10, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=10 ....................................
    [CV] ........ arbol_decision__max_depth=10, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=10 ....................................
    [CV] ........ arbol_decision__max_depth=10, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=10 ....................................
    [CV] ........ arbol_decision__max_depth=10, score=0.979, total=   0.0s
    [CV] arbol_decision__max_depth=10 ....................................
    [CV] ........ arbol_decision__max_depth=10, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=11 ....................................
    [CV] ........ arbol_decision__max_depth=11, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=11 ....................................
    [CV] ........ arbol_decision__max_depth=11, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=11 ....................................
    [CV] ........ arbol_decision__max_depth=11, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=11 ....................................
    [CV] ........ arbol_decision__max_depth=11, score=0.992, total=   0.0s
    [CV] arbol_decision__max_depth=11 ....................................
    [CV] ........ arbol_decision__max_depth=11, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=11 ....................................
    [CV] ........ arbol_decision__max_depth=11, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=11 ....................................
    [CV] ........ arbol_decision__max_depth=11, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=11 ....................................
    [CV] ........ arbol_decision__max_depth=11, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=11 ....................................
    [CV] ........ arbol_decision__max_depth=11, score=0.979, total=   0.0s
    [CV] arbol_decision__max_depth=11 ....................................
    [CV] ........ arbol_decision__max_depth=11, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=12 ....................................
    [CV] ........ arbol_decision__max_depth=12, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=12 ....................................
    [CV] ........ arbol_decision__max_depth=12, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=12 ....................................
    [CV] ........ arbol_decision__max_depth=12, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=12 ....................................
    [CV] ........ arbol_decision__max_depth=12, score=0.990, total=   0.0s
    [CV] arbol_decision__max_depth=12 ....................................
    [CV] ........ arbol_decision__max_depth=12, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=12 ....................................
    [CV] ........ arbol_decision__max_depth=12, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=12 ....................................
    [CV] ........ arbol_decision__max_depth=12, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=12 ....................................
    [CV] ........ arbol_decision__max_depth=12, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=12 ....................................
    [CV] ........ arbol_decision__max_depth=12, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=12 ....................................
    [CV] ........ arbol_decision__max_depth=12, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=13 ....................................
    [CV] ........ arbol_decision__max_depth=13, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=13 ....................................
    [CV] ........ arbol_decision__max_depth=13, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=13 ....................................
    [CV] ........ arbol_decision__max_depth=13, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=13 ....................................
    [CV] ........ arbol_decision__max_depth=13, score=0.992, total=   0.0s
    [CV] arbol_decision__max_depth=13 ....................................
    [CV] ........ arbol_decision__max_depth=13, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=13 ....................................
    [CV] ........ arbol_decision__max_depth=13, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=13 ....................................
    [CV] ........ arbol_decision__max_depth=13, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=13 ....................................
    [CV] ........ arbol_decision__max_depth=13, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=13 ....................................
    [CV] ........ arbol_decision__max_depth=13, score=0.975, total=   0.0s
    [CV] arbol_decision__max_depth=13 ....................................
    [CV] ........ arbol_decision__max_depth=13, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=14 ....................................
    [CV] ........ arbol_decision__max_depth=14, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=14 ....................................
    [CV] ........ arbol_decision__max_depth=14, score=0.990, total=   0.0s
    [CV] arbol_decision__max_depth=14 ....................................
    [CV] ........ arbol_decision__max_depth=14, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=14 ....................................
    [CV] ........ arbol_decision__max_depth=14, score=0.994, total=   0.0s
    [CV] arbol_decision__max_depth=14 ....................................
    [CV] ........ arbol_decision__max_depth=14, score=0.987, total=   0.0s
    [CV] arbol_decision__max_depth=14 ....................................
    [CV] ........ arbol_decision__max_depth=14, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=14 ....................................
    [CV] ........ arbol_decision__max_depth=14, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=14 ....................................
    [CV] ........ arbol_decision__max_depth=14, score=0.979, total=   0.0s
    [CV] arbol_decision__max_depth=14 ....................................
    [CV] ........ arbol_decision__max_depth=14, score=0.977, total=   0.0s
    [CV] arbol_decision__max_depth=14 ....................................
    [CV] ........ arbol_decision__max_depth=14, score=0.983, total=   0.0s
    [CV] arbol_decision__max_depth=15 ....................................
    [CV] ........ arbol_decision__max_depth=15, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=15 ....................................
    [CV] ........ arbol_decision__max_depth=15, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=15 ....................................
    [CV] ........ arbol_decision__max_depth=15, score=0.988, total=   0.0s
    [CV] arbol_decision__max_depth=15 ....................................
    [CV] ........ arbol_decision__max_depth=15, score=0.994, total=   0.0s
    [CV] arbol_decision__max_depth=15 ....................................
    [CV] ........ arbol_decision__max_depth=15, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=15 ....................................
    [CV] ........ arbol_decision__max_depth=15, score=0.985, total=   0.0s
    [CV] arbol_decision__max_depth=15 ....................................
    [CV] ........ arbol_decision__max_depth=15, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=15 ....................................
    [CV] ........ arbol_decision__max_depth=15, score=0.981, total=   0.0s
    [CV] arbol_decision__max_depth=15 ....................................
    [CV] ........ arbol_decision__max_depth=15, score=0.973, total=   0.0s
    [CV] arbol_decision__max_depth=15 ....................................
    [CV] ........ arbol_decision__max_depth=15, score=0.985, total=   0.0s
    

    [Parallel(n_jobs=1)]: Done 150 out of 150 | elapsed:    2.4s finished
    




    GridSearchCV(cv=10, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('arbol_decision',
                                            DecisionTreeClassifier(ccp_alpha=0.0,
                                                                   class_weight=None,
                                                                   criterion='gini',
                                                                   max_depth=None,
                                                                   max_features=None,
                                                                   max_leaf_nodes=None,
                                                                   min_impurity_decrease=0.0,
                                                                   min_impurity_split=None,
                                                                   min_samples_leaf=1,
                                                                   min_samples_split=2,
                                                                   min_weight_fraction_leaf=0.0,
                                                                   presort='deprecated',
                                                                   random_state=None,
                                                                   splitter='best'))],
                                    verbose=False),
                 iid='deprecated', n_jobs=None,
                 param_grid={'arbol_decision__max_depth': [1, 2, 3, 4, 5, 6, 7, 8,
                                                           9, 10, 11, 12, 13, 14,
                                                           15]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=3)




```python
print("Accuracy: ",gs_arbol_decision.best_score_)
print("Profundidad:", gs_arbol_decision.best_params_)
```

    Accuracy:  0.985374240403142
    Profundidad: {'arbol_decision__max_depth': 10}
    

## 3.3. K nearest neighbors


```python
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

pipeline_vecinos = Pipeline([("est", StandardScaler()),
                            ("pecea", PCA()),
                            ("vecinos", KNeighborsClassifier())
                            ])

grid_hyper_vecinos = {"pecea__n_components": [1,2,3],
                     "vecinos__n_neighbors": [1,3,5,7,9,11,13,15]}

gs_vecinos = GridSearchCV(pipeline_vecinos,
                         param_grid=grid_hyper_vecinos,
                         cv=10,
                         scoring="accuracy",
                         verbose=3)
```


```python
gs_vecinos.fit(train_wines[wine_features], train_wines["tipo"])
```

    Fitting 10 folds for each of 24 candidates, totalling 240 fits
    [CV] pecea__n_components=1, vecinos__n_neighbors=1 ...................
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    

    [CV]  pecea__n_components=1, vecinos__n_neighbors=1, score=0.977, total=   0.2s
    [CV] pecea__n_components=1, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=1, score=0.979, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=1, score=0.987, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=1, score=0.967, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=1, score=0.971, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=1, score=0.981, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=1, score=0.960, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=1, score=0.981, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=1, score=0.963, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=1 ...................
    

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.2s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.2s remaining:    0.0s
    

    [CV]  pecea__n_components=1, vecinos__n_neighbors=1, score=0.973, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=3, score=0.985, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=3, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=3, score=0.979, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=3, score=0.975, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=3, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=3, score=0.975, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=3, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=3, score=0.973, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=3, score=0.971, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=3, score=0.987, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=5, score=0.983, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=5, score=0.981, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=5, score=0.985, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=5, score=0.979, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=5, score=0.971, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=5, score=0.983, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=5, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=5, score=0.979, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=5, score=0.975, total=   0.1s
    [CV] pecea__n_components=1, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=5, score=0.988, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=7, score=0.983, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=7, score=0.983, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=7, score=0.987, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=7, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=7, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=7, score=0.983, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=7, score=0.979, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=7, score=0.979, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=7, score=0.973, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=7, score=0.987, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=9, score=0.985, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=9, score=0.983, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=9, score=0.988, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=9, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=9, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=9, score=0.983, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=9, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=9, score=0.981, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=9, score=0.973, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=9, score=0.987, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=11, score=0.985, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=11, score=0.983, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=11, score=0.988, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=11, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=11, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=11, score=0.985, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=11, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=11, score=0.979, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=11, score=0.973, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=11, score=0.988, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=13, score=0.985, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=13, score=0.985, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=13, score=0.990, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=13, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=13, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=13, score=0.985, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=13, score=0.979, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=13, score=0.981, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=13, score=0.975, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=13, score=0.988, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=15, score=0.985, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=15, score=0.985, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=15, score=0.990, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=15, score=0.977, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=15, score=0.975, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=15, score=0.985, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=15, score=0.979, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=15, score=0.981, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=15, score=0.975, total=   0.0s
    [CV] pecea__n_components=1, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=1, vecinos__n_neighbors=15, score=0.987, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=1, score=0.977, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=1, score=0.983, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=1, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=1, score=0.983, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=1, score=0.975, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=1, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=1, score=0.969, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=1, score=0.969, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=1, score=0.979, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=1, score=0.985, total=   0.1s
    [CV] pecea__n_components=2, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=3, score=0.985, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=3, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=3, score=0.988, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=3, score=0.979, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=3, score=0.979, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=3, score=0.985, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=3, score=0.967, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=3, score=0.977, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=3, score=0.975, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=3, score=0.990, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=5, score=0.983, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=5, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=5, score=0.988, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=5, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=5, score=0.983, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=5, score=0.983, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=5, score=0.977, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=5, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=5, score=0.975, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=5, score=0.988, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=7, score=0.983, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=7, score=0.983, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=7, score=0.990, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=7, score=0.979, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=7, score=0.979, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=7, score=0.987, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=7, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=7, score=0.983, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=7, score=0.971, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=7, score=0.992, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=9, score=0.987, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=9, score=0.983, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=9, score=0.988, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=9, score=0.979, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=9, score=0.977, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=9, score=0.985, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=9, score=0.977, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=9, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=9, score=0.975, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=9, score=0.990, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=11, score=0.987, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=11, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=11, score=0.988, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=11, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=11, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=11, score=0.985, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=11, score=0.979, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=11, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=11, score=0.975, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=11, score=0.990, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=13, score=0.988, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=13, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=13, score=0.987, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=13, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=13, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=13, score=0.987, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=13, score=0.979, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=13, score=0.983, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=13, score=0.977, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=13, score=0.990, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=15, score=0.988, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=15, score=0.983, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=15, score=0.985, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=15, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=15, score=0.981, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=15, score=0.987, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=15, score=0.979, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=15, score=0.979, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=15, score=0.977, total=   0.0s
    [CV] pecea__n_components=2, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=2, vecinos__n_neighbors=15, score=0.990, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=1, score=0.988, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=1, score=0.981, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=1, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=1, score=0.979, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=1, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=1, score=0.979, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=1, score=0.973, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=1, score=0.983, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=1, score=0.971, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=1 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=1, score=0.973, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=3, score=0.983, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=3, score=0.983, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=3, score=0.987, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=3, score=0.979, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=3, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=3, score=0.988, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=3, score=0.971, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=3, score=0.979, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=3, score=0.977, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=3 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=3, score=0.983, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=5, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=5, score=0.987, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=5, score=0.987, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=5, score=0.981, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=5, score=0.981, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=5, score=0.987, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=5, score=0.979, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=5, score=0.977, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=5, score=0.971, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=5 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=5, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=7, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=7, score=0.987, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=7, score=0.987, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=7, score=0.975, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=7, score=0.983, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=7, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=7, score=0.979, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=7, score=0.979, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=7, score=0.973, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=7 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=7, score=0.988, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=9, score=0.988, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=9, score=0.987, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=9, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=9, score=0.971, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=9, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=9, score=0.983, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=9, score=0.981, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=9, score=0.977, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=9, score=0.975, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=9 ...................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=9, score=0.988, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=11, score=0.987, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=11, score=0.988, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=11, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=11, score=0.975, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=11, score=0.983, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=11, score=0.983, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=11, score=0.981, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=11, score=0.979, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=11, score=0.975, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=11 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=11, score=0.988, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=13, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=13, score=0.987, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=13, score=0.987, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=13, score=0.981, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=13, score=0.981, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=13, score=0.983, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=13, score=0.983, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=13, score=0.979, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=13, score=0.973, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=13 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=13, score=0.985, total=   0.1s
    [CV] pecea__n_components=3, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=15, score=0.987, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=15, score=0.987, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=15, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=15, score=0.979, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=15, score=0.981, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=15, score=0.985, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=15, score=0.981, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=15, score=0.983, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=15, score=0.975, total=   0.0s
    [CV] pecea__n_components=3, vecinos__n_neighbors=15 ..................
    [CV]  pecea__n_components=3, vecinos__n_neighbors=15, score=0.983, total=   0.0s
    

    [Parallel(n_jobs=1)]: Done 240 out of 240 | elapsed:    6.4s finished
    




    GridSearchCV(cv=10, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('est',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('pecea',
                                            PCA(copy=True, iterated_power='auto',
                                                n_components=None,
                                                random_state=None,
                                                svd_solver='auto', tol=0.0,
                                                whiten=False)),
                                           ('vecinos',
                                            KNeighborsClassifier(algorithm='auto',
                                                                 leaf_size=30,
                                                                 metric='minkowski',
                                                                 metric_params=None,
                                                                 n_jobs=None,
                                                                 n_neighbors=5, p=2,
                                                                 weights='uniform'))],
                                    verbose=False),
                 iid='deprecated', n_jobs=None,
                 param_grid={'pecea__n_components': [1, 2, 3],
                             'vecinos__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=3)




```python
print("Accuracy: ",gs_vecinos.best_score_ * 100, "%")
print("Mejores parámetros:", gs_vecinos.best_params_)
```

    Accuracy:  98.32595968578627 %
    Mejores parámetros: {'pecea__n_components': 2, 'vecinos__n_neighbors': 13}
    

## 3.4. Random Forest


```python
from sklearn.ensemble import RandomForestClassifier

pipeline_rf = Pipeline([("bosque", RandomForestClassifier())])

grid_hyper_rf = {"bosque__n_estimators": [50,100,150,250,500],
                "bosque__max_depth": np.arange(1,100)}

gs_rf = GridSearchCV(pipeline_rf,
                    param_grid=grid_hyper_rf,
                    cv=10,
                    scoring="accuracy",
                    n_jobs=-1, #número de núcleos a usar en mi ordenador
                    verbose=-1)
```


```python
gs_rf.fit(train_wines[wine_features], train_wines["tipo"])
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:    8.3s
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed:   43.7s
    [Parallel(n_jobs=-1)]: Done 632 tasks      | elapsed:  2.2min
    [Parallel(n_jobs=-1)]: Done 1136 tasks      | elapsed:  4.4min
    [Parallel(n_jobs=-1)]: Done 1784 tasks      | elapsed:  7.2min
    [Parallel(n_jobs=-1)]: Done 2576 tasks      | elapsed: 10.6min
    [Parallel(n_jobs=-1)]: Done 3512 tasks      | elapsed: 14.7min
    [Parallel(n_jobs=-1)]: Done 4592 tasks      | elapsed: 19.4min
    [Parallel(n_jobs=-1)]: Done 4950 out of 4950 | elapsed: 20.9min finished
    




    GridSearchCV(cv=10, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('bosque',
                                            RandomForestClassifier(bootstrap=True,
                                                                   ccp_alpha=0.0,
                                                                   class_weight=None,
                                                                   criterion='gini',
                                                                   max_depth=None,
                                                                   max_features='auto',
                                                                   max_leaf_nodes=None,
                                                                   max_samples=None,
                                                                   min_impurity_decrease=0.0,
                                                                   min_impurity_split=None,
                                                                   min_samples_leaf=1,
                                                                   min_samples_split=2,
                                                                   min_weight_fraction...
           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
           52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
           69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
           86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]),
                             'bosque__n_estimators': [50, 100, 150, 250, 500]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=-1)




```python
print("Accuracy: ",gs_rf.best_score_ * 100, "%")
print("Mejores parámetros:", gs_rf.best_params_)
```

    Accuracy:  99.57673780939677 %
    Mejores parámetros: {'bosque__max_depth': 21, 'bosque__n_estimators': 150}
    

## 3.5. Support Vector Machines


```python
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV, SelectKBest, f_classif


svm = Pipeline(steps=[("scaler",StandardScaler()),
                      ("svm",SVC())
                     ]
              )


grid_svm = {"svm__C": [0.1, 0.5, 1.0, 5.0],
            "svm__kernel": ["linear","rbf"],
            "svm__degree": [2,3,4],
            "svm__gamma": [0.001, 0.1, "auto", 1.0, 10.0]
           }


gs_svm = GridSearchCV(svm,
                      grid_svm,
                      cv=10,
                      scoring="accuracy",
                      verbose=1,
                      n_jobs=-1)
```


```python
gs_svm.fit(train_wines[wine_features], train_wines["tipo"])
```

    Fitting 10 folds for each of 120 candidates, totalling 1200 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  52 tasks      | elapsed:    2.4s
    [Parallel(n_jobs=-1)]: Done 240 tasks      | elapsed:   18.0s
    [Parallel(n_jobs=-1)]: Done 490 tasks      | elapsed:   36.9s
    [Parallel(n_jobs=-1)]: Done 840 tasks      | elapsed:  1.0min
    [Parallel(n_jobs=-1)]: Done 1200 out of 1200 | elapsed:  1.5min finished
    




    GridSearchCV(cv=10, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('scaler',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('svm',
                                            SVC(C=1.0, break_ties=False,
                                                cache_size=200, class_weight=None,
                                                coef0=0.0,
                                                decision_function_shape='ovr',
                                                degree=3, gamma='scale',
                                                kernel='rbf', max_iter=-1,
                                                probability=False,
                                                random_state=None, shrinking=True,
                                                tol=0.001, verbose=False))],
                                    verbose=False),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'svm__C': [0.1, 0.5, 1.0, 5.0],
                             'svm__degree': [2, 3, 4],
                             'svm__gamma': [0.001, 0.1, 'auto', 1.0, 10.0],
                             'svm__kernel': ['linear', 'rbf']},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=1)




```python
print("Accuracy SVM: ",gs_svm.best_score_ * 100, "%")
print("Mejores parámetros:", gs_svm.best_params_)
```

    Accuracy SVM:  99.61508818734252 %
    Mejores parámetros: {'svm__C': 5.0, 'svm__degree': 2, 'svm__gamma': 'auto', 'svm__kernel': 'rbf'}
    

## 3.5. Mejores parámetros y modelo elegido

Todos los modelos tienen una buena "accuracy", pero tenemos que quedarnos con uno, así que elegimos al que tiene, aunque por décimas, la mejor


```python
todos_los_grid_searchs =[("Regresión Logística",gs_reglog.best_score_), ("Árbol de Decisión", gs_arbol_decision.best_score_),("K Nearest neighbors", gs_vecinos.best_score_),("Random Forest",gs_rf.best_score_),("SVM",gs_svm.best_score_)]

mejor_score_de_cada_gridsearch_df = pd.DataFrame(todos_los_grid_searchs,
                                                 columns=["GridSearchCV", "Mejor score"])

mejor_score_de_cada_gridsearch_df_ordenado = (mejor_score_de_cada_gridsearch_df
                                              .sort_values(by="Mejor score", ascending=False)
                                             )

mejor_score_de_cada_gridsearch_df_ordenado
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GridSearchCV</th>
      <th>Mejor score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>SVM</td>
      <td>0.996151</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Random Forest</td>
      <td>0.995767</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Regresión Logística</td>
      <td>0.993458</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Árbol de Decisión</td>
      <td>0.985374</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K Nearest neighbors</td>
      <td>0.983260</td>
    </tr>
  </tbody>
</table>
</div>



Tenemos a un ganador, así que entrenaré el modelo con Support Vector Machines


```python
mejor_gridsearch=todos_los_grid_searchs[4][1]
mejor_gridsearch
```




    0.9961508818734253




```python
mejor_pipeline = gs_svm.best_estimator_

mejor_pipeline.steps
```




    [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
     ('svm',
      SVC(C=5.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape='ovr', degree=2, gamma='auto', kernel='rbf',
          max_iter=-1, probability=False, random_state=None, shrinking=True,
          tol=0.001, verbose=False))]



Ahora entrenamos el modelo ganador con todo el conjunto de train (ya sin validación cruzada)



```python
mejor_pipeline.fit(train_wines[wine_features], train_wines["tipo"])
```




    Pipeline(memory=None,
             steps=[('scaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('svm',
                     SVC(C=5.0, break_ties=False, cache_size=200, class_weight=None,
                         coef0=0.0, decision_function_shape='ovr', degree=2,
                         gamma='auto', kernel='rbf', max_iter=-1, probability=False,
                         random_state=None, shrinking=True, tol=0.001,
                         verbose=False))],
             verbose=False)



# 4. Mediciones sobre el conjunto de test

## 4.1. Predicciones


```python
predicciones_test = mejor_pipeline.predict(test_wines[wine_features])
predicciones_test
```




    array([1, 1, 1, ..., 1, 1, 1], dtype=int64)




```python
test_wines["predicciones"]= predicciones_test
test_wines
```

    C:\Users\ksalg\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
      <th>tipo</th>
      <th>predicciones</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4081</th>
      <td>7.5</td>
      <td>0.24</td>
      <td>0.29</td>
      <td>1.10</td>
      <td>0.046</td>
      <td>34.0</td>
      <td>84.0</td>
      <td>0.99020</td>
      <td>3.04</td>
      <td>0.39</td>
      <td>11.45</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5735</th>
      <td>6.6</td>
      <td>0.36</td>
      <td>0.21</td>
      <td>1.50</td>
      <td>0.049</td>
      <td>39.0</td>
      <td>184.0</td>
      <td>0.99280</td>
      <td>3.18</td>
      <td>0.41</td>
      <td>9.90</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>243</th>
      <td>6.4</td>
      <td>0.26</td>
      <td>0.49</td>
      <td>6.40</td>
      <td>0.037</td>
      <td>37.0</td>
      <td>161.0</td>
      <td>0.99540</td>
      <td>3.38</td>
      <td>0.53</td>
      <td>9.70</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5100</th>
      <td>8.5</td>
      <td>0.46</td>
      <td>0.31</td>
      <td>2.25</td>
      <td>0.078</td>
      <td>32.0</td>
      <td>58.0</td>
      <td>0.99800</td>
      <td>3.33</td>
      <td>0.54</td>
      <td>9.80</td>
      <td>5</td>
      <td>red</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1732</th>
      <td>7.9</td>
      <td>0.18</td>
      <td>0.49</td>
      <td>5.20</td>
      <td>0.051</td>
      <td>36.0</td>
      <td>157.0</td>
      <td>0.99530</td>
      <td>3.18</td>
      <td>0.48</td>
      <td>10.60</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>766</th>
      <td>7.6</td>
      <td>0.22</td>
      <td>0.28</td>
      <td>12.00</td>
      <td>0.056</td>
      <td>68.0</td>
      <td>143.0</td>
      <td>0.99830</td>
      <td>2.99</td>
      <td>0.30</td>
      <td>9.20</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1363</th>
      <td>6.3</td>
      <td>0.27</td>
      <td>0.25</td>
      <td>5.80</td>
      <td>0.038</td>
      <td>52.0</td>
      <td>155.0</td>
      <td>0.99500</td>
      <td>3.28</td>
      <td>0.38</td>
      <td>9.40</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6255</th>
      <td>7.3</td>
      <td>0.19</td>
      <td>0.27</td>
      <td>13.90</td>
      <td>0.057</td>
      <td>45.0</td>
      <td>155.0</td>
      <td>0.99807</td>
      <td>2.94</td>
      <td>0.41</td>
      <td>8.80</td>
      <td>8</td>
      <td>white</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>63</th>
      <td>6.3</td>
      <td>0.27</td>
      <td>0.18</td>
      <td>7.70</td>
      <td>0.048</td>
      <td>45.0</td>
      <td>186.0</td>
      <td>0.99620</td>
      <td>3.23</td>
      <td>0.47</td>
      <td>9.00</td>
      <td>5</td>
      <td>white</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5470</th>
      <td>5.7</td>
      <td>0.22</td>
      <td>0.22</td>
      <td>16.65</td>
      <td>0.044</td>
      <td>39.0</td>
      <td>110.0</td>
      <td>0.99855</td>
      <td>3.24</td>
      <td>0.48</td>
      <td>9.00</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1300 rows × 15 columns</p>
</div>




```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy= accuracy_score(y_true=test_wines["tipo"], y_pred=test_wines["predicciones"])
print("El modelo tiene un accuracy de ", accuracy*100)
```

    El modelo tiene un accuracy de  99.76923076923076
    

## 4.2. Matriz de confusión

Para comprobar que tan acertado es nuestro modelo, lo vamos a graficar. Utilizaré dos tipos de gráficos: el de matriz de confusión y el de la curva ROC


```python
matriz_confusion = confusion_matrix(y_true=test_wines["tipo"], y_pred=test_wines["predicciones"])
matriz_confusion
```




    array([[308,   3],
           [  0, 989]], dtype=int64)




```python
print(classification_report(y_true=test_wines["tipo"], y_pred=test_wines["predicciones"]))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.99      1.00       311
               1       1.00      1.00      1.00       989
    
        accuracy                           1.00      1300
       macro avg       1.00      1.00      1.00      1300
    weighted avg       1.00      1.00      1.00      1300
    
    


```python
matriz_confusion_df = pd.DataFrame(matriz_confusion,
                                   columns=["Rojo", "Blanco"])

matriz_confusion_df.index = ["Rojo", "Blanco"]

# Y nombramos lo que son las columnas y las filas:
matriz_confusion_df.columns.name = "Predicho"
matriz_confusion_df.index.name = "Real"
```


```python
matriz_confusion_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicho</th>
      <th>Rojo</th>
      <th>Blanco</th>
    </tr>
    <tr>
      <th>Real</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rojo</th>
      <td>308</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Blanco</th>
      <td>0</td>
      <td>989</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,6))
sns.heatmap(matriz_confusion_df,                     
            annot=True,
            cmap="Blues",
            fmt='')
pass
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Ejercicio machine learning (clasificación y regresión)_files/Ejercicio machine learning (clasificación y regresión)_68_0.png">


## 4.3. Curva ROC


```python
from yellowbrick.classifier import ROCAUC

# Instantiate the visualizer with the classification model
model = SVC(C =5.0, degree=2, gamma='auto', kernel='rbf', probability=True)
visualizer = ROCAUC(model, classes=['rojo','blanco'])

visualizer.fit(train_wines[wine_features], train_wines["tipo"])      # Fit the training data to the visualizer
visualizer.score(test_wines[wine_features], test_wines["tipo"])        # Evaluate the model on the test data
visualizer.show()                       # Finalize and show the figure
```

    C:\Users\ksalg\Anaconda3\lib\site-packages\sklearn\base.py:197: FutureWarning: From version 0.24, get_params will raise an AttributeError if a parameter cannot be retrieved as an instance attribute. Previously it would return None.
      FutureWarning)
    


<img src="{{ site.url }}{{ site.baseurl }}/images/Ejercicio machine learning (clasificación y regresión)_files/Ejercicio machine learning (clasificación y regresión)_70_1.png">




    <matplotlib.axes._subplots.AxesSubplot at 0x1a206db72c8>



Nuestro modelo es bastante acertado, por lo que podemos decir que puede darnos predicciones buenas con respecto a nuestra variable "tipo".

# 5. Guardar el mejor modelo


```python
import pickle

# Para exportar, usamos pickle.dump:
with open("mejor_pipeline_wine.model", "wb") as archivo_salida:
    pickle.dump(mejor_pipeline, archivo_salida)
```


```python
with open("mejor_pipeline_wine.model", "rb") as archivo_entrada:
    pipeline_importada = pickle.load(archivo_entrada)
```


```python
pipeline_importada
```




    Pipeline(memory=None,
             steps=[('scaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('svm',
                     SVC(C=5.0, break_ties=False, cache_size=200, class_weight=None,
                         coef0=0.0, decision_function_shape='ovr', degree=2,
                         gamma='auto', kernel='rbf', max_iter=-1, probability=False,
                         random_state=None, shrinking=True, tol=0.001,
                         verbose=False))],
             verbose=False)




```python
predicciones_test = pipeline_importada.predict(test_wines[wine_features])
predicciones_test
test_wines["predicciones"]= predicciones_test
test_wines
accuracy_en_test_pipeline_cargada=  accuracy_score(y_true=test_wines["tipo"], y_pred=test_wines["predicciones"])

accuracy_en_test_pipeline_cargada
```

    C:\Users\ksalg\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    




    0.9976923076923077



# 6. Modelos de Regresión

Vamos a crear otro dataset de train y de test


```python
wine_features_quality = df_wine.columns[0:11]
wine_features_quality
```




    Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol'],
          dtype='object')




```python
from sklearn.model_selection import train_test_split

dataset_separado_quality = train_test_split(df_wine, train_size=0.8, test_size=0.2)

train_wine_quality=dataset_separado_quality[0]
test_wine_quality=dataset_separado_quality[1]

train_wine_quality.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
      <th>tipo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>438</th>
      <td>6.6</td>
      <td>0.13</td>
      <td>0.29</td>
      <td>13.9</td>
      <td>0.056</td>
      <td>33.0</td>
      <td>95.0</td>
      <td>0.99702</td>
      <td>3.17</td>
      <td>0.39</td>
      <td>9.4</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1031</th>
      <td>7.1</td>
      <td>0.18</td>
      <td>0.32</td>
      <td>12.2</td>
      <td>0.048</td>
      <td>36.0</td>
      <td>125.0</td>
      <td>0.99670</td>
      <td>2.92</td>
      <td>0.54</td>
      <td>9.4</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
    </tr>
    <tr>
      <th>159</th>
      <td>7.7</td>
      <td>0.39</td>
      <td>0.28</td>
      <td>4.9</td>
      <td>0.035</td>
      <td>36.0</td>
      <td>109.0</td>
      <td>0.99180</td>
      <td>3.19</td>
      <td>0.58</td>
      <td>12.2</td>
      <td>7</td>
      <td>white</td>
      <td>1</td>
    </tr>
    <tr>
      <th>760</th>
      <td>8.2</td>
      <td>0.33</td>
      <td>0.32</td>
      <td>2.8</td>
      <td>0.067</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>0.99473</td>
      <td>3.30</td>
      <td>0.76</td>
      <td>12.8</td>
      <td>7</td>
      <td>red</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4622</th>
      <td>7.9</td>
      <td>0.19</td>
      <td>0.26</td>
      <td>2.1</td>
      <td>0.039</td>
      <td>8.0</td>
      <td>143.0</td>
      <td>0.99420</td>
      <td>3.05</td>
      <td>0.74</td>
      <td>9.8</td>
      <td>5</td>
      <td>white</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Al ser la misma base, el análisis estadístico debería ser igual


```python
train_wine_quality[wine_features].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.205667</td>
      <td>0.338222</td>
      <td>0.317329</td>
      <td>5.441832</td>
      <td>0.055918</td>
      <td>30.616510</td>
      <td>115.964595</td>
      <td>0.994662</td>
      <td>3.218160</td>
      <td>0.529902</td>
      <td>10.500862</td>
      <td>5.825669</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.297110</td>
      <td>0.162046</td>
      <td>0.145424</td>
      <td>4.803964</td>
      <td>0.035628</td>
      <td>17.815436</td>
      <td>56.680275</td>
      <td>0.003016</td>
      <td>0.160313</td>
      <td>0.148619</td>
      <td>1.199703</td>
      <td>0.878936</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.800000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>0.009000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.987110</td>
      <td>2.720000</td>
      <td>0.220000</td>
      <td>8.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.400000</td>
      <td>0.230000</td>
      <td>0.240000</td>
      <td>1.800000</td>
      <td>0.038000</td>
      <td>17.000000</td>
      <td>78.000000</td>
      <td>0.992250</td>
      <td>3.110000</td>
      <td>0.430000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.000000</td>
      <td>0.290000</td>
      <td>0.310000</td>
      <td>2.900000</td>
      <td>0.047000</td>
      <td>29.000000</td>
      <td>118.000000</td>
      <td>0.994800</td>
      <td>3.210000</td>
      <td>0.500000</td>
      <td>10.300000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.700000</td>
      <td>0.400000</td>
      <td>0.390000</td>
      <td>8.000000</td>
      <td>0.064000</td>
      <td>42.000000</td>
      <td>156.000000</td>
      <td>0.996930</td>
      <td>3.320000</td>
      <td>0.600000</td>
      <td>11.300000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.660000</td>
      <td>65.800000</td>
      <td>0.611000</td>
      <td>289.000000</td>
      <td>440.000000</td>
      <td>1.038980</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>9.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr_train_wine_quality=train_wine_quality[wine_features].corr()
corr_train_wine_quality
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fixed acidity</th>
      <td>1.000000</td>
      <td>0.208195</td>
      <td>0.331580</td>
      <td>-0.106781</td>
      <td>0.291116</td>
      <td>-0.281794</td>
      <td>-0.323581</td>
      <td>0.455687</td>
      <td>-0.256198</td>
      <td>0.302114</td>
      <td>-0.100897</td>
      <td>-0.068354</td>
    </tr>
    <tr>
      <th>volatile acidity</th>
      <td>0.208195</td>
      <td>1.000000</td>
      <td>-0.383064</td>
      <td>-0.195792</td>
      <td>0.361957</td>
      <td>-0.351392</td>
      <td>-0.409622</td>
      <td>0.258103</td>
      <td>0.262145</td>
      <td>0.226211</td>
      <td>-0.023750</td>
      <td>-0.267278</td>
    </tr>
    <tr>
      <th>citric acid</th>
      <td>0.331580</td>
      <td>-0.383064</td>
      <td>1.000000</td>
      <td>0.144105</td>
      <td>0.041425</td>
      <td>0.130274</td>
      <td>0.196807</td>
      <td>0.104077</td>
      <td>-0.333018</td>
      <td>0.059192</td>
      <td>-0.022228</td>
      <td>0.092496</td>
    </tr>
    <tr>
      <th>residual sugar</th>
      <td>-0.106781</td>
      <td>-0.195792</td>
      <td>0.144105</td>
      <td>1.000000</td>
      <td>-0.118488</td>
      <td>0.404059</td>
      <td>0.492872</td>
      <td>0.566839</td>
      <td>-0.268444</td>
      <td>-0.191677</td>
      <td>-0.367345</td>
      <td>-0.042239</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>0.291116</td>
      <td>0.361957</td>
      <td>0.041425</td>
      <td>-0.118488</td>
      <td>1.000000</td>
      <td>-0.189603</td>
      <td>-0.270641</td>
      <td>0.357013</td>
      <td>0.038783</td>
      <td>0.397076</td>
      <td>-0.251235</td>
      <td>-0.194601</td>
    </tr>
    <tr>
      <th>free sulfur dioxide</th>
      <td>-0.281794</td>
      <td>-0.351392</td>
      <td>0.130274</td>
      <td>0.404059</td>
      <td>-0.189603</td>
      <td>1.000000</td>
      <td>0.722123</td>
      <td>0.037191</td>
      <td>-0.136355</td>
      <td>-0.188206</td>
      <td>-0.187050</td>
      <td>0.050036</td>
    </tr>
    <tr>
      <th>total sulfur dioxide</th>
      <td>-0.323581</td>
      <td>-0.409622</td>
      <td>0.196807</td>
      <td>0.492872</td>
      <td>-0.270641</td>
      <td>0.722123</td>
      <td>1.000000</td>
      <td>0.041691</td>
      <td>-0.244810</td>
      <td>-0.284281</td>
      <td>-0.273283</td>
      <td>-0.048736</td>
    </tr>
    <tr>
      <th>density</th>
      <td>0.455687</td>
      <td>0.258103</td>
      <td>0.104077</td>
      <td>0.566839</td>
      <td>0.357013</td>
      <td>0.037191</td>
      <td>0.041691</td>
      <td>1.000000</td>
      <td>-0.003846</td>
      <td>0.246817</td>
      <td>-0.686659</td>
      <td>-0.301582</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>-0.256198</td>
      <td>0.262145</td>
      <td>-0.333018</td>
      <td>-0.268444</td>
      <td>0.038783</td>
      <td>-0.136355</td>
      <td>-0.244810</td>
      <td>-0.003846</td>
      <td>1.000000</td>
      <td>0.196922</td>
      <td>0.141960</td>
      <td>0.027117</td>
    </tr>
    <tr>
      <th>sulphates</th>
      <td>0.302114</td>
      <td>0.226211</td>
      <td>0.059192</td>
      <td>-0.191677</td>
      <td>0.397076</td>
      <td>-0.188206</td>
      <td>-0.284281</td>
      <td>0.246817</td>
      <td>0.196922</td>
      <td>1.000000</td>
      <td>0.012995</td>
      <td>0.050257</td>
    </tr>
    <tr>
      <th>alcohol</th>
      <td>-0.100897</td>
      <td>-0.023750</td>
      <td>-0.022228</td>
      <td>-0.367345</td>
      <td>-0.251235</td>
      <td>-0.187050</td>
      <td>-0.273283</td>
      <td>-0.686659</td>
      <td>0.141960</td>
      <td>0.012995</td>
      <td>1.000000</td>
      <td>0.445344</td>
    </tr>
    <tr>
      <th>quality</th>
      <td>-0.068354</td>
      <td>-0.267278</td>
      <td>0.092496</td>
      <td>-0.042239</td>
      <td>-0.194601</td>
      <td>0.050036</td>
      <td>-0.048736</td>
      <td>-0.301582</td>
      <td>0.027117</td>
      <td>0.050257</td>
      <td>0.445344</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



La variable a predecir es calidad (qualility), en el mapa de calor de correlaciones no se le ve ninguna correlación fuerte con el resto de las variables. Al ser pocas variables, utilizaremos todas para nuestros modelos de Regresión.


```python
mask = np.zeros_like(corr_train_wine_quality)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(12, 8))
    ax = sns.heatmap(corr_train_wine_quality, 
                     mask=mask, 
                     vmin=-1, 
                     vmax=1, 
                     square=True,
                     annot=True,
                     cmap="RdBu_r")
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Ejercicio machine learning (clasificación y regresión)_files/Ejercicio machine learning (clasificación y regresión)_85_0.png">


```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
```

## 6.1. Regresión Lineal


```python
#crear pipeline
pipeline_reg = Pipeline([("estandarizar", StandardScaler()),
                           ("reg", LinearRegression())
                           ])

grid_hyper_reg = {}

gs_reg = GridSearchCV(pipeline_reg,
                        param_grid=grid_hyper_reg,
                        cv=10,
                        scoring="neg_mean_absolute_error",
                        verbose=3)
```


```python
gs_reg.fit(train_wine_quality[wine_features_quality], train_wine_quality["quality"])
```

    Fitting 10 folds for each of 1 candidates, totalling 10 fits
    [CV]  ................................................................
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    

    [CV] ................................... , score=-0.561, total=   0.3s
    [CV]  ................................................................
    [CV] ................................... , score=-0.569, total=   0.0s
    [CV]  ................................................................
    [CV] ................................... , score=-0.568, total=   0.0s
    [CV]  ................................................................
    [CV] ................................... , score=-0.568, total=   0.0s
    [CV]  ................................................................
    [CV] ................................... , score=-0.619, total=   0.0s
    [CV]  ................................................................
    [CV] ................................... , score=-0.583, total=   0.0s
    [CV]  ................................................................
    [CV] ................................... , score=-0.567, total=   0.0s
    [CV]  ................................................................
    [CV] ................................... , score=-0.564, total=   0.0s
    [CV]  ................................................................
    [CV] ................................... , score=-0.558, total=   0.0s
    [CV]  ................................................................
    [CV] ................................... , score=-0.544, total=   0.0s
    

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.2s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.2s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.2s finished
    




    GridSearchCV(cv=10, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('estandarizar',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('reg',
                                            LinearRegression(copy_X=True,
                                                             fit_intercept=True,
                                                             n_jobs=None,
                                                             normalize=False))],
                                    verbose=False),
                 iid='deprecated', n_jobs=None, param_grid={},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='neg_mean_absolute_error', verbose=3)




```python
print("El Error Absoluto medio es: ", gs_reg.best_score_)
```

    El Error Absoluto medio es:  -0.5698899304005819
    

## 6.2. Regresión polinómica


```python
from sklearn.preprocessing import PolynomialFeatures
```


```python
pipeline_reg_poly = Pipeline([("estandarizar", StandardScaler()),
                              ("poly", PolynomialFeatures()),
                              ("linear", LinearRegression(fit_intercept=False))
                           ])

grid_hyper_reg_poly = {"poly__degree": [1,3,5,7]}

gs_reg_poly = GridSearchCV(pipeline_reg_poly,
                        param_grid=grid_hyper_reg_poly,
                        cv=10,
                        scoring="neg_mean_absolute_error",
                        verbose=3)
```


```python
gs_reg_poly.fit(train_wine_quality[wine_features_quality], train_wine_quality["quality"])
```

    Fitting 10 folds for each of 4 candidates, totalling 40 fits
    [CV] poly__degree=1 ..................................................
    [CV] ..................... poly__degree=1, score=-0.561, total=   0.0s
    [CV] poly__degree=1 ..................................................
    [CV] ..................... poly__degree=1, score=-0.569, total=   0.0s
    [CV] poly__degree=1 ..................................................
    [CV] ..................... poly__degree=1, score=-0.568, total=   0.0s
    [CV] poly__degree=1 ..................................................
    [CV] ..................... poly__degree=1, score=-0.568, total=   0.0s
    [CV] poly__degree=1 ..................................................
    [CV] ..................... poly__degree=1, score=-0.619, total=   0.0s
    [CV] poly__degree=1 ..................................................
    [CV] ..................... poly__degree=1, score=-0.583, total=   0.0s
    [CV] poly__degree=1 ..................................................
    [CV] ..................... poly__degree=1, score=-0.567, total=   0.0s
    [CV] poly__degree=1 ..................................................
    [CV] ..................... poly__degree=1, score=-0.564, total=   0.0s
    [CV] poly__degree=1 ..................................................
    [CV] ..................... poly__degree=1, score=-0.558, total=   0.0s
    [CV] poly__degree=1 ..................................................
    [CV] ..................... poly__degree=1, score=-0.544, total=   0.0s
    [CV] poly__degree=3 ..................................................
    [CV] ..................... poly__degree=3, score=-0.568, total=   0.1s
    [CV] poly__degree=3 ..................................................
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s
    

    [CV] ..................... poly__degree=3, score=-0.613, total=   0.1s
    [CV] poly__degree=3 ..................................................
    [CV] ..................... poly__degree=3, score=-0.801, total=   0.1s
    [CV] poly__degree=3 ..................................................
    [CV] ..................... poly__degree=3, score=-0.547, total=   0.1s
    [CV] poly__degree=3 ..................................................
    [CV] ..................... poly__degree=3, score=-0.622, total=   0.1s
    [CV] poly__degree=3 ..................................................
    [CV] ..................... poly__degree=3, score=-0.562, total=   0.1s
    [CV] poly__degree=3 ..................................................
    [CV] ..................... poly__degree=3, score=-0.563, total=   0.1s
    [CV] poly__degree=3 ..................................................
    [CV] ..................... poly__degree=3, score=-0.545, total=   0.1s
    [CV] poly__degree=3 ..................................................
    [CV] ..................... poly__degree=3, score=-0.565, total=   0.1s
    [CV] poly__degree=3 ..................................................
    [CV] ..................... poly__degree=3, score=-0.527, total=   0.1s
    [CV] poly__degree=5 ..................................................
    [CV] .................... poly__degree=5, score=-46.361, total=  32.2s
    [CV] poly__degree=5 ..................................................
    [CV] ................... poly__degree=5, score=-354.785, total=  32.0s
    [CV] poly__degree=5 ..................................................
    [CV] ................... poly__degree=5, score=-933.546, total=  31.9s
    [CV] poly__degree=5 ..................................................
    [CV] .................... poly__degree=5, score=-24.707, total=  32.1s
    [CV] poly__degree=5 ..................................................
    [CV] .................... poly__degree=5, score=-32.117, total=  31.5s
    [CV] poly__degree=5 ..................................................
    [CV] .................... poly__degree=5, score=-25.943, total=  32.0s
    [CV] poly__degree=5 ..................................................
    [CV] ................... poly__degree=5, score=-237.492, total=  32.0s
    [CV] poly__degree=5 ..................................................
    [CV] ................... poly__degree=5, score=-119.943, total=  31.3s
    [CV] poly__degree=5 ..................................................
    [CV] ................... poly__degree=5, score=-106.200, total=  31.8s
    [CV] poly__degree=5 ..................................................
    [CV] .................... poly__degree=5, score=-75.279, total=  32.2s
    [CV] poly__degree=7 ..................................................
    [CV] .................... poly__degree=7, score=-70.070, total=  58.1s
    [CV] poly__degree=7 ..................................................
    [CV] .................. poly__degree=7, score=-2740.563, total=  58.7s
    [CV] poly__degree=7 ..................................................
    [CV] ................. poly__degree=7, score=-89496.685, total=  56.4s
    [CV] poly__degree=7 ..................................................
    [CV] .................... poly__degree=7, score=-30.883, total=  54.0s
    [CV] poly__degree=7 ..................................................
    [CV] .................... poly__degree=7, score=-44.624, total=  55.4s
    [CV] poly__degree=7 ..................................................
    [CV] .................... poly__degree=7, score=-31.966, total=  54.5s
    [CV] poly__degree=7 ..................................................
    [CV] ................... poly__degree=7, score=-969.540, total=  54.3s
    [CV] poly__degree=7 ..................................................
    [CV] ................... poly__degree=7, score=-210.744, total=  54.3s
    [CV] poly__degree=7 ..................................................
    [CV] ................... poly__degree=7, score=-943.549, total=  54.0s
    [CV] poly__degree=7 ..................................................
    [CV] .................... poly__degree=7, score=-59.243, total=  54.3s
    

    [Parallel(n_jobs=1)]: Done  40 out of  40 | elapsed: 14.6min finished
    




    GridSearchCV(cv=10, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('estandarizar',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('poly',
                                            PolynomialFeatures(degree=2,
                                                               include_bias=True,
                                                               interaction_only=False,
                                                               order='C')),
                                           ('linear',
                                            LinearRegression(copy_X=True,
                                                             fit_intercept=False,
                                                             n_jobs=None,
                                                             normalize=False))],
                                    verbose=False),
                 iid='deprecated', n_jobs=None,
                 param_grid={'poly__degree': [1, 3, 5, 7]}, pre_dispatch='2*n_jobs',
                 refit=True, return_train_score=False,
                 scoring='neg_mean_absolute_error', verbose=3)




```python
print("El Error Absoluto medio es: ", gs_reg_poly.best_score_)
print("Mejores parámetros:", gs_reg_poly.best_params_)
```

    El Error Absoluto medio es:  -0.5698899304005819
    Mejores parámetros: {'poly__degree': 1}
    

## 6.3. Árbol de Decisión Regresor


```python
from sklearn.ensemble import RandomForestRegressor

pipeline_rf_regresor = Pipeline([("bosque", RandomForestRegressor())])

grid_hyper_rf_regresor = {"bosque__n_estimators": [50,100,150,250,500],
                "bosque__max_depth": np.arange(1,100)}

gs_rf_regresor = GridSearchCV(pipeline_rf_regresor,
                    param_grid=grid_hyper_rf_regresor,
                    cv=10,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1, #número de núcleos a usar en mi ordenador
                    verbose=3)
```


```python
gs_rf_regresor.fit(train_wine_quality[wine_features_quality], train_wine_quality["quality"])
```

    Fitting 10 folds for each of 495 candidates, totalling 4950 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:    2.6s
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed:   19.7s
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed:  1.3min
    [Parallel(n_jobs=-1)]: Done 496 tasks      | elapsed:  3.8min
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:  8.2min
    [Parallel(n_jobs=-1)]: Done 1136 tasks      | elapsed: 14.8min
    [Parallel(n_jobs=-1)]: Done 1552 tasks      | elapsed: 22.4min
    [Parallel(n_jobs=-1)]: Done 2032 tasks      | elapsed: 31.1min
    [Parallel(n_jobs=-1)]: Done 2576 tasks      | elapsed: 40.8min
    [Parallel(n_jobs=-1)]: Done 3184 tasks      | elapsed: 51.9min
    [Parallel(n_jobs=-1)]: Done 3856 tasks      | elapsed: 64.5min
    [Parallel(n_jobs=-1)]: Done 4592 tasks      | elapsed: 77.8min
    [Parallel(n_jobs=-1)]: Done 4950 out of 4950 | elapsed: 84.4min finished
    




    GridSearchCV(cv=10, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('bosque',
                                            RandomForestRegressor(bootstrap=True,
                                                                  ccp_alpha=0.0,
                                                                  criterion='mse',
                                                                  max_depth=None,
                                                                  max_features='auto',
                                                                  max_leaf_nodes=None,
                                                                  max_samples=None,
                                                                  min_impurity_decrease=0.0,
                                                                  min_impurity_split=None,
                                                                  min_samples_leaf=1,
                                                                  min_samples_split=2,
                                                                  min_weight_fraction_leaf=0.0,
                                                                  n_estimato...
           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
           52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
           69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
           86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]),
                             'bosque__n_estimators': [50, 100, 150, 250, 500]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='neg_mean_absolute_error', verbose=3)




```python
print("El Error Absoluto medio es: ", gs_rf_regresor.best_score_)
print("Mejores parámetros:", gs_rf_regresor.best_params_)
```

    El Error Absoluto medio es:  -0.4371238650566197
    Mejores parámetros: {'bosque__max_depth': 31, 'bosque__n_estimators': 500}
    

## 6.4. Gradient Boosting Regressor


```python
from sklearn.ensemble import GradientBoostingRegressor
pipeline_gb_regresor = Pipeline([("gradient", GradientBoostingRegressor())])

grid_hyper_gb_regresor = {"gradient__n_estimators": [50,100,150,250,500],
                "gradient__max_depth": np.arange(1,100)}

gs_gb_regresor = GridSearchCV(pipeline_gb_regresor,
                    param_grid=grid_hyper_gb_regresor,
                    cv=10,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1, #número de núcleos a usar en mi ordenador
                    verbose=3)
```


```python
gs_gb_regresor.fit(train_wine_quality[wine_features_quality], train_wine_quality["quality"])
```

    Fitting 10 folds for each of 495 candidates, totalling 4950 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:    2.3s
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed:   16.5s
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=-1)]: Done 496 tasks      | elapsed:  4.8min
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:  9.7min
    [Parallel(n_jobs=-1)]: Done 1136 tasks      | elapsed: 13.2min
    [Parallel(n_jobs=-1)]: Done 1552 tasks      | elapsed: 17.1min
    [Parallel(n_jobs=-1)]: Done 2032 tasks      | elapsed: 21.9min
    [Parallel(n_jobs=-1)]: Done 2576 tasks      | elapsed: 27.1min
    [Parallel(n_jobs=-1)]: Done 3184 tasks      | elapsed: 32.9min
    [Parallel(n_jobs=-1)]: Done 3856 tasks      | elapsed: 39.3min
    [Parallel(n_jobs=-1)]: Done 4592 tasks      | elapsed: 46.4min
    [Parallel(n_jobs=-1)]: Done 4950 out of 4950 | elapsed: 49.8min finished
    




    GridSearchCV(cv=10, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('gradient',
                                            GradientBoostingRegressor(alpha=0.9,
                                                                      ccp_alpha=0.0,
                                                                      criterion='friedman_mse',
                                                                      init=None,
                                                                      learning_rate=0.1,
                                                                      loss='ls',
                                                                      max_depth=3,
                                                                      max_features=None,
                                                                      max_leaf_nodes=None,
                                                                      min_impurity_decrease=0.0,
                                                                      min_impurity_split=None,
                                                                      min_samples_leaf=1,
                                                                      min_samples_split=2,
                                                                      min_weight_fr...
           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
           52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
           69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
           86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]),
                             'gradient__n_estimators': [50, 100, 150, 250, 500]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='neg_mean_absolute_error', verbose=3)




```python
print("El Error Absoluto medio es: ", gs_gb_regresor.best_score_)
print("Mejores parámetros:", gs_gb_regresor.best_params_)
```

    El Error Absoluto medio es:  -0.4136292328756165
    Mejores parámetros: {'gradient__max_depth': 9, 'gradient__n_estimators': 500}
    

## 6.5. Support Vector Regressor


```python
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV, SelectKBest, f_classif


svr = Pipeline(steps=[("scaler",StandardScaler()),
                      ("svr",SVR())
                     ]
              )


grid_svr = {"svr__kernel": ["linear","rbf"],
            "svr__degree": [2,3,4,5],
            "svr__gamma": [0.001, 0.1, "scale", 1.0, 10.0, 30.0]
           }


gs_svr = GridSearchCV(svr,
                      grid_svr,
                      cv=10,
                      scoring="neg_mean_absolute_error",
                      verbose=1,
                      n_jobs=-1)
```


```python
gs_svr.fit(train_wine_quality[wine_features_quality], train_wine_quality["quality"])
```

    Fitting 10 folds for each of 48 candidates, totalling 480 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   15.0s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:  3.1min
    [Parallel(n_jobs=-1)]: Done 480 out of 480 | elapsed:  3.5min finished
    




    GridSearchCV(cv=10, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('scaler',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('svr',
                                            SVR(C=1.0, cache_size=200, coef0=0.0,
                                                degree=3, epsilon=0.1,
                                                gamma='scale', kernel='rbf',
                                                max_iter=-1, shrinking=True,
                                                tol=0.001, verbose=False))],
                                    verbose=False),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'svr__degree': [2, 3, 4, 5],
                             'svr__gamma': [0.001, 0.1, 'scale', 1.0, 10.0, 30.0],
                             'svr__kernel': ['linear', 'rbf']},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='neg_mean_absolute_error', verbose=1)




```python
print("El Error Absoluto medio es: ", gs_svr.best_score_)
print("Mejores parámetros:", gs_svr.best_params_)
```

    El Error Absoluto medio es:  -0.4780354288175344
    Mejores parámetros: {'svr__degree': 2, 'svr__gamma': 1.0, 'svr__kernel': 'rbf'}
    

# 7. Mejores Parámetros y modelo elegido

Vamos a comprobar qué modelo nos ha dado el menor error para entrenar con él nuestro conjunto de test


```python
todos_los_grid_searchs_regresion =[("Regresión Lineal",gs_reg.best_score_), ("Regresión Polinómica",gs_reg_poly.best_score_),("Árbol de Decisión Regresor", gs_rf_regresor.best_score_),("Gradient Boosting", gs_gb_regresor.best_score_),("SVM Regresor",gs_svr.best_score_)]

mejor_score_de_cada_gridsearch_df_regresion = pd.DataFrame(todos_los_grid_searchs_regresion,
                                                 columns=["GridSearchCV", "Score"])

mejor_score_de_cada_gridsearch_df_regresion_ordenado = (mejor_score_de_cada_gridsearch_df_regresion
                                              .sort_values(by="Score", ascending=False)
                                             )

mejor_score_de_cada_gridsearch_df_regresion_ordenado
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GridSearchCV</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Gradient Boosting</td>
      <td>-0.413629</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Árbol de Decisión Regresor</td>
      <td>-0.437124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SVM Regresor</td>
      <td>-0.478035</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Regresión Lineal</td>
      <td>-0.569890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Regresión Polinómica</td>
      <td>-0.569890</td>
    </tr>
  </tbody>
</table>
</div>



Tenemos un modelo ganador: Gradient Boosting


```python
mejor_gridsearch_regresion=todos_los_grid_searchs_regresion[3][1]
mejor_gridsearch_regresion
```




    -0.4136292328756165




```python
mejor_pipeline_regresion = gs_gb_regresor.best_estimator_

mejor_pipeline_regresion.steps
```




    [('gradient',
      GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                                init=None, learning_rate=0.1, loss='ls', max_depth=9,
                                max_features=None, max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=500,
                                n_iter_no_change=None, presort='deprecated',
                                random_state=None, subsample=1.0, tol=0.0001,
                                validation_fraction=0.1, verbose=0, warm_start=False))]



Ahora entrenamos nuestro cojunto de test con el modelo ganador y ya sin validación cruzada


```python
mejor_pipeline_regresion.fit(train_wine_quality[wine_features_quality], train_wine_quality["quality"])
```




    Pipeline(memory=None,
             steps=[('gradient',
                     GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0,
                                               criterion='friedman_mse', init=None,
                                               learning_rate=0.1, loss='ls',
                                               max_depth=9, max_features=None,
                                               max_leaf_nodes=None,
                                               min_impurity_decrease=0.0,
                                               min_impurity_split=None,
                                               min_samples_leaf=1,
                                               min_samples_split=2,
                                               min_weight_fraction_leaf=0.0,
                                               n_estimators=500,
                                               n_iter_no_change=None,
                                               presort='deprecated',
                                               random_state=None, subsample=1.0,
                                               tol=0.0001, validation_fraction=0.1,
                                               verbose=0, warm_start=False))],
             verbose=False)



# 8. Mediciones sobre el conjunto de test

## 8.1. Predicciones


```python
predicciones_test_regresion = mejor_pipeline_regresion.predict(test_wine_quality[wine_features_quality])
predicciones_test_regresion
```




    array([5.99896962, 5.00089544, 5.00674158, ..., 5.30597078, 5.20223377,
           5.91397423])




```python
test_wine_quality["predicciones reg"]= predicciones_test_regresion
test_wine_quality
```

    C:\Users\ksalg\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
      <th>tipo</th>
      <th>predicciones reg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6069</th>
      <td>6.1</td>
      <td>0.150</td>
      <td>0.29</td>
      <td>6.2</td>
      <td>0.046</td>
      <td>39.0</td>
      <td>151.0</td>
      <td>0.99471</td>
      <td>3.60</td>
      <td>0.44</td>
      <td>10.6</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
      <td>5.998970</td>
    </tr>
    <tr>
      <th>4078</th>
      <td>7.4</td>
      <td>0.310</td>
      <td>0.26</td>
      <td>8.6</td>
      <td>0.048</td>
      <td>47.0</td>
      <td>206.0</td>
      <td>0.99640</td>
      <td>3.26</td>
      <td>0.36</td>
      <td>9.1</td>
      <td>5</td>
      <td>white</td>
      <td>1</td>
      <td>5.000895</td>
    </tr>
    <tr>
      <th>5653</th>
      <td>7.0</td>
      <td>0.160</td>
      <td>0.32</td>
      <td>8.3</td>
      <td>0.045</td>
      <td>38.0</td>
      <td>126.0</td>
      <td>0.99580</td>
      <td>3.21</td>
      <td>0.34</td>
      <td>9.2</td>
      <td>5</td>
      <td>white</td>
      <td>1</td>
      <td>5.006742</td>
    </tr>
    <tr>
      <th>3378</th>
      <td>7.4</td>
      <td>0.560</td>
      <td>0.09</td>
      <td>1.5</td>
      <td>0.071</td>
      <td>19.0</td>
      <td>117.0</td>
      <td>0.99496</td>
      <td>3.22</td>
      <td>0.53</td>
      <td>9.8</td>
      <td>5</td>
      <td>white</td>
      <td>1</td>
      <td>4.906840</td>
    </tr>
    <tr>
      <th>26</th>
      <td>6.0</td>
      <td>0.280</td>
      <td>0.25</td>
      <td>1.8</td>
      <td>0.042</td>
      <td>8.0</td>
      <td>108.0</td>
      <td>0.99290</td>
      <td>3.08</td>
      <td>0.55</td>
      <td>9.0</td>
      <td>5</td>
      <td>white</td>
      <td>1</td>
      <td>4.995440</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5116</th>
      <td>7.9</td>
      <td>0.190</td>
      <td>0.45</td>
      <td>1.5</td>
      <td>0.045</td>
      <td>17.0</td>
      <td>96.0</td>
      <td>0.99170</td>
      <td>3.13</td>
      <td>0.39</td>
      <td>11.0</td>
      <td>6</td>
      <td>white</td>
      <td>1</td>
      <td>6.172742</td>
    </tr>
    <tr>
      <th>705</th>
      <td>6.3</td>
      <td>0.430</td>
      <td>0.32</td>
      <td>8.8</td>
      <td>0.042</td>
      <td>18.0</td>
      <td>106.0</td>
      <td>0.99172</td>
      <td>3.28</td>
      <td>0.33</td>
      <td>12.9</td>
      <td>7</td>
      <td>white</td>
      <td>1</td>
      <td>6.833655</td>
    </tr>
    <tr>
      <th>6286</th>
      <td>8.0</td>
      <td>0.715</td>
      <td>0.22</td>
      <td>2.3</td>
      <td>0.075</td>
      <td>13.0</td>
      <td>81.0</td>
      <td>0.99688</td>
      <td>3.24</td>
      <td>0.54</td>
      <td>9.5</td>
      <td>6</td>
      <td>red</td>
      <td>0</td>
      <td>5.305971</td>
    </tr>
    <tr>
      <th>4063</th>
      <td>7.5</td>
      <td>0.705</td>
      <td>0.10</td>
      <td>13.0</td>
      <td>0.044</td>
      <td>44.0</td>
      <td>214.0</td>
      <td>0.99741</td>
      <td>3.10</td>
      <td>0.50</td>
      <td>9.1</td>
      <td>5</td>
      <td>white</td>
      <td>1</td>
      <td>5.202234</td>
    </tr>
    <tr>
      <th>4465</th>
      <td>6.7</td>
      <td>0.160</td>
      <td>0.28</td>
      <td>2.5</td>
      <td>0.046</td>
      <td>40.0</td>
      <td>153.0</td>
      <td>0.99210</td>
      <td>3.38</td>
      <td>0.51</td>
      <td>11.4</td>
      <td>7</td>
      <td>white</td>
      <td>1</td>
      <td>5.913974</td>
    </tr>
  </tbody>
</table>
<p>1300 rows × 15 columns</p>
</div>




```python
from sklearn.metrics import mean_squared_error

mse= mean_squared_error(y_true=test_wine_quality["quality"], y_pred=test_wine_quality["predicciones reg"])
print("El modelo tiene un MSE de ", mse)
```

    El modelo tiene un MSE de  0.38980756248300547
    

# 8.2. Y bueno, ¿esto qué quiere decir?

Nuestro modelo tiene un MSE de 0.3898, no está mal, teniendo en cuenta que al ser una métrica de error, mientras sea menor, mejor

Para entenderlo, usemos unas gráficas. En esta lo verde es lo predicho dentro del set de test y la azul los valores reales de "quality" del mismo set.


```python
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import train_test_split
```


```python
model = gs_gb_regresor.best_estimator_
visualizer = ResidualsPlot(model, is_fitted=False, test_alpha=0.25)

visualizer.fit(train_wine_quality[wine_features_quality], train_wine_quality["quality"])  # Fit the training data to the visualizer
visualizer.score(test_wine_quality[wine_features_quality], test_wine_quality["quality"])  # Evaluate the model on the test data
visualizer.show()   
```

<img src="{{ site.url }}{{ site.baseurl }}/images/Ejercicio machine learning (clasificación y regresión)_files/Ejercicio machine learning (clasificación y regresión)_124_0.png">





    <matplotlib.axes._subplots.AxesSubplot at 0x1a209d51e08>



No tiene mala pinta, de hecho con este conjunto de test el modelo es bastante acertado, pero vamos a graficarlo de otro modo. Creo un dataframe con lo predicho y lo real (con solo 30 filas porque sino la gráfica no se apreciará bien)


```python
df_rg = pd.DataFrame({'Actual': test_wine_quality["quality"], 'Predicted': test_wine_quality["predicciones reg"]})
df_rg_pred = df_rg.head(30)
df_rg_pred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6069</th>
      <td>6</td>
      <td>5.998970</td>
    </tr>
    <tr>
      <th>4078</th>
      <td>5</td>
      <td>5.000895</td>
    </tr>
    <tr>
      <th>5653</th>
      <td>5</td>
      <td>5.006742</td>
    </tr>
    <tr>
      <th>3378</th>
      <td>5</td>
      <td>4.906840</td>
    </tr>
    <tr>
      <th>26</th>
      <td>5</td>
      <td>4.995440</td>
    </tr>
  </tbody>
</table>
</div>



Y hacemos un gráfico de barras para comparar los valores predichos con los reales, para ver qué pinta tienen, y comprobamos que tampoco tiene mala pinta:


```python
df_rg_pred.plot(kind='bar',figsize=(15,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Ejercicio machine learning (clasificación y regresión)_files/Ejercicio machine learning (clasificación y regresión)_128_0.png">


Finalmente, ya para la comprobación final, evaluamos la performance del algoritmo, no solo con el MSE, sino también con otras métricas como MAE y RMSE


```python
from sklearn.metrics import r2_score
print('Mean Absolute Error:', mean_absolute_error(test_wine_quality["quality"], test_wine_quality["predicciones reg"])) 
print('Mean Squared Error:', mean_squared_error(test_wine_quality["quality"], test_wine_quality["predicciones reg"])) 
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(test_wine_quality["quality"], test_wine_quality["predicciones reg"])))
print('Square R:', r2_score(test_wine_quality["quality"], test_wine_quality["predicciones reg"]))
```

    Mean Absolute Error: 0.40685658199190217
    Mean Squared Error: 0.38980756248300547
    Root Mean Squared Error: 0.6243457075074718
    Square R: 0.49300212981406344
    

Para recordar, la media de la variable "Quality" es 5.8183


```python
np.mean(df_wine["quality"])
```




    5.818377712790519



El valor del Error Cuadrático Medio está ligeramente por encima del 10% de la media. Lo cual nos indica que nuestro algoritmo no está del todo mal y que puede darnos predicciones razonables, pero que seguramente podría mejorar con un dataset más grande y con unas features que tengan mayor correlación con nuestra variable a predecir. 

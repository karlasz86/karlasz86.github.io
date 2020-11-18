# Introducción

¿Te ha pasado que te piden un análisis y no quieres presentar algo hecho con los gráficos de Excel, pero tampoco tienes a mano un programa de visualización de datos por falta de presupuesto en el trabajo?

Si bien lo importante es el análisis, la forma en la que presentamos los datos también es importante. Por eso te mostraré un ejemplo de análisis que hice con las librerías <strong>Pandas, Seaborn y Matplotlib</strong>.

Recuerda que para elegir los gráficos tienes que tener en cuenta no solo los datos con los que cuentas, sino lo que quieres transmitir y a quién se lo vas a transmitir (no es lo mismo el/la encargado/a de Ventas que el/la Gerente de Operaciones)

¡vamos al lío!


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
```

Tenemos una base de datos de 122 pisos obtenidos del portal Fotocasa, todos ubicados en el Municipio de Madrid y amueblados, con descripción, precio, número de baños, si tiene o no ascensor, garaje o terraza (0 es NO, 1 es SÍ), el barrio y el precio medio por metro cuadrado. En un post anterior explico cómo conseguí los datos.


```python
housing = pd.read_pickle('final_result.pickle')
```


```python
housing.head()
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
      <th>Descripción</th>
      <th>Price</th>
      <th>Rooms</th>
      <th>Baths</th>
      <th>Area</th>
      <th>Elevator</th>
      <th>Parking</th>
      <th>Terrace</th>
      <th>Type</th>
      <th>NB</th>
      <th>SquareMeter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Magnífica vivienda  amueblada en estilo clase ...</td>
      <td>1200</td>
      <td>3</td>
      <td>2</td>
      <td>115</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Latina</td>
      <td>10.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Le ofrecemos este bonito y luminoso piso para ...</td>
      <td>850</td>
      <td>1</td>
      <td>1</td>
      <td>54</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>15.74</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VOhome Propiedades Las Rosas / San Blas ofrece...</td>
      <td>900</td>
      <td>2</td>
      <td>2</td>
      <td>75</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>San Blas</td>
      <td>12.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Moderna y luminosa vivienda totalmente exterio...</td>
      <td>1200</td>
      <td>0</td>
      <td>1</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Salamanca</td>
      <td>17.14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SOLFAI CONSULTING PONE A SU DISPOSICIÓN PRECIO...</td>
      <td>650</td>
      <td>1</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Arganzuela</td>
      <td>16.25</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 122 entries, 0 to 121
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Descripción  122 non-null    object 
     1   Price        122 non-null    int64  
     2   Rooms        122 non-null    int64  
     3   Baths        122 non-null    int64  
     4   Area         122 non-null    int64  
     5   Elevator     122 non-null    int64  
     6   Parking      122 non-null    int64  
     7   Terrace      122 non-null    int64  
     8   Type         122 non-null    int64  
     9   NB           122 non-null    object 
     10  SquareMeter  122 non-null    float64
    dtypes: float64(1), int64(8), object(2)
    memory usage: 10.6+ KB
    

Con la función describe(), obtenemos un resumen estadístico de la distribución de nuestras características numéricas. Por ejemplo, por sacar datos que puedan interesarnos para este reporte:
    <li>La media de <strong>número de habitaciones</strong> es 2. Siendo el mínimo 0 habitaciones (estudios) y el máximo 4.</li>
    <li>La media de número de baños es 1, siendo el mínimo 1 y el máximo 3.</li>
    <li>En promedio <strong>el área de los pisos ronda sobre los 78 metros</strong>. El área mínima es 25 metros y la máxima 200</li>
    <li>El precio mínimo es 550€ y el máximo 5000€</li>


```python
housing.describe()
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
      <th>Price</th>
      <th>Rooms</th>
      <th>Baths</th>
      <th>Area</th>
      <th>Elevator</th>
      <th>Parking</th>
      <th>Terrace</th>
      <th>Type</th>
      <th>SquareMeter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>122.000000</td>
      <td>122.000000</td>
      <td>122.000000</td>
      <td>122.000000</td>
      <td>122.000000</td>
      <td>122.000000</td>
      <td>122.000000</td>
      <td>122.000000</td>
      <td>122.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1303.237705</td>
      <td>2.016393</td>
      <td>1.368852</td>
      <td>78.942623</td>
      <td>0.860656</td>
      <td>0.057377</td>
      <td>0.213115</td>
      <td>0.950820</td>
      <td>16.953525</td>
    </tr>
    <tr>
      <th>std</th>
      <td>681.622373</td>
      <td>1.020319</td>
      <td>0.548489</td>
      <td>32.304414</td>
      <td>0.347733</td>
      <td>0.233521</td>
      <td>0.411197</td>
      <td>0.217136</td>
      <td>6.080150</td>
    </tr>
    <tr>
      <th>min</th>
      <td>550.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.730000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>890.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>56.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>13.082500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1100.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>75.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>15.355000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1400.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>94.750000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>19.597500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5000.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>200.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>56.410000</td>
    </tr>
  </tbody>
</table>
</div>



Veamos estos datos en gráficos de barras, para poder obtener más información interesante.


```python
%matplotlib inline
```

¿Qué podemos destacar más de estas gráficas?
<li> Hay más pisos con <strong>un solo baño</strong></li>
<li> La mayoría de los pisos <strong>tienen ascensor</strong></li>
<li> La mayoría de los pisos <strong>no tiene parking</strong></li>
<li> Es destacable que la mayoría de estos pisos <strong>no tiene terraza</strong></li>


```python
housing.hist(bins=50, figsize=(20,15))
plt.show()
```


![png](output_12_0.png)


## Analizando outliers

De las gráficas anteriores podemos detectar que hay algunos outliers, sobre todo en los precios, y podemos verlo mejor con un diagrama de cajas. Vemos que hay pisos entre 2000 y 5000€, que son atípicos en nuestros datos. 


```python
housing['Price'].plot(kind="box",
                figsize=(4,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x14bc16ee688>




![png](output_15_1.png)



```python
outliers = 2000
```


```python
outliers_price = housing.query('(Price >= @outliers and not Price.isnull())',engine='python')
```


```python
outliers_price
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
      <th>Descripción</th>
      <th>Price</th>
      <th>Rooms</th>
      <th>Baths</th>
      <th>Area</th>
      <th>Elevator</th>
      <th>Parking</th>
      <th>Terrace</th>
      <th>Type</th>
      <th>NB</th>
      <th>SquareMeter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>Vivienda luminosa y reformada de 3 dormitorios...</td>
      <td>2200</td>
      <td>3</td>
      <td>2</td>
      <td>98</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>22.45</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Se alquila ÁTICO en calle Joaquín María López,...</td>
      <td>2000</td>
      <td>4</td>
      <td>2</td>
      <td>150</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Chamberí</td>
      <td>13.33</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Ubicado en la zona de Ibiza y muy cerca del Pa...</td>
      <td>2800</td>
      <td>3</td>
      <td>3</td>
      <td>190</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>14.74</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Este precioso piso de 110m²  amueblado, muy lu...</td>
      <td>3200</td>
      <td>3</td>
      <td>2</td>
      <td>107</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Chamartín</td>
      <td>29.91</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Este maravilloso piso con las mejores calidade...</td>
      <td>2900</td>
      <td>2</td>
      <td>2</td>
      <td>118</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Centro</td>
      <td>24.58</td>
    </tr>
    <tr>
      <th>45</th>
      <td>.En uno de los edificios más emblemáticos de M...</td>
      <td>2100</td>
      <td>2</td>
      <td>1</td>
      <td>107</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Moncloa</td>
      <td>19.63</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Situado en el castizo barrio de Malasaña, se e...</td>
      <td>2300</td>
      <td>3</td>
      <td>2</td>
      <td>170</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Centro</td>
      <td>13.53</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Fantástica vivienda de 200m² en el centro de M...</td>
      <td>3000</td>
      <td>2</td>
      <td>2</td>
      <td>200</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Fuencarral</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>76</th>
      <td>En plena Plaza de España de Madrid, se encuent...</td>
      <td>2100</td>
      <td>2</td>
      <td>1</td>
      <td>105</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Moncloa</td>
      <td>20.00</td>
    </tr>
    <tr>
      <th>81</th>
      <td>SOLUCIONES TENGACASA presenta esta espectacula...</td>
      <td>2000</td>
      <td>3</td>
      <td>2</td>
      <td>112</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Salamanca</td>
      <td>17.86</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Moderno y espectacular piso en el barrio de Ju...</td>
      <td>2000</td>
      <td>1</td>
      <td>1</td>
      <td>90</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Centro</td>
      <td>22.22</td>
    </tr>
    <tr>
      <th>85</th>
      <td>A escasos metros del Palacio de Oriente, Plaza...</td>
      <td>2200</td>
      <td>0</td>
      <td>1</td>
      <td>39</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Moncloa</td>
      <td>56.41</td>
    </tr>
    <tr>
      <th>92</th>
      <td>REDFRIN alquila ÁTICO en calle Joaquín María L...</td>
      <td>2500</td>
      <td>4</td>
      <td>2</td>
      <td>150</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Chamberí</td>
      <td>16.67</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Ubicado en una segunda planta exterior en plen...</td>
      <td>2700</td>
      <td>2</td>
      <td>2</td>
      <td>128</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Centro</td>
      <td>21.09</td>
    </tr>
    <tr>
      <th>97</th>
      <td>La Inmobiliaria Internacional CPM gestiona est...</td>
      <td>2500</td>
      <td>2</td>
      <td>2</td>
      <td>140</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Chamberí</td>
      <td>17.86</td>
    </tr>
    <tr>
      <th>101</th>
      <td>En la prestigiosa zona del Viso, se encuentra ...</td>
      <td>2500</td>
      <td>2</td>
      <td>2</td>
      <td>119</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Chamartín</td>
      <td>21.01</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Piso reformado con calidades de lujo, se distr...</td>
      <td>2200</td>
      <td>2</td>
      <td>3</td>
      <td>85</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Centro</td>
      <td>25.88</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Magnífico piso en la zona de los Jerónimos con...</td>
      <td>5000</td>
      <td>3</td>
      <td>3</td>
      <td>151</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>33.11</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Propiedad amueblada, con dos dormitorios con b...</td>
      <td>3200</td>
      <td>2</td>
      <td>2</td>
      <td>110</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Centro</td>
      <td>29.09</td>
    </tr>
  </tbody>
</table>
</div>



Podemos apreciar que estos pisos están ubicados en barrios en los que los precios altos nos son familiares, como Centro, Retiro, Moncloa, Chamberí, Chamartín, Salamanca y Fuencarral.


```python
outliers_price.groupby('NB')['NB'].count().sort_values(ascending=False)
```




    NB
    Centro        6
    Retiro        3
    Moncloa       3
    Chamberí      3
    Chamartín     2
    Salamanca     1
    Fuencarral    1
    Name: NB, dtype: int64



En este cuadro podemos ver los valores medios de cada característica por barrio. El precio por metro cuadrado más caro es el de Moncloa, cuya media de habitaciones, baños y area, es menor que la de otros barrios.


```python
outliers_price.groupby(['NB']).mean()
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
      <th>Price</th>
      <th>Rooms</th>
      <th>Baths</th>
      <th>Area</th>
      <th>Elevator</th>
      <th>Parking</th>
      <th>Terrace</th>
      <th>Type</th>
      <th>SquareMeter</th>
    </tr>
    <tr>
      <th>NB</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Centro</th>
      <td>2550.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>116.833333</td>
      <td>0.833333</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>22.731667</td>
    </tr>
    <tr>
      <th>Chamartín</th>
      <td>2850.000000</td>
      <td>2.500000</td>
      <td>2.000000</td>
      <td>113.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1.0</td>
      <td>25.460000</td>
    </tr>
    <tr>
      <th>Chamberí</th>
      <td>2333.333333</td>
      <td>3.333333</td>
      <td>2.000000</td>
      <td>146.666667</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>1.0</td>
      <td>15.953333</td>
    </tr>
    <tr>
      <th>Fuencarral</th>
      <td>3000.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>200.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>Moncloa</th>
      <td>2133.333333</td>
      <td>1.333333</td>
      <td>1.000000</td>
      <td>83.666667</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.666667</td>
      <td>1.0</td>
      <td>32.013333</td>
    </tr>
    <tr>
      <th>Retiro</th>
      <td>3333.333333</td>
      <td>3.000000</td>
      <td>2.666667</td>
      <td>146.333333</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>23.433333</td>
    </tr>
    <tr>
      <th>Salamanca</th>
      <td>2000.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>112.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>17.860000</td>
    </tr>
  </tbody>
</table>
</div>



Si echamos un ojo a las descripciones de estos pisos, veremos que tienen una ubicación privilegiada, además de características que le dan un valor añadido como vistas y equipamiento.


```python
outliers_price['Descripción'][76]
```




    'En plena Plaza de España de Madrid, se encuentra este precioso apartamento exterior de dos dormitorios con un cuarto de baño completo, con ducha. Cocina amueblada y equipada con todos los electrodomésticos. El salón tiene acceso a una terraza particular, con orientación oeste, y con vistas espect...'



Pero si vamos un poco más allá, veremos que el <strong>precio por metro cuadrado</strong> de algunos de estos pisos pueden ser un mejor referente para analizar outliers, así que analizaremos qué pisos tienen un precio mayor o igual a 30€/m2


```python
housing['SquareMeter'].plot(kind="box",
                figsize=(4,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x14bc1787ac8>




![png](output_26_1.png)



```python
outliers=30
```


```python
outliers_square_meter = housing.query('(SquareMeter >= @outliers and not SquareMeter.isnull())',engine='python')
```

Además de los 2 pisos que ya habíamos detectado líneas más arriva, vemos que tenermo un piso de una habitación en el Retiro, cuyo precio por metro cuadrado es de 33,10€


```python
outliers_square_meter
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
      <th>Descripción</th>
      <th>Price</th>
      <th>Rooms</th>
      <th>Baths</th>
      <th>Area</th>
      <th>Elevator</th>
      <th>Parking</th>
      <th>Terrace</th>
      <th>Type</th>
      <th>NB</th>
      <th>SquareMeter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>54</th>
      <td>Precioso apartamento reformado de 42m², a estr...</td>
      <td>1390</td>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>33.10</td>
    </tr>
    <tr>
      <th>85</th>
      <td>A escasos metros del Palacio de Oriente, Plaza...</td>
      <td>2200</td>
      <td>0</td>
      <td>1</td>
      <td>39</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Moncloa</td>
      <td>56.41</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Magnífico piso en la zona de los Jerónimos con...</td>
      <td>5000</td>
      <td>3</td>
      <td>3</td>
      <td>151</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>33.11</td>
    </tr>
  </tbody>
</table>
</div>



Si vemos su descripción, también es un piso "top"


```python
outliers_square_meter['Descripción'][54]
```




    'Precioso apartamento reformado de 42m², a estrenar, completamente equipado y listo para entrar. Decorado con muebles modernos de diseño con el máximo aprovechamiento del espacio. Tiene una cocina con todos los electrodomésticos de primeras marcas y menaje, salón-comedor, dormitorio y baño, tiene ...'



## Continuando con nuestra base completa...

Ya hemos echado un vistazo a los outliers de nuestra base (nuestra decisión es dejarlos), así que vamos a retomar la base de los 122 pisos y mirar qué más hay ahí que pueda sernos de utilidad.

Porque es importante para mi análisis, quiero averiguar de todo mi conjunto de datos, qué número de habitaciones es el más común. Eso podemos averiguarlo con una operación sencilla utilizando counts() y dividiendo entre el número de filas de nuestro dataframe. Como ya habíamos visto en la gráfica anterior, hay más pisos de 2 habitaciones, pero porcentualmente ¿cómo se divide esto? 

<li>2 habitaciones: 33%</li>
<li>3 habitaciones: 28%</li>
<li>1 habitación: 25%</li>

Y los "raros" son los estudios y los pisos de 4 habitaciones. Podemos encontrar una oportunidad en los pisos de 4 habitaciones, así por ejemplo, si tenemos un piso a un precio que esté dentro de la media y con una buena decoración, podemos ser más firmes a la hora de la negociación con nuestro potencial inquilino (previo averiguar las características de la competencia): no hay tantos pisos de 4 habitaciones y si son 4 amigos buscando piso, no les va a encajar un piso de 3 así esté tirado de precio. Por supuesto habrá que analizar también otros factores, así como hay pocos pisos disponibles con 4 habitaciones, ¿hay también una baja demanda de este tipo de inmuebles?


```python
(housing['Rooms'].value_counts() / len(housing)).sort_values(ascending=False)
```




    2    0.336066
    3    0.286885
    1    0.254098
    0    0.065574
    4    0.057377
    Name: Rooms, dtype: float64




```python
sns.catplot(x="Rooms", kind="count", palette="ch:start=.2,rot=-.3", data=housing)
```




    <seaborn.axisgrid.FacetGrid at 0x14bc165ea08>




![png](output_37_1.png)


## Pisos de 1 habitación y estudios en el Viso

Tenemos disponible un piso de una habitación en el barrio del Viso y no sabemos qué precio podría estar bien para no quedarnos cortos pero tampoco pasarnos dos pueblos y que no lo quiera nadie. Nosotros lo estamos alquilando a 800€+gastos, pero, ¿cómo está la competencia? Para eso podemos evaluar los pisos de 1 habitación y estudios, no solo en El Viso, sino en todo el barrio de Chamartín y sus alrededores (Salamanca, Retiro y Chamberí), ¿por qué? porque todos estos barrios están muy bien comunicados con el Viso, son muy similares en calidad y si mi piso lo alquilo a 800€, pero la media por pisos de 1 habitación en el barrio de Salamanca es de 600€ (un barrio que se considera más caro), pues ya voy mal.


```python
habs = 1
```


```python
one_bedroom = housing.query('(Rooms <= @habs and not Rooms.isnull()) and ((NB == "Chamberí") | (NB == "Chamartín") | (NB == "Retiro") | (NB == "Salamanca"))',engine='python')
```


```python
one_bedroom.head()
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
      <th>Descripción</th>
      <th>Price</th>
      <th>Rooms</th>
      <th>Baths</th>
      <th>Area</th>
      <th>Elevator</th>
      <th>Parking</th>
      <th>Terrace</th>
      <th>Type</th>
      <th>NB</th>
      <th>SquareMeter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Le ofrecemos este bonito y luminoso piso para ...</td>
      <td>850</td>
      <td>1</td>
      <td>1</td>
      <td>54</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>15.74</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Moderna y luminosa vivienda totalmente exterio...</td>
      <td>1200</td>
      <td>0</td>
      <td>1</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Salamanca</td>
      <td>17.14</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Espectacular dúplex - loft de diseño en calle ...</td>
      <td>980</td>
      <td>0</td>
      <td>1</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Chamberí</td>
      <td>12.25</td>
    </tr>
    <tr>
      <th>15</th>
      <td>""Agencia inmobiliaria de MADRID – zona PACIFI...</td>
      <td>600</td>
      <td>1</td>
      <td>1</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>24.00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Magnifico estudio en la Travesía Andrés Mellad...</td>
      <td>600</td>
      <td>0</td>
      <td>1</td>
      <td>38</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Chamberí</td>
      <td>15.79</td>
    </tr>
  </tbody>
</table>
</div>



Tenemos 21 propiedades entre pisos de una habitación y estudios. 
<li>El precio mínimo es 550€ y el máximo 1390€. El más pequeño tiene un área de 25 m2 y el más grande de 85 m2.</li>
<li>Ninguno tiene garaje incluido.</li>
<li>La mayoría de estos pisos tiene 1 baño y ascensor</li>



```python
one_bedroom.describe()
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
      <th>Price</th>
      <th>Rooms</th>
      <th>Baths</th>
      <th>Area</th>
      <th>Elevator</th>
      <th>Parking</th>
      <th>Terrace</th>
      <th>Type</th>
      <th>SquareMeter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>21.000000</td>
      <td>21.000000</td>
      <td>21.000000</td>
      <td>21.000000</td>
      <td>21.000000</td>
      <td>21.0</td>
      <td>21.000000</td>
      <td>21.000000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>872.380952</td>
      <td>0.666667</td>
      <td>1.095238</td>
      <td>51.428571</td>
      <td>0.904762</td>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.809524</td>
      <td>17.752857</td>
    </tr>
    <tr>
      <th>std</th>
      <td>255.380202</td>
      <td>0.483046</td>
      <td>0.300793</td>
      <td>16.663047</td>
      <td>0.300793</td>
      <td>0.0</td>
      <td>0.483046</td>
      <td>0.402374</td>
      <td>5.034570</td>
    </tr>
    <tr>
      <th>min</th>
      <td>550.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.670000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>650.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>780.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>54.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>17.140000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1050.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>60.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>18.570000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1390.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>85.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>33.100000</td>
    </tr>
  </tbody>
</table>
</div>



Del gráfico de barras podemos destacar que hay más pisos de una habitación que estudios disponibles y que son más los pisos sin terraza que con terraza. Además, las áreas y los precios son muy variados


```python
one_bedroom.hist(bins=50, figsize=(20,15))
plt.show()
```


![png](output_46_0.png)


Por eso, vamos a agrupar los datos en barrios


```python
one_bedroom.groupby('NB')['NB'].count().sort_values(ascending=False)
```




    NB
    Retiro       8
    Salamanca    6
    Chamberí     5
    Chamartín    2
    Name: NB, dtype: int64



Tenemos más pisos disponibles en el barrio del Retiro


```python
sns.catplot(x="NB", kind="count", palette="ch:start=.2,rot=-.3", data=one_bedroom)
```




    <seaborn.axisgrid.FacetGrid at 0x14bc14ae188>




![png](output_50_1.png)


Si miramos las áreas por barrio, detectamos que:
<li>El <strong>Barrio del Retiro</strong> tiene pisos de una media de 42 metros cuadrados, pero áreas que van desde los 25 hasta los 60 metros cuadrados</li>
<li><strong>Salamanca</strong> tiene pisos más grandes, entre 50 y 70m2, con una media de 60m2. Se ve un área atípica de menos de 40m2</li>
<li><strong>Chamberí</strong> tiene pisos entre 30 y 50m2, y su área promedio es de 40m2. Se ve un área atípicade 80m2</li>
<li>Finalmente, <strong>Chamartín</strong> tiene solo 2 pisos, entre 55 y 85 m2</li>


```python
sns.catplot(x="NB", y="Area", kind="box", data=one_bedroom)
```




    <seaborn.axisgrid.FacetGrid at 0x14bc1e18188>




![png](output_52_1.png)


Si hacemos lo mismo con los precios por barrio, podemos apreciar la gran diferencia de precios que tienen los barrios del Retiro y Salamanca, mientras que, salvo valores atípicos, Chamberí y Chamartín tiene un rango más acotado de precios.


```python
sns.catplot(x="NB", y="Price", kind="box", data=one_bedroom)
```




    <seaborn.axisgrid.FacetGrid at 0x14bc1de1a48>




![png](output_54_1.png)


Y ¿qué pasa con el precio por metro cuadrado? Aquí podemos encontrar un poco más de sentido que en los precios al ser una medida que une el precio y el área de los pisos. En cada barrio es distinto, pero los precios van desde los 12€/m2 hasta los 26€/m2.


```python
sns.catplot(x="NB", y="SquareMeter", kind="box", data=one_bedroom)
```




    <seaborn.axisgrid.FacetGrid at 0x14bc16d0848>




![png](output_56_1.png)


El piso al que queremos corroborar el precio, es interior y no tiene terraza, tiene 35 m2, 1 habitación, 1 baño, no tiene parking, el precio por metro cuadrado es 22,85€ y está ubicado en el barrio de Chamartín. Así que vamos a compararlo con pisos de 1 habitación o estudios sin terraza en los barrios ya mencionados.


```python
one_bedroom_no_terrace = one_bedroom.query('(Terrace == 0 and not Terrace.isnull())',engine='python')
```


```python
one_bedroom_no_terrace
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
      <th>Descripción</th>
      <th>Price</th>
      <th>Rooms</th>
      <th>Baths</th>
      <th>Area</th>
      <th>Elevator</th>
      <th>Parking</th>
      <th>Terrace</th>
      <th>Type</th>
      <th>NB</th>
      <th>SquareMeter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Le ofrecemos este bonito y luminoso piso para ...</td>
      <td>850</td>
      <td>1</td>
      <td>1</td>
      <td>54</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>15.74</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Espectacular dúplex - loft de diseño en calle ...</td>
      <td>980</td>
      <td>0</td>
      <td>1</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Chamberí</td>
      <td>12.25</td>
    </tr>
    <tr>
      <th>15</th>
      <td>""Agencia inmobiliaria de MADRID – zona PACIFI...</td>
      <td>600</td>
      <td>1</td>
      <td>1</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>24.00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Magnifico estudio en la Travesía Andrés Mellad...</td>
      <td>600</td>
      <td>0</td>
      <td>1</td>
      <td>38</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Chamberí</td>
      <td>15.79</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Agencia inmobiliaria de MADRID - zona AVDA. CI...</td>
      <td>550</td>
      <td>1</td>
      <td>1</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>16.18</td>
    </tr>
    <tr>
      <th>46</th>
      <td>REDFRIN alquila estudio en calle GUZMAN EL BUE...</td>
      <td>780</td>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Chamberí</td>
      <td>26.00</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Agencia inmobiliaria de MADRID - zona AVDA. CI...</td>
      <td>600</td>
      <td>1</td>
      <td>1</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>17.65</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Precioso apartamento reformado de 42m², a estr...</td>
      <td>1390</td>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>33.10</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Fantástico apartamento seminuevo situado junto...</td>
      <td>750</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Chamberí</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>83</th>
      <td>Vivienda exterior, amueblada con una superfici...</td>
      <td>750</td>
      <td>1</td>
      <td>1</td>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Salamanca</td>
      <td>12.50</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Magnifica vivienda en la calle Fernando el Cat...</td>
      <td>780</td>
      <td>1</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Chamberí</td>
      <td>19.50</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Agencia inmobiliaria de Madrid - zona Avda. Ci...</td>
      <td>600</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Retiro</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>112</th>
      <td>Vivienda exterior, amueblada con una superfici...</td>
      <td>700</td>
      <td>1</td>
      <td>1</td>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Salamanca</td>
      <td>11.67</td>
    </tr>
    <tr>
      <th>120</th>
      <td>""Agencia inmobiliaria de madrid - zona avda. ...</td>
      <td>1100</td>
      <td>1</td>
      <td>2</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Retiro</td>
      <td>18.33</td>
    </tr>
  </tbody>
</table>
</div>



No hay pisos sin terraza disponibles para analizar en el bario de Chamartín, pero podemos ver que en Chamberí y Retiro, hay mucha amplitud en cuanto a precio por metro cuadrado, mientras que en el barrio de Salamanca, la diferencia es menor.


```python
sns.catplot(x="NB", y="SquareMeter", kind="box", data=one_bedroom_no_terrace)
```




    <seaborn.axisgrid.FacetGrid at 0x14bc158c708>




![png](output_61_1.png)


Haremos lo mismo con los pisos con terraza


```python
one_bedroom_terrace = one_bedroom.query('(Terrace == 1 and not Terrace.isnull())',engine='python')
```


```python
one_bedroom_terrace
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
      <th>Descripción</th>
      <th>Price</th>
      <th>Rooms</th>
      <th>Baths</th>
      <th>Area</th>
      <th>Elevator</th>
      <th>Parking</th>
      <th>Terrace</th>
      <th>Type</th>
      <th>NB</th>
      <th>SquareMeter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Moderna y luminosa vivienda totalmente exterio...</td>
      <td>1200</td>
      <td>0</td>
      <td>1</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Salamanca</td>
      <td>17.14</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Fantástico Ático totalmente exterior con una s...</td>
      <td>1000</td>
      <td>1</td>
      <td>1</td>
      <td>56</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Salamanca</td>
      <td>17.86</td>
    </tr>
    <tr>
      <th>36</th>
      <td>""SOLFAI CONSULTING PONE A SU DISPOSICIÓN ESTU...</td>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>57</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Chamartín</td>
      <td>15.61</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Moderna y luminosa vivienda totalmente exterio...</td>
      <td>1300</td>
      <td>0</td>
      <td>1</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Salamanca</td>
      <td>18.57</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Exclusiva vivienda exterior en edificio de nue...</td>
      <td>1200</td>
      <td>1</td>
      <td>1</td>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Retiro</td>
      <td>20.00</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Solfai Consulting pone a su disposición hermos...</td>
      <td>1050</td>
      <td>1</td>
      <td>1</td>
      <td>85</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Chamartín</td>
      <td>12.35</td>
    </tr>
    <tr>
      <th>107</th>
      <td>SIN COMISIÓN DE AGENCIA - GASTOS INCLUIDOS\n\n...</td>
      <td>650</td>
      <td>0</td>
      <td>1</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Salamanca</td>
      <td>18.57</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.catplot(x="NB", y="SquareMeter", kind="box", data=one_bedroom_terrace)
```




    <seaborn.axisgrid.FacetGrid at 0x14bc36abd08>




![png](output_65_1.png)


## Repasando lo que encontramos

<li>Hay más pisos de una habitación que estudios</li>
<li>La mayoría de estos pisos tiene un solo baño</li>
<li>Ninguno de los pisos analizados tiene plaza de garaje incluida, esto puede significar que son zonas en las que los garajes son escasos o que el propietario alquila la plaza de garaje aparte.</li>
<li>El 90% de estos pisos tiene ascensor</li>
<li>No parece haber un patrón en el precio de los pisos, la terraza no parece ser un diferencial en esta base de datos, habría que ahondar más en la decoración de los pisos que puede ser un punto más a favor al momento de elegir el precio adecuado</li>
<li>Sin tomar en cuenta valores atípicos, los precios en estos barrios van desde los 12€/m2(estudios) hasta los 26€/m2.</li>

### ¿Qué precio podría tener mi piso de 1 habitación?
Un dato que debemos tomar en cuenta es que hay una alta oferta de pisos en alquiler. Según datos de Idealists, la cartera de pisos en alquiler en España ha aumentado en un 63%. Así que tomando esto en cuenta, en que finalmente es un piso con una habitación y no estudio, que no tiene ascensor y en la oferta encontrada en la zona, el precio de nuestro pios podría oscilar entre los 525€ (15€/m2) y 700€ (20€/m2).


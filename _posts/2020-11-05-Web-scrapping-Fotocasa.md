Vamos a hacer un web scraping del portal de pisos Fotocasa con Beautiful Soup y Selenium, que son unas librerías de Python muy amigables y sencillas de utilizar. Este ejercicio se puede repetir no solo en páginas de búsquedas de pisos, sino en páginas de distintos comercios, ya sea porque queremos analizar los datos para hacer una compra (o alquiler) inteligente, como para utilizarlos en alguna decisión de nuestro negocio.


```python
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains #al final esta no la usé pero échenle un ojo está guay
from bs4 import BeautifulSoup
import pandas as pd
import requests # for making standard html requests
import re
from random import randint
import time
```

Recuerda que si no tienes instalada alguna de estas librerías, puedes hacerlo desde el mismo jupyter notebook. En mi caso no tenía instalado web_drive manager de Selenium, que me servirá para abrir la página que quiero "scrapear", intentando imitar a cómo lo haría un ser humano.


```python
#!pip install webdriver-manager

```


```python
# driver = webdriver.Chrome(ChromeDriverManager().install())
```

Decidí hacerlo con Chrome, pero también se puede hacer con Firefox, y además le he agregado un tiempo de espera para darle tiempo a la página a cargar


```python
driver_chrome = webdriver.Chrome(executable_path="C:/Users/ksalg/Curso Python Datahack/chromedriver.exe")
driver_chrome.implicitly_wait(20)
```

Creas unas listas vacías de las características que quieres analizar, en mi caso: precios, características(número de habitaciones, baños, etc.), direcciones y descripciones

<img src="karlasz86.github.io/images/WebScraping/Scraping1.PNG">


```python
prices=[] #List to store price of the apartment
features =[] #list to store features
addresses=[] #List to store address
descriptions = [] #list to store descriptions
```

Lo primero que hay que hacer antes de empezar con esto, es inspeccionar la página para detectar tanto el contenedor que tiene todo los datos que queremos almacenar, como los subcontenedores que contienen cada característica a almacenar. Para esto hay que dar a Inspeccionar, y dentro del mismo código veremos lo que necesitamos, esta parte es muy importante, así que dedícale el tiempo que veas necesario.

Por ejemplo, detectamos que el contenedor que contiene toda la información de los pisos es "re-Searchresult-itemrow", que está dentro de un tag <div>

<img src="karlasz86.github.io/images/WebScraping/Scraping2.PNG">

Y luego dentro del contenedor, tenemos que detectar dónde están las características del piso que queremos analizar, y en qué contenedores están, con el inspector es sencillo porque mientras vas apuntando al código, la página se marca sobre la parte que apuntamos:
   <li> **Precio**: Ubicado bajo el tag "span> y la clase "re-Card-Price" </li>
   <li> **Características**: Ubicado bajo el tag "div" y la clase "re-CardFeatures-wrapper" </li>
   <li> **Direcciones**: Ubicado bajo el tag "h3" y la clase "re-Card-title" </li>
   <li> **Descripciones**: Ubicado bajo el tag "span" y la clase "re-Card-description </li>

<img src="karlasz86.github.io/images/WebScraping/Scraping3.PNG">

Llamo a mi url con Selenium y con Beautiful Soup convierto este código de la página en algo más "comestible". Hago un bucle "for" ya que quiero analizar no solo la primer página, sino las primeras 31 páginas. ¿Y cómo encuentro mi contenedor? Podemos utilizar find(). Para evitar que mi scraping se detenga por si se encuentra con un contenedor vacío, le indico que si se encuentra con un "None", continúe.

Llegado a este punto identifiqué 2 problemas, el primero, que tienen un buen detector de spiders, y la página detecta que hay algo que no es normal, a pesar de mis esfuerzos por hacer pausas (con sleep por ejemplo para que tarde entre 1 y 3 segundo en pasar de una página a otra). Así que tengo que hacer el CAPCHA me guste o no.

El segundo problema es que la el contenedor 're-Searchresult-itemRow' que es el que tiene toda la información de cada propiedad, tiene un stopper, a partir de la cuarta propiedad de cada página, solo cuando que dado scroll down y me he mantenido algunos segundo, la información del contenedor aparece. Para no dilatar más el proceso y habiendo detectado que hay muchísimas propiedades duplicas, lo que hice fue hacer web scraping de las primeras 31 páginas durante 5 días consecutivos. No es una página fácil de scrapear, hay otras más sencillas, pero la data que tiene sin tener que entrar a cada propiedad es muy rica y se puede aprovechar mucho.


```python
for page in range(1, 32):
    pisos_url = 'https://www.fotocasa.es/es/alquiler/viviendas/madrid-capital/todas-las-zonas/amueblado/l/'+str(page)+'combinedLocationIds=724%2C14%2C28%2C173%2C0%2C28079%2C0%2C0%2C0&gridType=3&latitude=40.4096&longitude=-3.68624'
        
    driver_chrome.get(pisos_url)
    content = driver_chrome.page_source
    soup = BeautifulSoup(content, 'html.parser')
    flat_containers = soup.find_all('div', attrs={'class':'re-Searchresult-itemRow'})
    for a in flat_containers:
        price=a.find('span', attrs={'class': 're-Card-price'})
        feature=a.find('div', attrs={'class': 're-CardFeatures-wrapper'})
        address=a.find('h3', attrs={'class': 're-Card-title'})
        description=a.find('span', attrs={'class': 're-Card-description'})
        
        if None in (price, feature, address, description):
            continue
        prices.append(price.text)
        features.append(feature.text)
        addresses.append(address.text)
        descriptions.append(description.text)
        time.sleep(randint(1,3))
        
```

He capturado 96 propiedades, así que lo primero que hago es trasladarlas a un dataframe de Pandas.


```python
df_apt = pd.DataFrame({'Precio':prices,'Características':features, 'Dirección':addresses, 'Descripción':descriptions}) 
```


```python
df_apt.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 96 entries, 0 to 95
    Data columns (total 4 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   Precio           96 non-null     object
     1   Características  96 non-null     object
     2   Dirección        96 non-null     object
     3   Descripción      96 non-null     object
    dtypes: object(4)
    memory usage: 3.1+ KB
    

Guardé cada base en un archivo csv con distinto nombre


```python
df_apt.to_csv('precios_varios_scraping_3.csv', index=False, encoding='utf-8')
```

# Segunda parte: Duplicados y guardar mi dataframe

Ahora que hemos obtenido los datos de los pisos, vamos a abrir el csv que acabamos de crear, además, de los csvs que generé los días anteriores para tener una base de datos un poco más robusta:


```python
import csv
path="C:/Users/ksalg/Curso Python Datahack/"
df_pisos=pd.read_csv(path+'precios_varios_scraping_3.csv')
```


```python
df_pisos_2=pd.read_csv(path+'precios_pisos_madrid_2.csv')
df_pisos_3=pd.read_csv(path+'precios_pisos_madrid.csv')
df_pisos_4=pd.read_csv(path+'precios_fotocasa_test.csv')
df_pisos_5=pd.read_csv(path+'precios_varios_scraps.csv')
df_pisos_6=pd.read_csv(path+'precios_varios_scraping.csv')
df_pisos_7=pd.read_csv(path+'precios_varios_scraping_2.csv')

```

Ahora tenemos que unirlos y lo haremos con concat


```python
frames = [df_pisos, df_pisos_2, df_pisos_3, df_pisos_4, df_pisos_5, df_pisos_6, df_pisos_7]
result = pd.concat(frames)
```


```python
result.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1468 entries, 0 to 92
    Data columns (total 4 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   Precio           1468 non-null   object
     1   Características  1468 non-null   object
     2   Dirección        1468 non-null   object
     3   Descripción      1468 non-null   object
    dtypes: object(4)
    memory usage: 57.3+ KB
    

Vamos a empezar por los duplicados, según la información de nuestro dataframe tenemos 1468 propiedades, pero antes de saltar de emoción, ¿realmente tenemos tantos datos? Para eso debemos cerciorarnos de que no haya duplicados.

Para ello utilizaré duplicated(), que marca como `True` todas aquellas filas (registros) que tienen el mismo valor en todos sus campos (marcandolas como `False` en caso contrario). Si quisiéramos solo centrarnos en algún campo en concreto le pasaríamos a `duplicated` una lista con los nombres de los campos a tomar en consideración:  `duplicated('campo1')` ó `duplicated(['campo1','campo2'...])`.


```python
result[result.duplicated(keep='last')]
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
      <th>Precio</th>
      <th>Características</th>
      <th>Dirección</th>
      <th>Descripción</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.200 € /mes</td>
      <td>3 habs.2 baños115 m²con ascensor y terraza</td>
      <td>PisoSan Crispin, Latina</td>
      <td>Magnífica vivienda  amueblada en estilo clase ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>850 € /mes</td>
      <td>1 hab.1 baño54 m²con ascensor</td>
      <td>PisoCalle del Doctor Castelo, 12, Retiro</td>
      <td>Le ofrecemos este bonito y luminoso piso para ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>900 € /mes</td>
      <td>2 habs.2 baños75 m²con ascensor</td>
      <td>PisoCracovia, San Blas</td>
      <td>VOhome Propiedades Las Rosas / San Blas ofrece...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>900 € /mes</td>
      <td>2 habs.2 baños75 m²con ascensor</td>
      <td>PisoCracovia, San Blas</td>
      <td>VOhome Propiedades Las Rosas / San Blas ofrece...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.200 € /mes</td>
      <td>70 m²con ascensor y terraza</td>
      <td>PisoCalle de Eraso, Salamanca</td>
      <td>Moderna y luminosa vivienda totalmente exterio...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>990 € /mes</td>
      <td>2 habs.1 baño51 m²con ascensor</td>
      <td>PisoCalle de Antonio Arias, 12, Retiro</td>
      <td>Disfrute del privilegio de vivir en el emblemá...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1.150 € /mes</td>
      <td>3 habs.1 baño84 m²</td>
      <td>PisoCalle de Santa Fe, Moncloa</td>
      <td>Vivienda XXI alquila piso exterior de 3 dormit...</td>
    </tr>
    <tr>
      <th>80</th>
      <td>990 € /mes</td>
      <td>2 habs.1 baño67 m²con ascensor</td>
      <td>ÁticoCarros, Centro</td>
      <td>CONSORCIO REAL ESTATE ÓPERA ALQUILA última pla...</td>
    </tr>
    <tr>
      <th>84</th>
      <td>1.900 € /mes</td>
      <td>2 habs.3 baños100 m²con ascensor</td>
      <td>PisoCentro</td>
      <td>En una finca histórica con una ubicación inmej...</td>
    </tr>
    <tr>
      <th>89</th>
      <td>1.200 € /mes</td>
      <td>2 habs.2 baños86 m²con ascensor y parking</td>
      <td>PisoCalle de Mozart, Moncloa</td>
      <td>Vivienda XXI alquila piso exterior totalmente ...</td>
    </tr>
  </tbody>
</table>
<p>1346 rows × 4 columns</p>
</div>



Lo siguiente sería eliminar los duplicados detectados, para ello utilizaríamos la sentencia `drop_duplicates()` indicando mediante el parámetro `keep` si queremos preservar la primera aparición (`'first'`, comportamiento por defecto), la última (`'last'`) ó eliminar todo duplicado (`False`). Además, el parámetro `inplace` permitirá que la eliminación de duplicados se haga in situ, sobre el propio dataframe sin necesidad de asignar el resultado de la función a un nuevo dataframe. Este parámetro es característico de todo método de `pandas` que conlleve posibles modificaciones del dataframe.


```python
result.drop_duplicates(inplace=True)
```

Para que no quede ningún hueco en el indexado, vamos a reiniciar el índice con reset_index()


```python
result.reset_index(drop=True, inplace=True)
```

Y como vemos, nos hemos quedado solo con 122 propiedades


```python
result.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 122 entries, 0 to 121
    Data columns (total 4 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   Precio           122 non-null    object
     1   Características  122 non-null    object
     2   Dirección        122 non-null    object
     3   Descripción      122 non-null    object
    dtypes: object(4)
    memory usage: 3.9+ KB
    

## Time to Pickle!

No, no es la hora de la merienda, ya que hemos empleado un tiempo importante en hacer el scraping, organizar los csvs y quitar los duplicados del medio, si el portátil me deja colgada, ¿tengo que empezar todo de nuevo? La buena noticia es que no. Y esto me lo enseñó un profe del Máster (gracias Alejandro) así que lo cito:

*pickle permite guardar un objeto Python (dataframe de pandas, array de numpy, diccionario...) como un fichero binario en tu disco duro. Una vez guardado, ya da igual que tu sesión muera o que reinicies tu máquina. El objeto estará ahí disponible para que lo cargues cuando lo necesite*


```python
# Si existe otro pickle con el mismo nombre, se sobreescribirá
result.to_pickle('result.pickle')
```


```python
# Se lee el pickle
housing = pd.read_pickle('result.pickle')
```


```python
#Comprobamos que todo esté ok
housing.head(3)
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
      <th>Precio</th>
      <th>Características</th>
      <th>Dirección</th>
      <th>Descripción</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.200 € /mes</td>
      <td>3 habs.2 baños115 m²con ascensor y terraza</td>
      <td>PisoSan Crispin, Latina</td>
      <td>Magnífica vivienda  amueblada en estilo clase ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>850 € /mes</td>
      <td>1 hab.1 baño54 m²con ascensor</td>
      <td>PisoCalle del Doctor Castelo, 12, Retiro</td>
      <td>Le ofrecemos este bonito y luminoso piso para ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>900 € /mes</td>
      <td>2 habs.2 baños75 m²con ascensor</td>
      <td>PisoCracovia, San Blas</td>
      <td>VOhome Propiedades Las Rosas / San Blas ofrece...</td>
    </tr>
  </tbody>
</table>
</div>



# Cuarta Parte: Limpieza de datos

Vemos que antes de iniciar nuestro análisis, necesitamos un poco de limpieza de datos, en el precio quitar el símbolo de € y /mes y convertirlo en integer, en características, separar número de habitaciones, baños, área y alguna característica extra, en Dirección tenemos el tipo de vivienda pega a la dirección y también eliminar duplicados.

## Para los precios:

Se puede resolver con strip y replace; además aprovechamos y ya lo convertimos en un integer, para poder utilizarlo como filtro más adelante en nuestro análisis


```python
Precio = []
for price in housing["Precio"]:
    precio = price.strip(" € /mes")
    if precio.find('.'):
        precio = precio.replace(".", "")
        Precio.append(int(precio))
        
housing["Price"] = Precio
```

Echamos un ojo, y, ¡perfecto!


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
      <th>Precio</th>
      <th>Características</th>
      <th>Dirección</th>
      <th>Descripción</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.200 € /mes</td>
      <td>3 habs.2 baños115 m²con ascensor y terraza</td>
      <td>PisoSan Crispin, Latina</td>
      <td>Magnífica vivienda  amueblada en estilo clase ...</td>
      <td>1200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>850 € /mes</td>
      <td>1 hab.1 baño54 m²con ascensor</td>
      <td>PisoCalle del Doctor Castelo, 12, Retiro</td>
      <td>Le ofrecemos este bonito y luminoso piso para ...</td>
      <td>850</td>
    </tr>
    <tr>
      <th>2</th>
      <td>900 € /mes</td>
      <td>2 habs.2 baños75 m²con ascensor</td>
      <td>PisoCracovia, San Blas</td>
      <td>VOhome Propiedades Las Rosas / San Blas ofrece...</td>
      <td>900</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.200 € /mes</td>
      <td>70 m²con ascensor y terraza</td>
      <td>PisoCalle de Eraso, Salamanca</td>
      <td>Moderna y luminosa vivienda totalmente exterio...</td>
      <td>1200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>650 € /mes</td>
      <td>1 hab.1 baño40 m²con ascensor</td>
      <td>PisoRonda de Toledo, 32, Arganzuela</td>
      <td>SOLFAI CONSULTING PONE A SU DISPOSICIÓN PRECIO...</td>
      <td>650</td>
    </tr>
  </tbody>
</table>
</div>



## Para las habitaciones

Para esto se me ocurrió convertir en una lista separada por "hab" a cada elemento de "Características", siendo el primer elemento de la lista el número de habitaciones. Revisando el dataframe, si la palabra "hab" no existe, es porque el piso no tiene habitaciones y es un estudio, por lo que si no encuentra la palabra hab, el resultado será directamente 0. Además convertirmos estos valores en integers para poder hacer nuestro análisis más adelante


```python
Rooms = []
for c in housing["Características"]:
    if c.find("hab"):
        feat = c.replace("hab", ",")
    if 'hab' not in c:
        feat = '0'
    
    Rooms.append(int(feat[0]))
```


```python
housing['Rooms'] = Rooms
```

Comprobamos y ya tenemos el número de habitaciones en nuestro dataframe


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
      <th>Precio</th>
      <th>Características</th>
      <th>Dirección</th>
      <th>Descripción</th>
      <th>Price</th>
      <th>Rooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.200 € /mes</td>
      <td>3 habs.2 baños115 m²con ascensor y terraza</td>
      <td>PisoSan Crispin, Latina</td>
      <td>Magnífica vivienda  amueblada en estilo clase ...</td>
      <td>1200</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>850 € /mes</td>
      <td>1 hab.1 baño54 m²con ascensor</td>
      <td>PisoCalle del Doctor Castelo, 12, Retiro</td>
      <td>Le ofrecemos este bonito y luminoso piso para ...</td>
      <td>850</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>900 € /mes</td>
      <td>2 habs.2 baños75 m²con ascensor</td>
      <td>PisoCracovia, San Blas</td>
      <td>VOhome Propiedades Las Rosas / San Blas ofrece...</td>
      <td>900</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.200 € /mes</td>
      <td>70 m²con ascensor y terraza</td>
      <td>PisoCalle de Eraso, Salamanca</td>
      <td>Moderna y luminosa vivienda totalmente exterio...</td>
      <td>1200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>650 € /mes</td>
      <td>1 hab.1 baño40 m²con ascensor</td>
      <td>PisoRonda de Toledo, 32, Arganzuela</td>
      <td>SOLFAI CONSULTING PONE A SU DISPOSICIÓN PRECIO...</td>
      <td>650</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Para los baños


```python
Baths = []
for c in housing["Características"]:
    if c.find("baño"):
        feat = c.replace(".", ",").split(',')
        if len(feat)>=2:
            feat2=feat[1][0]
        else:
            feat2=1
    if 'baño' not in c:
        feat2=1

    Baths.append(int(feat2))
```


```python
housing['Baths'] = Baths
```

## Area del piso


```python
area = []
for c in housing["Características"]:
    indice = c.find("m²")
    try:
        bla = int(c[indice-4]+c[indice-3]+c[indice-2])
    except ValueError:
        bla= int(c[indice-3]+c[indice-2])
    area.append(bla)
```


```python
housing['Area'] = area
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
      <th>Precio</th>
      <th>Características</th>
      <th>Dirección</th>
      <th>Descripción</th>
      <th>Price</th>
      <th>Rooms</th>
      <th>Baths</th>
      <th>Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.200 € /mes</td>
      <td>3 habs.2 baños115 m²con ascensor y terraza</td>
      <td>PisoSan Crispin, Latina</td>
      <td>Magnífica vivienda  amueblada en estilo clase ...</td>
      <td>1200</td>
      <td>3</td>
      <td>2</td>
      <td>115</td>
    </tr>
    <tr>
      <th>1</th>
      <td>850 € /mes</td>
      <td>1 hab.1 baño54 m²con ascensor</td>
      <td>PisoCalle del Doctor Castelo, 12, Retiro</td>
      <td>Le ofrecemos este bonito y luminoso piso para ...</td>
      <td>850</td>
      <td>1</td>
      <td>1</td>
      <td>54</td>
    </tr>
    <tr>
      <th>2</th>
      <td>900 € /mes</td>
      <td>2 habs.2 baños75 m²con ascensor</td>
      <td>PisoCracovia, San Blas</td>
      <td>VOhome Propiedades Las Rosas / San Blas ofrece...</td>
      <td>900</td>
      <td>2</td>
      <td>2</td>
      <td>75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.200 € /mes</td>
      <td>70 m²con ascensor y terraza</td>
      <td>PisoCalle de Eraso, Salamanca</td>
      <td>Moderna y luminosa vivienda totalmente exterio...</td>
      <td>1200</td>
      <td>0</td>
      <td>1</td>
      <td>70</td>
    </tr>
    <tr>
      <th>4</th>
      <td>650 € /mes</td>
      <td>1 hab.1 baño40 m²con ascensor</td>
      <td>PisoRonda de Toledo, 32, Arganzuela</td>
      <td>SOLFAI CONSULTING PONE A SU DISPOSICIÓN PRECIO...</td>
      <td>650</td>
      <td>1</td>
      <td>1</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



## ¿Ascensor?


```python
elevator = []
for c in housing["Características"]:
    if 'ascensor' not in c:
        feat= 0
    else:
        feat = 1
    elevator.append(feat)
```


```python
housing['Elevator'] = elevator
```

## Parking


```python
parking = []
for c in housing["Características"]:
    if 'parking' not in c:
        feat= 0
    else:
        feat = 1
    parking.append(feat)
```


```python
housing['Parking'] = parking
```


```python
terrace = []
for c in housing["Características"]:
    if 'terraza' not in c:
        feat = 0
    else:
        feat = 1
    terrace.append(feat)
```


```python
housing['Terrace'] = terrace
```

## ¿Piso o Estudio?


```python
tipo = []
for c in housing["Dirección"]:
    if 'Estudio' not in c:
        feat= 1
    else:
        feat = 0
    tipo.append(feat)
```


```python
housing['Type'] = tipo 
```

## Barrio

Importante si luego queremos hacer una segmentación por barrios. Para esto hice una lista de los barrios de Madrid, también podríamos obtener este listado de alguna base de datos externa, por ejemplo si son todos los barrios de todas las ciudades de España, ya que sería demasiado manua y laborioso hacerlo de esta forma.


```python
barrio = []
barrios = ['Arganzuela', 'Barajas', 'Carabanchel', 'Centro', 'Chamartín','Chamberí', 'Ciudad Lineal', 'Fuencarral', 'Hortaleza',
           'Latina', 'Moncloa', 'Puente de Vallecas', 'Retiro', 'Salamanca', 'San Blas', 'Tetúan', 'Usera', 'Vicávaro']
```


```python
for c in housing["Dirección"]:
    for b in barrios:
        if b in c:
            feat=b
    barrio.append(feat)
```


```python
housing['NB'] = barrio
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
      <th>Precio</th>
      <th>Características</th>
      <th>Dirección</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.200 € /mes</td>
      <td>3 habs.2 baños115 m²con ascensor y terraza</td>
      <td>PisoSan Crispin, Latina</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>850 € /mes</td>
      <td>1 hab.1 baño54 m²con ascensor</td>
      <td>PisoCalle del Doctor Castelo, 12, Retiro</td>
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
    </tr>
    <tr>
      <th>2</th>
      <td>900 € /mes</td>
      <td>2 habs.2 baños75 m²con ascensor</td>
      <td>PisoCracovia, San Blas</td>
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
    </tr>
    <tr>
      <th>3</th>
      <td>1.200 € /mes</td>
      <td>70 m²con ascensor y terraza</td>
      <td>PisoCalle de Eraso, Salamanca</td>
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
    </tr>
    <tr>
      <th>4</th>
      <td>650 € /mes</td>
      <td>1 hab.1 baño40 m²con ascensor</td>
      <td>PisoRonda de Toledo, 32, Arganzuela</td>
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
    </tr>
  </tbody>
</table>
</div>



Ahora vamos a eliminar las columnas que no nos sirven


```python
final_result = housing.drop(['Dirección','Características', 'Precio'], axis=1)
```

Y echamos un vistazo, para ver cómo ha quedado nuestro dataframe


```python
final_result.head()
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
    </tr>
  </tbody>
</table>
</div>



## Precio por metro cuadrado

Es importante saber cuál es el precio medio de las propiedades, y ya que tenemos el precio y al área de nuestros pisos, aprovechemos y agreguemos esta columna en una sencilla operación y vamos a darle un formato con 2 decimales.


```python
final_result['SquareMeter'] = final_result['Price']/final_result['Area']
```


```python
final_result['SquareMeter'] = pd.Series([round(val, 2) for val in final_result['SquareMeter']], index = final_result.index)
```


```python
final_result.head()
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



# Cuarta parte

Y ahora, ¡manos a la obra!, ¿qué es lo que queremos averiguar de estos datos? Tenemos: Precio, número de habitaciones, número de baños, si tiene o no ascensor, si tiene o no plaza de garaje si es estudio o piso completo (aunque eso lo podemos deducir también con el número de habitaciones) y el barrio al que pertenece el piso


```python
final_result.info()
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
    

## Revisando nuestro dataset

Con la función describe podemos obteher un resumen estadístico de la distribución de nuestras features numéricas.

Por ejemplo, por sacar datos que puedan interesarnos:
    <li>La media de número de habitaciones es 2. Siendo el mínimo 0 habitaciones (estudios) y el máximo 4.</li>
    <li>La media de número de baños es 1, siendo el mínimo 1 y el máximo 3.</li>
    <li>En promedio el área ronda los 78 metros. El área mínima es 25 metros y la máxima 200</li>
    <li>El precio mínimo es 550€ y el máximo 5000€</li>


```python
final_result.describe()
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



Con la librería matplotlib podemos tener una visión general más amigable


```python
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
```


```python
print("Matplotlib version " + matplotlib.__version__)
```

    Matplotlib version 3.1.3
    

¿Qué podemos destacar más de estas gráficas? Por ejemplo, de que hay más pisos con un baño, de que la mayoría de los pisos tiene ascensor, y la mayoría no tiene parking. También podemos destacar que la mayoría no tiene terraza.


```python
final_result.hist(bins=50, figsize=(20,15))
plt.show()
```


![png](output_102_0.png)


De las gráficas anteriores podemos detectar que hay algunos outliers, sobre todo en los precios, y podemos verlo mejor con un diagrama de cajas, ya luego nos adentraremos más en la base de datos para saber las características de los pisos con estos precios que se alejan mucho del grueso de nuestras propiedades.


```python
final_result['Price'].plot(kind="box",
                figsize=(4,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2ea240523c8>




![png](output_104_1.png)


Ahora, porque es importante para mi análisis, quiero averiguar de todo mi conjunto de datos, qué número de habitaciones es el más común. Eso podemos averiguarlo con una operación sencilla utilizando counts() y dividiendo entre el número de filas de nuestro dataframe. Como ya habíamos visto en la gráfica anterior, hay más pisos de 2 habitaciones, pero porcentualmente ¿cómo se divide esto? 

<li>2 habitaciones: 33%</li>
<li>3 habitaciones: 28%</li>
<li>1 habitación: 25%</li>

Y los "raros" son los estudios y los pisos de 4 habitaciones. Podemos encontrar una oportunidad en los pisos de 4 habitaciones, así por ejemplo, si tenemos un piso a un precio que esté dentro de la media y con una buena decoración, podemos ser más firmes a la hora de la negociación con nuestro potencial inquilino (previo averiguar las características de la competencia): no hay tantos pisos de 4 habitaciones y si son 4 amigos buscando piso, no les va a encajar un piso de 3 así esté tirado de precio. Por supuesto habrá que analizar también otros factores, así como hay pocos pisos disponibles con 4 habitaciones, ¿hay también una baja demanda de este tipo de inmuebles?


```python
final_result['Rooms'].value_counts() / len(final_result)
```




    2    0.336066
    3    0.286885
    1    0.254098
    0    0.065574
    4    0.057377
    Name: Rooms, dtype: float64



## Quiero filtrar mis datos

Supongamos que tenemos un piso de una habitación en el barrio del Viso y no sabemos qué precio podría estar bien para no quedarnos cortos pero tampoco pasarnos dos pueblos y que no lo quiera nadie. Para eso podemos evaluar los pisos de 1 habitación y estudios, no solo en El Viso, sino en todo el barrio de Chamartín y sus alrededores (Salamanca, Retiro y Chamberí), ¿por qué? porque todos estos barrios están muy bien comunicados con el Viso, son muy similares en calidad y si mi piso lo alquilo a 700€, pero la media por pisos de 1 habitación en el barrio de Salamanca es de 600€ (un barrio que se considera más caro), pues ya voy mal.


```python
habs = 1
```


```python
final_result.query('(Rooms <= @habs and not Rooms.isnull()) and ((NB == "Chamberí") | (NB == "Chamartín") | (NB == "Retiro") | (NB == "Salamanca"))',engine='python')
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



Como ven podemos sacar muchísimo provecho a nuestros datos obtenidos con web scraping y con una buena limpieza en pandas y utilizando librerías como matplotlib podemos hacer magia. En una próxima entrega haré un reporte con estos datos creado con Jupyter Notebook y utilizando Python. Espero que les sirva de algo :)

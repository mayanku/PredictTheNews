# PredictTheNews
Competition to predict the type of news. The competition is on Machine Hack platform. 

```python
import pandas as pd
import numpy as np
```


```python
df_train=pd.read_excel("E:/Datasets/PredictTheNews/Data_Train.xlsx")
```


```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
```


```python
df_train.head()
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
      <th>STORY</th>
      <th>SECTION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>But the most painful was the huge reversal in ...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>How formidable is the opposition alliance amon...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Most Asian currencies were trading lower today...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>If you want to answer any question, click on ‘...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>In global markets, gold prices edged up today ...</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
nltk.download('stopwords') #downloading the stopwords from nltk
corpus = [] # List for storing cleaned data
ps = PorterStemmer() #Initializing object for stemming
for i in range(len(df_train)): # for each obervation in the dataset
   #Removing special characters
   text = re.sub('[^a-zA-Z]', ' ', df_train['STORY'][i]).lower().split()
   #Stemming and removing stop words
   text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
   #Joining all the cleaned words to form a sentence
   text = ' '.join(text)
   #Adding the cleaned sentence to a list
   corpus.append(text)
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\Mayank\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
cv = CountVectorizer(max_features = 120)
X = cv.fit_transform(corpus).toarray()
y = df_train.iloc[:, 1].values
```


```python
df_test=pd.read_excel("E:/Datasets/PredictTheNews/Data_Test.xlsx")
```


```python
nltk.download('stopwords') #downloading the stopwords from nltk
corpus1 = [] # List for storing cleaned data
ps = PorterStemmer() #Initializing object for stemming
for i in range(len(df_test)): # for each obervation in the dataset
   #Removing special characters
   text = re.sub('[^a-zA-Z]', ' ', df_test['STORY'][i]).lower().split()
   #Stemming and removing stop words
   text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
   #Joining all the cleaned words to form a sentence
   text = ' '.join(text)
   #Adding the cleaned sentence to a list
   corpus1.append(text)
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\Mayank\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
x_test=cv.fit_transform(corpus1).toarray()
```


```python
x_test
```




    array([[0, 0, 0, ..., 0, 0, 3],
           [0, 0, 0, ..., 0, 0, 0],
           [3, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [1, 0, 0, ..., 1, 0, 0],
           [0, 0, 0, ..., 1, 0, 0]], dtype=int64)




```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state = 0)
```


```python
classifier = SVC()
classifier.fit(X_train, y_train)
```

    C:\Users\Mayank\Anaconda3\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
        kernel='rbf', max_iter=-1, probability=False, random_state=None,
        shrinking=True, tol=0.001, verbose=False)




```python
y_predict = classifier.predict(x_test)
```


```python
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements
```


```python
cm = confusion_matrix(y_val, y_pred)
```


```python
print("Accuracy : ", accuracy(cm))
```

    Accuracy :  0.8813892529488859
    


```python
X_val
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 3, 1, 1],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 1, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 2, 2],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int64)




```python
y_predict
```




    array([0, 2, 0, ..., 0, 1, 1], dtype=int64)




```python
submission = pd.DataFrame({
        "SECTION": y_predict
    })
submission.to_csv('submission_news.csv', encoding='utf-8', index=False)
```


```python

```


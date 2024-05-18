```python
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import  matplotlib.pyplot as plt

```


```python
df = pd.read_csv("./insurance.csv")
df.tail()
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>expenses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1333</th>
      <td>50</td>
      <td>male</td>
      <td>31.0</td>
      <td>3</td>
      <td>no</td>
      <td>northwest</td>
      <td>10600.55</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>18</td>
      <td>female</td>
      <td>31.9</td>
      <td>0</td>
      <td>no</td>
      <td>northeast</td>
      <td>2205.98</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>18</td>
      <td>female</td>
      <td>36.9</td>
      <td>0</td>
      <td>no</td>
      <td>southeast</td>
      <td>1629.83</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>21</td>
      <td>female</td>
      <td>25.8</td>
      <td>0</td>
      <td>no</td>
      <td>southwest</td>
      <td>2007.95</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>61</td>
      <td>female</td>
      <td>29.1</td>
      <td>0</td>
      <td>yes</td>
      <td>northwest</td>
      <td>29141.36</td>
    </tr>
  </tbody>
</table>
</div>




```python
import warnings
warnings.filterwarnings("ignore")

```


```python
sns.lmplot(x="bmi",y="expenses",data = df)
```




    <seaborn.axisgrid.FacetGrid at 0x214b69ecc50>




    
![png](linearregression_files/linearregression_3_1.png)
    



```python
df.describe()
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
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>expenses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.207025</td>
      <td>30.665471</td>
      <td>1.094918</td>
      <td>13270.422414</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.049960</td>
      <td>6.098382</td>
      <td>1.205493</td>
      <td>12110.011240</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>16.000000</td>
      <td>0.000000</td>
      <td>1121.870000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>26.300000</td>
      <td>0.000000</td>
      <td>4740.287500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>30.400000</td>
      <td>1.000000</td>
      <td>9382.030000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>51.000000</td>
      <td>34.700000</td>
      <td>2.000000</td>
      <td>16639.915000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>64.000000</td>
      <td>53.100000</td>
      <td>5.000000</td>
      <td>63770.430000</td>
    </tr>
  </tbody>
</table>
</div>



check for missing value


```python
sns.heatmap(df.isnull(),cbar = False,cmap = "hot")
plt.title("missing value in the dataset")
```




    Text(0.5, 1.0, 'missing value in the dataset')




    
![png](linearregression_files/linearregression_6_1.png)
    



```python
# Select columns with numerical values
numerical_columns = df.select_dtypes(include=['number'])
corr = numerical_columns.corr()
sns.heatmap(corr,cmap = "summer",annot = True)
```




    <Axes: >




    
![png](linearregression_files/linearregression_7_1.png)
    


Data Preprocessing
Encoding
Machine learning algorithms cannot work with categorical data directly, categorical data must be converted to number.

Label Encoding
One hot encoding
Dummy variable trap

By using pandas get_dummies function we can do all above three step in line of code. We will this fuction to get dummy variable for sex, children,smoker,region features. By setting drop_first =True function will remove dummy variable trap by droping one variable and original variable.The pandas makes our life easy.



```python
#dummy variable
catagorical_columns = ["sex","children","smoker","region"]
df_encode = pd.get_dummies(data=df,columns = catagorical_columns,drop_first = True,dtype = 'int8')
df_encode
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
      <th>age</th>
      <th>bmi</th>
      <th>expenses</th>
      <th>sex_male</th>
      <th>children_1</th>
      <th>children_2</th>
      <th>children_3</th>
      <th>children_4</th>
      <th>children_5</th>
      <th>smoker_yes</th>
      <th>region_northwest</th>
      <th>region_southeast</th>
      <th>region_southwest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>27.9</td>
      <td>16884.92</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>33.8</td>
      <td>1725.55</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>33.0</td>
      <td>4449.46</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>22.7</td>
      <td>21984.47</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>28.9</td>
      <td>3866.86</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>1333</th>
      <td>50</td>
      <td>31.0</td>
      <td>10600.55</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>18</td>
      <td>31.9</td>
      <td>2205.98</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>18</td>
      <td>36.9</td>
      <td>1629.83</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>21</td>
      <td>25.8</td>
      <td>2007.95</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>61</td>
      <td>29.1</td>
      <td>29141.36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1338 rows × 13 columns</p>
</div>



Box -Cox transformation


```python
from scipy.stats import boxcox
y_bc,lam,ci = boxcox(df_encode['expenses'],alpha = 0.05)
ci,lam
```




    ((-0.011402950284988304, 0.09880965012231949), 0.04364902969059508)



log transform


```python

df_encode['expenses'] = np.log(df_encode['expenses'])

```

Train Test split


```python
from sklearn.model_selection import train_test_split
X = df_encode.drop('expenses',axis=1)
Y = df_encode['expenses']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=23)



```


```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)
sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)

```


```python
Model evaluation
```


```python
y_pred_sk = lin_reg.predict(X_test)
from sklearn.metrics import mean_squared_error
j_mse_sk = mean_squared_error(y_pred_sk,Y_test)
R_square_sk = lin_reg.score(X_test,Y_test)
print('The Mean Square Error(MSE) or J(theta) is: ',j_mse_sk)
print('R square obtain for scikit learn library is :',R_square_sk)

```

    The Mean Square Error(MSE) or J(theta) is:  0.1767305468975296
    R square obtain for scikit learn library is : 0.8027634107495771


The model returns  𝑅2
  value of 77.95%, so it fit our data test very well, but still we can imporve the the performance of by diffirent technique. Please make a note that we have transformer out variable by applying natural log. When we put model into production antilog is applied to the equation
  .

## Model Validation
In order to validated model we need to check few assumption of linear regression model. The common assumption for *Linear Regression* model are following
1. Linear Relationship: In linear regression the relationship between the dependent and independent variable to be *linear*. This can be checked by scatter ploting Actual value Vs Predicted value
2. The residual error plot should be *normally* distributed.
3. The *mean* of *residual error* should be 0 or close to 0 as much as possible
4. The linear regression require all variables to be multivariate normal. This assumption can best checked with Q-Q plot.
5. Linear regession assumes that there is little or no *Multicollinearity in the data. Multicollinearity occurs when the independent variables are too highly correlated with each other. The variance inflation factor *VIF* identifies correlation between independent variables and strength of that correlation. $\mathbf{VIF = \frac {1}{1-R^2}}$, If VIF >1 & VIF <5 moderate correlation, VIF < 5 critical level of multicollinearity.
6. Homoscedasticity: The data are homoscedastic meaning the residuals are equal across the regression line. We can look at residual Vs fitted value scatter plot. If heteroscedastic plot would exhibit a funnel shape pattern. pattern.

---
jupyter:
  colab:
    name: Project 2.ipynb
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="6LlLryJhGSSe"}
# Machine Learning - Nonparametric Models

## Masaccio Braun

Linear models are a useful means of showing the relationship between two
quantities, but oftentimes a linear model is not sufficient to make
novel and accurate predictions. As such, we want to discover trends
without assuming linearity. But how are we to transition from a straight
line to a smooth curve? The answer is that a smooth curve can be
interpreted as collection of many small lines.

### Locally Weighted Regression - Loess

One way of interpreting this collection of many small lines is through
loess, which is form of nonparametric regression. Although there are
local parametric assumptions, such that local predictions can be linear
but global predictions are nonlinear.
:::

::: {.cell .markdown id="7fiCWeoHGThY"}
### Linear Regression

The main idea of linear regression is the assumption that:

![Equation
1](vertopal_57658d7995a642f7a7c38170f561711d/05d77a00dfbcb1ac01e01df1d5e667b8762406e6.jpg)

\...where *y* is our target dependent variable and *X* is our predictive
independent variable(s).

If we pre-multiply this equation with a matrix of weights we get:

![Equation
2](vertopal_57658d7995a642f7a7c38170f561711d/e5d00dca6277bd4047f66bcf535ab0f433600d72.jpg)

The independent observations are the rows of the matrix *X*. Each row
has a number of columns (this is the number of features) and we can
denote it by *p*. The distance between two data points or independent
observations is the Euclidean distance between the two represented
*p-*dimensional vectors. The equation is:

![Equation
3](vertopal_57658d7995a642f7a7c38170f561711d/030df7bf2b7a01945222cd69a194c4ca80c84eb8.jpg)

We shall have *i* different weight vectors because we have $n$ different
observations.

All in all, linear regression can be seen as a linear combination of the
observed outputs (values of the dependent variable) and the predictions
we make are linear combinations of the actual observed values of the
dependent variables.

### Comparision with loess

So for loess, $\hat{y}$ (our prediction) is obtained as a different
linear combination of the values of $y$. The loess model does not learn
a fixed set of parameters ($\beta$) like linear regression does.
Instead, parameters are determined for each individual $x$. While
$\beta$ is calculated, larger weights are given to the points in the
training set lying closer to $x$ than to the points lying farther away
from $x$.

### Random Forest Regression

The random forest regression is an ensemble bagging algorithm that
combines and averages the outputs of multiple decision trees. Decision
trees are nonlinear computational structures that can be likened to a
flowchart. The algorithm \"learns\" data by splitting it into subsets or
\"leaves\" based on specific criteria for each attribute or prediction
variable, which takes place at a \"node\". The data scientist decides
the criteria for the model This process is repeated for each newly split
subset through what is called \"recursive partitioning\" (method by
which these criteria are used to determine dichotomous outcomes without
the assumption of linearity). This process terminates when the subset at
a node predicts the same value as the target value; however, the process
will also terminate if continued splitting does not increase the
accuracy of the predictions. Despite this, decision trees have a
propensity to overfit the data.

**Decision Tree Diagram**

![Decision Tree
Diagram](vertopal_57658d7995a642f7a7c38170f561711d/aa0c5f8e6f74ac24f77171f972a19f96478fa0b5.jpg)

Source:
<https://stackoverflow.com/questions/57005265/comprehending-large-decision-tree-diagram-for-variable-selection>

One set of models, known as ensemble learning methods, solve this
problem by aggregating multiple learned models to identify the best one.
One of the ways it does this is through a process called \"bagging\",
which is the core framework of random forest regression, an uncorrelated
set of decision trees. The algorithm utilizes bagging by generating a
random subset of features or attributes to ensure a low correlation
coefficient among the decision trees. In other words, because decision
trees consider all possible attribute splits, they tend to have very
high correlation coefficients and become overfit. Random forests solve
this problem by selecting only a random subset of the attributes and
averaging them. Because random forests are an ensemble of nonparametric
decision trees, they are also nonparametric models, and stand to compete
with loess.

**Random Forest Diagram**

![Random Forest
Diagram](vertopal_57658d7995a642f7a7c38170f561711d/e073a7544d28a61b36677211c2b99940eec5664a.jpg)

Source: <https://www.ibm.com/cloud/learn/random-forest>
:::

::: {.cell .markdown id="JjJzHBR-oVNF"}
To compare the effectiveness of each nonlinear model, I will be
performing a univariate analysis using each model on the Boston Housing
Prices.csv dataset, which lists the median house prices (categorical
dependent variable) of many homes in the Boston area, along with 16
predictor attributes (4 categorical and 12 numerical independent
variables). I will be performing a univariate analysis of the \'rooms\'
predictor (number of rooms) on the \'cmedv\' target.
:::

::: {.cell .code id="bns5jPn7JYbr"}
``` {.python}
# Import libraries and models
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import linalg
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
```
:::

::: {.cell .code id="uvJcOKMAJj06"}
``` {.python}
# High-resolution images
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
mpl.rcParams['figure.dpi'] = 120
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="u3GWUujIJj7q" outputId="f2a72f47-769c-49af-ae93-11e909a8b30c"}
``` {.python}
# Mount Google Drive
from google.colab import files
from google.colab import drive
drive.mount('/content/drive')
```

::: {.output .stream .stdout}
    Mounted at /content/drive
:::
:::

::: {.cell .code colab="{\"height\":270,\"base_uri\":\"https://localhost:8080/\"}" id="olrWr0sdK938" outputId="92dbb97e-c82f-4889-d09a-4d5ba6341c0c"}
``` {.python}
# Load data
data = pd.read_csv('drive/MyDrive/Data/Boston Housing Prices.csv')
data.head(5)
```

::: {.output .execute_result execution_count="4"}
```{=html}
  <div id="df-18782c43-73fe-44e2-9fda-675572774030">
    <div class="colab-df-container">
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
      <th>town</th>
      <th>tract</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>crime</th>
      <th>residential</th>
      <th>industrial</th>
      <th>river</th>
      <th>nox</th>
      <th>rooms</th>
      <th>older</th>
      <th>distance</th>
      <th>highway</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>lstat</th>
      <th>cmedv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nahant</td>
      <td>2011</td>
      <td>-70.955002</td>
      <td>42.255001</td>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>no</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.199997</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.300000</td>
      <td>4.98</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Swampscott</td>
      <td>2021</td>
      <td>-70.949997</td>
      <td>42.287498</td>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>no</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.900002</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.799999</td>
      <td>9.14</td>
      <td>21.600000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Swampscott</td>
      <td>2022</td>
      <td>-70.935997</td>
      <td>42.283001</td>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>no</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.099998</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.799999</td>
      <td>4.03</td>
      <td>34.700001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Marblehead</td>
      <td>2031</td>
      <td>-70.928001</td>
      <td>42.292999</td>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>no</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.799999</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.700001</td>
      <td>2.94</td>
      <td>33.400002</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Marblehead</td>
      <td>2032</td>
      <td>-70.921997</td>
      <td>42.298000</td>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>no</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.200001</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.700001</td>
      <td>5.33</td>
      <td>36.200001</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-18782c43-73fe-44e2-9fda-675572774030')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-18782c43-73fe-44e2-9fda-675572774030 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-18782c43-73fe-44e2-9fda-675572774030');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code colab="{\"height\":848,\"base_uri\":\"https://localhost:8080/\"}" id="G5yYLOaoK9-H" outputId="521f217d-94bb-4e7b-ffad-6a5573fae12a"}
``` {.python}
# Select variables and plot data
x = data['rooms'].values
y = data['cmedv'].values

# Data plot
fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlim(3, 9)
ax.set_ylim(0, 51)
ax.scatter(x=x, y=y,s=25)
ax.set_xlabel('Number of Rooms',fontsize=12)
ax.set_ylabel('House Price (Thousands of Dollars)',fontsize=12)
ax.set_title('Boston Housing Prices',fontsize=14)
ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
ax.minorticks_on()
plt.savefig('RoomVsCMedV.png')
files.download('RoomVsCMedV.png')
plt.show()
```

::: {.output .display_data}
    <IPython.core.display.Javascript object>
:::

::: {.output .display_data}
    <IPython.core.display.Javascript object>
:::

::: {.output .display_data}
![](vertopal_57658d7995a642f7a7c38170f561711d/897c6d9f98a7d44f9c52e5714af2b5ec8efd9aeb.png){height="831"
width="1018"}
:::
:::

::: {.cell .markdown id="rEE5097Ec7O5"}
As we can see from the scatterplot, though in an overall sense, as the
number of rooms increases the median house price tends to increase, the
data is decidedly nonlinear. Because of this, it will likely be the case
that our linear model will make relatively inaccurate predictions.
:::

::: {.cell .code id="tzNINy57KNJh"}
``` {.python}
# Define data standardization and cross-validation methods
k = 10

ss = StandardScaler()
kf = KFold(n_splits=k, shuffle=True, random_state=410)
```
:::

::: {.cell .code id="RxGHIr6NKNPX"}
``` {.python}
# Define imported model execution function
def DoKFold(model, x , y, scaler=ss, split=kf):
  pred_mse = []

  for idxTrain, idxTest in kf.split(x, y):
    xtrain = ss.fit_transform(x[idxTrain].reshape(-1,1))
    xtest = ss.transform(x[idxTest].reshape(-1,1))
    ytrain = y[idxTrain]
    ytest = y[idxTest]

    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)

    pred_mse.append(mse(ytest, ypred))

  A = np.column_stack([xtest,ypred])
  A = A[np.argsort(A[:,0])]
        
  return A, xtest, ytest, pred_mse
```
:::

::: {.cell .code id="_5hsyq4WW5pU"}
``` {.python}
# Linear model

ols = LinearRegression()
```
:::

::: {.cell .code colab="{\"height\":475,\"base_uri\":\"https://localhost:8080/\"}" id="DscSdOTTYGTO" outputId="6425cbb3-f5e7-48f6-b5ef-725ca73ff2dd"}
``` {.python}
# Ordinary Least Squares
L, xtest_ols, ytest_ols, mse_ols = DoKFold(ols, x, y)

# Model plot
plt.scatter(xtest_ols, ytest_ols, color='blue', alpha=.5, edgecolors='k')
plt.plot(L[:,0], L[:,1], color='red', lw=2, label='Ordinary Least Squares')
plt.title('Linear Model')
plt.legend()
plt.show()

print('The MSE for the Ordinary Least Squares Linear Regression is: ' + str(np.mean(mse_ols)))
```

::: {.output .display_data}
![](vertopal_57658d7995a642f7a7c38170f561711d/17e01b41a295c12b5d1a30fb3f18de079101befe.png){height="440"
width="614"}
:::

::: {.output .stream .stdout}
    The MSE for the Ordinary Least Squares Linear Regression is: 44.28641009426175
:::
:::

::: {.cell .markdown id="jm5k7u3rdWtq"}
Certainly the linear model reflects an overall trend, but we want to
create a model that is capable of predicting a greater portion of the
median house prices based on the number of rooms with greater
specificity, such that it is better fit to the data on which we will
train it.
:::

::: {.cell .code colab="{\"height\":492,\"base_uri\":\"https://localhost:8080/\"}" id="VnZL6WoNYK6E" outputId="79787906-8993-413a-a566-5c603a810f6c"}
``` {.python}
# Random Forest Regression
num_est = np.arange(30, 51, 1)
max_depth = 3
min_split = 2

te_num_est = []
mse_rfr = []

for n in num_est:
  rfr = RandomForestRegressor(n_estimators=n, max_depth=max_depth, 
                              min_samples_split=min_split)
  F, xtest_rfr, ytest_rfr, yhat_mse_rfr = DoKFold(rfr, x, y, 'Random Forest', 'Ensemble')
    
  mse_rfr.append(np.mean(yhat_mse_rfr))
  te_num_est.append(n)

idx_min_rfr = np.argmin(mse_rfr)

# Model plot
plt.scatter(xtest_rfr, ytest_rfr, color='blue', alpha=.5, edgecolors='k')
plt.plot(F[:,0], F[:,1], color='red', lw=2, label='Random Forest')
plt.title('Ensemble Model')
plt.legend()
plt.show()

print('Optimal number of estimator trees in range tested: ', te_num_est[idx_min_rfr])  
print('The MSE for the Random Forest Regression is: ', mse_rfr[idx_min_rfr])
```

::: {.output .display_data}
![](vertopal_57658d7995a642f7a7c38170f561711d/5cdd4ede4688647935aed9ee29835156133fb2cd.png){height="440"
width="614"}
:::

::: {.output .stream .stdout}
    Optimal number of estimator trees in range tested:  45
    The MSE for the Random Forest Regression is:  35.51087903147884
:::
:::

::: {.cell .markdown id="o95L334DduQX"}
The ensemble model certainly predicts the the trend of the relationship
between the number of rooms and median house price with greater
precision. But since the data, though nonlinear, has undertones of
linearity, I suspect loess will be a better model.
:::

::: {.cell .markdown id="4UFxxaBDeYpR"}
Perhaps the most important aspect of loess is the weights. I will test 3
different kernelling methods for loess: tricubic, Epanechnikov, and
quartic. Each kernel has a specific bandwith with which a given
neigborhood of data points will be interpolated.
:::

::: {.cell .code id="ucrC__W2KM7a"}
``` {.python}
# Define kernels

# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Epanechnikov Kernel
def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2)) 

# Quartic Kernel
def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2) 
```
:::

::: {.cell .markdown id="S6YHeeztenzP"}
This animation is from the scikit-lego documentation.

```{=html}
<figure>
<center>
<img src='https://drive.google.com/uc?id=1bQmo-j35etyEWt7Ce8TSo01YSOhZQBeY'width='800px'/>
<figcaption>Example of how weights work</figcaption></center>
</figure>
```
As the model runs, groupings of data points are weighted according to
the progression of the weight function at the given time.
:::

::: {.cell .code id="zuPzMGxlK9kd"}
``` {.python}
# Smoother model

def lwr(x, y, xnew, kern, tau):
    n = len(x)
    yest = np.zeros(n)   
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     

    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)], [np.sum(weights * x), np.sum(weights * x * x)]])
        beta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = beta[0] + beta[1] * x[i]

    f = interp1d(x, yest, fill_value='extrapolate')

    return f(xnew)
```
:::

::: {.cell .code colab="{\"height\":492,\"base_uri\":\"https://localhost:8080/\"}" id="2MM-g0RMMUDV" outputId="2b23e1ac-480a-4911-9809-26e93152018c"}
``` {.python}
# Locally Weighted Regression

# Tricubic
tau = np.arange(.01, .11, .01)

mse_lwr_tri = []
te_tau_tri = []

for idxtrain,idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = ss.fit_transform(xtrain.reshape(-1,1))
  xtrain = xtrain.ravel()

  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = ss.transform(xtest.reshape(-1,1))
  xtest = xtest.ravel()


  for t in tau:
    yhat_lwr_tri = lwr(xtrain,ytrain,xtest,Tricubic,t)
    mse_lwr_tri.append(mse(ytest, yhat_lwr_tri))
    te_tau_tri.append(t)

  T = np.column_stack([xtest,yhat_lwr_tri])
  T = T[np.argsort(T[:,0])]

idx_min_lwr = np.argmin(mse_lwr_tri)

# Model plot 
plt.scatter(xtest, ytest, color='blue', alpha=.5, edgecolors='k')
plt.plot(T[:,0], T[:,1], color='red', lw=2, label='Loess')
plt.legend()
plt.title('Loess with Tricubic Kernel')
plt.show()

print('Optimal Tau in range tested: ', te_tau_tri[idx_min_lwr])
print('The MSE for the Locally Weighted Regression is: ', mse_lwr_tri[idx_min_lwr])
```

::: {.output .display_data}
![](vertopal_57658d7995a642f7a7c38170f561711d/2bab8b51b417cd27daa8fee0767be6ae0cdfcbbd.png){height="440"
width="614"}
:::

::: {.output .stream .stdout}
    Optimal Tau in range tested:  0.09999999999999999
    The MSE for the Locally Weighted Regression is:  15.580597196050634
:::
:::

::: {.cell .code colab="{\"height\":492,\"base_uri\":\"https://localhost:8080/\"}" id="ZarY-skm11Lz" outputId="1bbdd451-c640-4354-fe82-a481da98695f"}
``` {.python}
# Locally Weighted Regression

# Epanechnikov

tau = np.arange(.01, .11, .01)

mse_lwr_epa = []
te_tau_epa = []

for idxtrain,idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = ss.fit_transform(xtrain.reshape(-1,1))
  xtrain = xtrain.ravel()

  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = ss.transform(xtest.reshape(-1,1))
  xtest = xtest.ravel()


  for t in tau:
    yhat_lwr_epa = lwr(xtrain,ytrain,xtest,Epanechnikov,t)
    mse_lwr_epa.append(mse(ytest, yhat_lwr_epa))
    te_tau_epa.append(t)

  E = np.column_stack([xtest,yhat_lwr_epa])
  E = E[np.argsort(E[:,0])]

idx_min_lwr = np.argmin(mse_lwr_epa)
# Model plot
plt.scatter(xtest, ytest, color='blue', alpha=.5, edgecolors='k')
plt.plot(E[:,0], E[:,1], color='red', lw=2, label='Loess')
plt.legend()
plt.title('Loess with Epanechnikov Kernel')
plt.show()

print('Optimal Tau in range tested: ', te_tau_epa[idx_min_lwr])
print('The MSE for the Locally Weighted Regression is: ', mse_lwr_epa[idx_min_lwr])
```

::: {.output .display_data}
![](vertopal_57658d7995a642f7a7c38170f561711d/47bbb7ee53b9b0519bc260c3049f03cc7cfbf811.png){height="440"
width="614"}
:::

::: {.output .stream .stdout}
    Optimal Tau in range tested:  0.09
    The MSE for the Locally Weighted Regression is:  15.660251150735611
:::
:::

::: {.cell .code colab="{\"height\":492,\"base_uri\":\"https://localhost:8080/\"}" id="KoM_I-tY11am" outputId="b8079997-7dc6-4836-a6d4-39c2bd8e6671"}
``` {.python}
# Locally Weighted Regression

# Quartic
tau = np.arange(.01, .11, .01)

mse_lwr_qua = []
te_tau_qua = []

for idxtrain,idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = ss.fit_transform(xtrain.reshape(-1,1))
  xtrain = xtrain.ravel()

  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = ss.transform(xtest.reshape(-1,1))
  xtest = xtest.ravel()


  for t in tau:
    yhat_lwr_qua = lwr(xtrain,ytrain,xtest, Quartic,t)
    mse_lwr_qua.append(mse(ytest, yhat_lwr_qua))
    te_tau_qua.append(t)
  
  Q = np.column_stack([xtest,yhat_lwr_qua])
  Q = Q[np.argsort(Q[:,0])]

idx_min_lwr = np.argmin(mse_lwr_qua)
# Model plot
plt.scatter(xtest, ytest, color='blue', alpha=.5, edgecolors='k')
plt.plot(Q[:,0], Q[:,1], color='red', lw=2, label='Loess')
plt.legend()
plt.title('Loess with Quartic Kernel')
plt.show()

print('Optimal Tau in range tested: ', te_tau_qua[idx_min_lwr])
print('The MSE for the Locally Weighted Regression is: ', mse_lwr_qua[idx_min_lwr])
```

::: {.output .display_data}
![](vertopal_57658d7995a642f7a7c38170f561711d/7b2e1005f8f1d70ff30e2e2712c0e807c98900f2.png){height="440"
width="614"}
:::

::: {.output .stream .stdout}
    Optimal Tau in range tested:  0.09999999999999999
    The MSE for the Locally Weighted Regression is:  15.684994438470044
:::
:::

::: {.cell .markdown id="kjC3JIlNQExE"}
## Comparison between the Locally Weighted Regression and the Random Forest Regression

Both the locally weighted regression and the random forest regression
perform marginally better than the ordinary least squares linear
regression which had a RMSE of 6.65, which is expected since the
relationship between the two variables is nonlinear. However, between
the two nonparametric models, loess performed much better than the
random forest with a RMSE of 3.95, while the random forest had a RMSE of
5.96. Between each of the three kernels I tested, the tricubic performed
the best, however the differences in the MSE were $\pm$ 0.06, leaving
the differences in the RMSE essentially negligible; all three are
equally effective.
:::

::: {.cell .code id="nfQOBObZavQw"}
``` {.python}
```
:::

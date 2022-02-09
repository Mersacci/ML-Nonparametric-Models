# Efficacy of Non-Parametric Regression Methods

## Locally Weighted Regression vs. Random Forest Regression

``` Python
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


# High-resolution images
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
mpl.rcParams['figure.dpi'] = 120


# Mount Google Drive
from google.colab import files
from google.colab import drive
drive.mount('/content/drive')
```

The following animations are from the scikit-lego documentation:

<figure>
<center>
<img src='https://drive.google.com/uc?id=1bQmo-j35etyEWt7Ce8TSo01YSOhZQBeY'width='800px'/>
<figcaption>Example of how weights work</figcaption></center>
</figure>


``` Python
# Define kernels

# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2))

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2)
```



``` Python
# Define data standardization and cross-validation methods
k = 10

ss = StandardScaler()
kf = KFold(n_splits=k, shuffle=True, random_state=410)


# Define model execution function
def DoKFold(model, x , y, k, scaler=ss, split=kf):
  train_acc = []
  test_acc = []
  pred_mse = []

  for idxTrain, idxTest in kf.split(x, y):
    xtrain = ss.fit_transform(x[idxTrain].reshape(-1,1))
    xtest = ss.transform(x[idxTest].reshape(-1,1))
    ytrain = y[idxTrain]
    ytest = y[idxTest]

    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)

    train_acc.append(model.score(xtrain, ytrain))
    test_acc.append(model.score(xtest, ytest))
    pred_mse.append(mse(ytest, ypred))

  return train_acc, test_acc, pred_mse
```



``` Python
# Linear model
ols = LinearRegression()
```


``` Python
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


# Ensemble model
num_est = np.arange(80, 121, 1)
max_depth = 3
min_split = 2

rfr = RandomForestRegressor(n_estimators=n, max_depth=max_depth,
                            min_samples_split=min_split)
```

``` Python
# Load data
data = pd.read_csv('drive/MyDrive/Data/Boston Housing Prices.csv')

# Select variables
x = data['rooms'].values
y = data['cmedv'].values

# Plot data
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
plt.show()
```

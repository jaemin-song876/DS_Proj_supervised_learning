#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ISLP import load_data
Boston = load_data('Boston')
Boston.columns


# In[2]:


Boston.describe()


# In[4]:


import numpy as np

# log transformation (medv → medv_log)
Boston['medv_log'] = np.log(Boston['medv'])


Boston.drop(columns=['medv'], inplace=True)


print(Boston.head())


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
Boston.hist(figsize=(12, 10), bins=30)
plt.suptitle("Histogram of Boston Housing Variables")
plt.show()


# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Z-score
z_scores = np.abs(stats.zscore(Boston))


outliers = (z_scores > 3).sum(axis=0)
outliers_df = pd.DataFrame({"Feature": Boston.columns, "Outlier Count": outliers})


print("### Feature outliers (Z-score > 3 ) ###")
print(outliers_df.sort_values(by="Outlier Count", ascending=False))


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


corr_matrix = Boston.corr()


corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
    .rename(columns={0: "Correlation", "level_0": "Feature1", "level_1": "Feature2"})
)

top_corr_pairs = corr_pairs.reindex(corr_pairs["Correlation"].abs().sort_values(ascending=False).index)

print("### highest correlation pair ###")
print(top_corr_pairs.head(10))

# Heatmap 
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Boston Housing Dataset")
plt.show()


# In[10]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Z-score: remove outlier (Z-score > 3), except for chas
z_scores = np.abs(stats.zscore(Boston.drop(columns=['chas'])))  
Boston_cleaned = Boston[(z_scores < 3).all(axis=1)]  


Boston_cleaned['chas'] = Boston['chas']


y_cleaned = Boston_cleaned['medv_log']
X_cleaned = Boston_cleaned.drop(columns=['medv_log'])
X_cleaned = sm.add_constant(X_cleaned) 


model_cleaned = sm.OLS(y_cleaned, X_cleaned).fit()


print(model_cleaned.summary())


vif_data_cleaned = pd.DataFrame()
vif_data_cleaned["Feature"] = X_cleaned.columns
vif_data_cleaned["VIF"] = [variance_inflation_factor(X_cleaned.values, i) for i in range(X_cleaned.shape[1])]


print("\n### VIF after removing outliers) ###")
print(vif_data_cleaned)


# In[12]:


# Remove insignificant variables
X_reduced = X_cleaned.drop(columns=['indus', 'age', 'zn'])

# Fit the reduced regression model
model_reduced = sm.OLS(y_cleaned, X_reduced).fit()


print(model_reduced.summary())

# VIF
vif_data_reduced = pd.DataFrame()
vif_data_reduced["Feature"] = X_reduced.columns
vif_data_reduced["VIF"] = [variance_inflation_factor(X_reduced.values, i) for i in range(X_reduced.shape[1])]


print("\n### VIF (after removing indus, age, zn) ###")
print(vif_data_reduced)


# In[74]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

baseline_values = pd.DataFrame({
    "const": [1],  #must be 1
    "crim": [0.5],
    "chas": [0],  
    "nox": [0.5],
    "rm": [6],
    "dis": [5],
    "rad": [5],
    "tax": [300],
    "ptratio": [18],
    "lstat": [12]
})


print("Baseline Variables:", baseline_values.columns)


predictions = model_reduced.get_prediction(baseline_values)

log_medv_pred = predictions.predicted_mean[0]

conf_int = predictions.conf_int(alpha=0.05)  # 95% 신뢰구간
log_medv_lower = conf_int[0][0]
log_medv_upper = conf_int[0][1]

medv_pred = round(np.exp(log_medv_pred) * 1000, 2)
medv_lower = round(np.exp(log_medv_lower) * 1000, 2)
medv_upper = round(np.exp(log_medv_upper) * 1000, 2)

medv_pred, (medv_lower, medv_upper)


# In[13]:


#make a dataset without rad
X_reduced_rad = X_cleaned.drop(columns=['rad'])


model_reduced_rad = sm.OLS(y_cleaned, X_reduced_rad).fit()


print(model_reduced_rad.summary())

# VIF
vif_data_reduced_rad = pd.DataFrame()
vif_data_reduced_rad["Feature"] = X_reduced_rad.columns
vif_data_reduced_rad["VIF"] = [variance_inflation_factor(X_reduced_rad.values, i) for i in range(X_reduced_rad.shape[1])]

# VIF 
print("\n### VIF (without rad) ###")
print(vif_data_reduced_rad)


# In[58]:


## Diagnostic checks for linear regression assumptions (normality, homoscedasticity, independence)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Calculate residuals
residuals = model_reduced.resid

# Histogram (to check normality of residuals)
plt.figure(figsize=(12, 5))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Q-Q Plot (to check normality of residuals)
fig, ax = plt.subplots(figsize=(6,6))
stats.probplot(residuals, dist="norm", plot=ax)
plt.title("Q-Q Plot of Residuals")
plt.show()

# Residuals vs Fitted values plot (to check homoscedasticity)
plt.figure(figsize=(12, 5))
sns.scatterplot(x=model_reduced.fittedvalues, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()


# In[14]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

#
vif_data_final = pd.DataFrame()
vif_data_final["Feature"] = X_reduced.columns  
vif_data_final["VIF"] = [variance_inflation_factor(X_reduced.values, i) for i in range(X_reduced.shape[1])]

# VIF 결과 출력
print("\n### final model's VIF ###")
print(vif_data_final)


# In[ ]:





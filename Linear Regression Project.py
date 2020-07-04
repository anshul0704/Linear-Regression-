
# coding: utf-8

# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website.
# 
# 

# In[275]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[276]:


customers = pd.read_csv("Ecommerce Customers")


# In[277]:


customers.head()


# In[278]:


customers.describe()


# In[279]:


customers.info()


# ## Data Analysis

# In[280]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[281]:


# More time on site, more money spent.
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)


# # ##with the Time on App column

# In[282]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)


# # **  comparing Time on App and Length of Membership.**

# In[283]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)


# In[284]:


sns.pairplot(customers)


# # ** a linear model plot of  Yearly Amount Spent vs. Length of Membership. **

# In[286]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


# ## Training and Testing Data

# In[287]:


y = customers['Yearly Amount Spent']


# In[288]:


X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[289]:


from sklearn.model_selection import train_test_split


# In[290]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# # Linear Regression

# In[291]:


from sklearn.linear_model import LinearRegression


# In[292]:


lm = LinearRegression()


# In[293]:


lm.fit(X_train,y_train)


# **Print out the coefficients of the model**

# In[294]:


print('Coefficients: \n', lm.coef_)


# ## Predicting Test Data

# In[295]:


predictions = lm.predict( X_test)


# In[296]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# ## Evaluating the Model

# In[303]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[298]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# # **  interpreting the coefficients? **

# 
# a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
# a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
# a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
# a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.

# Develop the Website to catch up to the performance of the mobile app, or develop the app more since that is what is working better.

#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().system('pip install wget')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


To download the data, we will use !wget to download it from IBM Object Storage.


# In[9]:


df=pd.read_csv("FuelConsumptionCO2.csv")
df.head()


# In[10]:


#summarize the data
df.describe()


# In[11]:


#SELECTING AND EXPLORING SOME FEATURES
tdf = df[['ENGINESIZE' , 'CYLINDERS' , 'FUELCONSUMPTION_COMB' , 'CO2EMISSIONS']]
tdf.head(9)


# In[12]:


#PLOTTING EACH FEATURE
viz = tdf[['ENGINESIZE' , 'CYLINDERS' , 'FUELCONSUMPTION_COMB' , 'CO2EMISSIONS']]
viz.hist()
plt.show()


# In[14]:


#PLOTTING EACH FEATURE AGAINST CO2 EMISSIONS TO SEE LINEAR RELATIONSHIP
plt.scatter(tdf.ENGINESIZE, tdf.CO2EMISSIONS, color='green')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[15]:


plt.scatter(tdf.FUELCONSUMPTION_COMB, tdf.CO2EMISSIONS, color='green')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[17]:


plt.scatter(tdf.CYLINDERS, tdf.CO2EMISSIONS, color='green')
plt.xlabel("CYLINDERS")
plt.ylabel("EMISSIONS")
plt.show()


# In[18]:


#CREATING TRAIN AND TEST DATASET
msk = np.random.rand(len(df)) < 0.8
train = tdf[msk]
test = tdf[~msk] 


# In[20]:


#SIMPLE REGRESSION MODEL

#TRAIN DATA DISTRIBUTION

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Co2 Emissions")
plt.show()


# In[22]:


#Using Sklearn to model data

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[23]:


#PLOTTING OUTPUTS
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[24]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )


# In[ ]:


#Lets see what the evaluation metrics are if we trained a regression model using the FUELCONSUMPTION_COMB feature


# In[27]:


from sklearn import linear_model

df=pd.read_csv("FuelConsumptionCO2.csv")
fdf = df[["FUELCONSUMPTION_COMB" , "CO2EMISSIONS"]]
#fdf.head(9)
train = fdf["FUELCONSUMPTION_COMB"]
train_x = fdf[["FUELCONSUMPTION_COMB"]]
test_x = fdf[["FUELCONSUMPTION_COMB"]]

train_y = fdf[["CO2EMISSIONS"]]
test_y = fdf[["CO2EMISSIONS"]]

#Now train a Linear Regression Model using the train_x you created and the train_y created previously
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

#Find the predictions using the model's predict function and the test_x data
predictions = regr.predict(test_x)

print("Mean Absolute Error: %.2f" % np.mean(np.absolute(predictions - test_y)))
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[29]:


#PLOTTING OUTPUTS
plt.scatter(fdf.FUELCONSUMPTION_COMB, fdf.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Fuel Consumption_Comb")
plt.ylabel("Emission")


# In[ ]:





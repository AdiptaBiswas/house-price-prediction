#!/usr/bin/env python
# coding: utf-8

# ### Loaders 

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy.stats import anderson
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.simplefilter(action='ignore', category=FutureWarning)
import scipy.stats as stats
pd.pandas.set_option('display.max_columns',None)
from IPython.display import HTML
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split 
from scipy.stats import skew
from scipy.stats.stats import pearsonr

main_train_df = pd.read_csv("C:\\Users\\RICK\\Documents\\Data set\\train.csv")
main_test_df = pd.read_csv("C:\\Users\\RICK\\Documents\\Data set\\test.csv")


# In[2]:


main_train_df.head(20)


# ### Data summary 

# In[3]:


train_df = main_train_df
train_df.drop(columns="Id", inplace = True)
test_df = main_test_df
print("Data shape:",train_df.shape)
print()
type_ls = []
type_dict = {}
for col in train_df.columns:
    if train_df[col].dtype not in type_ls:
        type_ls.append(train_df[col].dtype)
        type_dict[train_df[col].dtype] = 1
    else:
        counter = type_dict[train_df[col].dtype]
        type_dict[train_df[col].dtype] = counter + 1
print("Feature datatypes:")
for key, value in type_dict.items():
    print(key,"->",value)    
print()
print("Features with NA values more than 20%:")
for col in train_df.columns:
    na_ptg = round(100*(train_df[col].isna().sum()/len(train_df)),2)
    if na_ptg >= 20:
        print("{} -> {}".format(col, na_ptg))
print()
print("Data description:\n")
train_df.describe()


# * The data provided has 1460 unique data-points along with 81 columns.
# * There are 35 "Integer" type columns, 43 columns of type "Object" and 3 of "Float" type.
# * There are 4 columns with blank values (>50%), they are: Alley (93.77%), PoolQC (99.52%), Fence (80.75%) and MiscFeature (96.3%).

# In[4]:


numeric_discrete_vars = []
categorical_vars = []
numeric_vars = []
for col in train_df.columns:
    if train_df[col].dtype == "object":
        categorical_vars.append(col)
    elif len(train_df[col].unique()) < 25:
           numeric_discrete_vars.append(col)
    else:
        numeric_vars.append(col)
print("Numeric Discrete variables:\n",numeric_discrete_vars)
print()
print("Categorical variables:\n",categorical_vars)
print()
print("Numeric variables:\n",numeric_vars)


# ### 1. Data visualization, Variable analysis and EDA

# In[5]:


corr = train_df[numeric_vars].corr()
corr.style.background_gradient(cmap='coolwarm', axis=None)


# In[6]:


high_cor_feat = set()
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.5:
            colname = corr.columns[i]
            high_cor_feat.add(colname)
print("Highly correlated features:\n",high_cor_feat)


# * There are outliers present in GrLivArea and GarageArea. These two area features show high correlations.
# * BsmtFinSF1 is highly negatively correlated with BsmtUnfSF. GrLivArea has a high correlation with the TotRmsAbvGrd, GarageCars has a high correlation with the GarageArea and vice-versa.
# * Majority of the GarageArea are between 300 and 700. A large number of GrLivArea are between 1000 and 2500.
# * The Slope between GrLivArea and SalePrice is steeper than GarageArea and SalePrice.

# In[7]:


for i in ['TotalBsmtSF', 'GarageArea', 'GrLivArea']:
    fig, ax = plt.subplots()
    ax.set_title("Area feature with outliers present", fontsize=15)
    ax.scatter(x = train_df[i], y = train_df['SalePrice'], alpha = 0.5, c="red")
    plt.ylabel('SalePrice', fontsize=15)
    plt.xlabel(i, fontsize=15)
    plt.show()


# The space or area related features having a correlation more than 50% have been plotted in the above. The plots  show that they have many outliers. The presence of outliers can often lead to a fluctuating correlation factor. 

# In[8]:


train_NoOutlier = train_df[train_df.GrLivArea < 3500]
fig, ax = plt.subplots()
ax.set_title("GrLivArea with outliers removed", fontsize=15)
ax.scatter(x=train_NoOutlier['GrLivArea'], y=train_NoOutlier['SalePrice'], alpha = 0.5, c="purple")
plt.ylabel('SalePrice', fontsize=15)
plt.show()


# In[9]:


train_NoOutlier = train_df[(train_df.TotalBsmtSF < 2500) & (train_df.TotalBsmtSF > 200)]
fig, ax = plt.subplots()
ax.set_title("TotalBsmtSF with outliers removed", fontsize=15)
ax.scatter(x=train_NoOutlier['TotalBsmtSF'], y=train_NoOutlier['SalePrice'], alpha = 0.4, c="purple")
plt.ylabel('SalePrice', fontsize=15)
plt.show()


# The demonstration done manually shows that the outliers have been removed from GrLivArea and TotalBsmtSF features. They have been plotted against SalePrice.

# In[10]:


plot_df = train_df.copy()
col_colr = ["red", "orange", "lime", "purple"]
labels = [i for i in train_df.columns if "Yr" in i or "Year" in i]
cont = 0
for year_cols in labels:
    if "Yr" in year_cols or "Year" in year_cols: 
        temp_df = plot_df.groupby(year_cols)["SalePrice"].median()
        temp_df.plot(color = col_colr[cont], figsize=(22,10))
        plt.xlabel("Timeline", size=15)
        plt.legend(labels = labels)
        plt.ylabel("Median House Price", size=15)
        plt.title("House Sale Price and Year features", size=15)
        cont += 1


# * From the above time-series visualization, we can see that with an increase in built year or year of establishment of a house YearBuilt, there is an increase in median price. In other words, if a house is recently built, it's median SP is highly likely to be more than a house which was built a decade earlier.
# * The same goes with house garage built year GarageYrBlt and house remodification year YearRemodAdd.
# * The year of selling YrSold is just the opposite as the median price is seen to be decreasing with an increasing in the year of house getting sold, i.e the houses sold recently have lower median price than the houses sold a few years back.

# In[11]:


value = train_df["SalePrice"]
sns.distplot(value, kde=False, fit=stats.norm, color = "blue")
plt.xlabel("Skewed SalePrice", size = 15)


# In[12]:


norm_test = anderson(train_df["SalePrice"])
print("Anderson Darling test of Normality:")
norm_test


# SalesPrice isn't following a normal distibution as it is right-skewed. From the Aderson Darling test, we have achieved a statistic or around 41.69. We have also obtained a list of critical values which are far less than the statistic value. Thus, we can infer that the distribution in the data isn't Normal, thus fails to accept the null hypothesis. A Log transformation might change the shape of the distribution.

# In[13]:


SP_Log_Trnsfmd = train_df["SalePrice"].copy()
SP_Log_Trnsfmd = SP_Log_Trnsfmd.apply(lambda x: np.log1p(x))
sns.distplot(SP_Log_Trnsfmd, kde=False, fit=stats.norm, color = "red")
plt.xlabel("Log transformed SalePrice", size = 15)


# Log transformation demonstration performed on "SalePrice". As the distribution of the dependent variable looks normal, a Linear model can be fit on the variable.

# In[14]:


print("Top 20 important features as per correlation values and SalePrice:")
corr = abs(train_df.corr())
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
print(corr.SalePrice[1:21])
Imp_feat_df = pd.DataFrame(corr.SalePrice[1:11])
Imp_feat_df.rename(columns = {"SalePrice": "Importance"}, inplace = True)


# In[15]:


x = list(Imp_feat_df.index) 
y = list(Imp_feat_df['Importance'])  
fig = plt.figure(figsize = (12, 5)) 
plt.bar(x, y, color ='blue',  
        width = 0.4)
plt.xlabel("Features", size=15) 
plt.ylabel("Importance", size=15) 
plt.title("Top 10 features with correlation sorted by SalePrice", size=15) 
plt.show() 


# The above plot is demonstrating the top 10 important features w.r.t their correlation factors against the y-variable SalePrice.
# This is a correlation sorted by SalePrice values, which is our dependent variable. 

# In[16]:


print("Top 5 important features and their distributions:")
Imp_feat_df["Feature"] = Imp_feat_df.index
Imp_feat_df.set_index(np.arange(1,11))
important_feats = [i for i in Imp_feat_df["Feature"]]
sns.set()
cols = important_feats[:5]
sns.pairplot(train_df[cols], height = 2.5)
plt.show();


# * OverallQual and GrLivArea are strongly correlated with the target variable with 0.82 and 0.7.
# * GarageCars, GarageArea, TotalBsmtSF, etc. follows the top 2 important variables w.r.t SalePrice.
# * GarageCars and GarageArea follow an almost similar correlation w.r.t SalePrice. More number of cars one would grab, much larger space that individual would need. That would iventually play an important role deciding the price of his/her plot.
# * Higher the quality of a house overall OverallQual, higher is the price of a house. The OverallQual has a high correlation with the target variable.
# * The presence of outliers are prominent in every features. Some of them are not showing a linear relationship as well. The ouliers are making those features important.  

# In[17]:


def imp_feat(name, vars_ls):
    print(f"Top 20 important {name} as per correlation values and SalePrice:")
    cols = vars_ls
    cols.append('SalePrice')
    for c in cols:
        if c == 'SalePrice' and cols.count(c) > 1:
            cols.pop()
    corr = abs(train_df[cols].corr())
    corr.sort_values(['SalePrice'], ascending=False, inplace=True)
    print(corr.SalePrice[1:21])
imp_feat("Discrete variables", numeric_discrete_vars)
print()
imp_feat("Mumeric variables", numeric_vars)


# Top 20 highly correlated features w.r.t SalePrice demonstrated seperately as per column data types. We can estimate the important features from the values above shown seperately. 

# ### 2. Data cleaning and pre-processing

# In[18]:


train_df[categorical_vars] = train_df[categorical_vars].fillna('None')


# The NULL values in the categorical variables are replaced with None in order to fill the missing values.

# In[19]:


for col in numeric_vars:
    train_df[col].fillna(train_df[col].median(),inplace=True)


# The NULL values in the numeric variables are replaced with Median of each variables in order to fill the missing values.

# In[20]:


for col in numeric_discrete_vars:
    train_df[col].fillna(train_df[col].median(),inplace=True)


# The NULL values in the discrete variables are replaced with Median of each variables in order to fill the missing values.

# In[21]:


train_df["SalePrice"] = train_df["SalePrice"].apply(lambda x: np.log1p(x))
train_df["SalePrice"].skew()


# The dependent variable SalePrice has been Log transformed in order to remove skewness from it.  

# In[22]:


for col in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:  
    train_df[col] = train_df['YrSold'] - train_df[col]


# Replacing year related variables with related age in values. The data of getting sold YrSold column is marked as the end date. This is done to gain some coefficient values from those columns. 

# In[23]:


for col in numeric_vars:
    train_df[col] = train_df[col].apply(lambda x: np.log1p(x))


# Log transform on the numeric features as well in order to remove skewness.

# In[24]:


print("Unique values in all the categorical variables:")
_encoder = preprocessing.LabelEncoder() 
for col in categorical_vars:  
    train_df[col] = _encoder.fit_transform(train_df[col])
    unique = list(set(train_df[col]))
    print(f"{col} -> {unique}")


# Applying label encoding method on all the category features to convert each label into numeric discrete values. 

# In[25]:


train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.fillna(train_df.mean(), inplace=True)
scale_feats = [cols for cols in train_df.columns if cols != 'SalePrice']
scale = MinMaxScaler()
scale.fit(train_df[scale_feats])
scale.transform(train_df[scale_feats])


# Min-Max Scaling is applied on the data set to scale all the independent variables under an unified standard. This is to avoid variation in values non-uniformity in units.  

# In[26]:


train_df_prepared = pd.DataFrame(scale.transform(train_df[scale_feats]), columns = scale_feats)
train_df_prepared = pd.concat([train_df_prepared, train_df['SalePrice']], axis = 1)
print("Tranformed and prepared data set:")
train_df_prepared.to_csv("House_Train_DF.csv")
train_df_prepared.head(20)


#1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import scipy.stats as sm
from statsmodels.formula.api import ols
import pylab
import statsmodels.api as st
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
import pickle as pkl
import plotly.express as ex
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import shap

#2. Load Files
data1 = pd.read_csv("E:/iNeuron/iNeuron Flight Fare Prediction/Project_internship/Air_Flight/output-000001.csv")
data2 = pd.read_csv("E:/iNeuron/iNeuron Flight Fare Prediction/Project_internship/Air_Flight/output-000002.csv")
data3 = pd.read_csv("E:/iNeuron/iNeuron Flight Fare Prediction/Project_internship/Air_Flight/output-000003.csv")
data4 = pd.read_csv("E:/iNeuron/iNeuron Flight Fare Prediction/Project_internship/Air_Flight/output-000004.csv")
data = pd.concat([data1, data2, data3, data4], axis = 0)
data.head()

#3. EDA Analysis
#Size of the data
data.shape

#Data Types
data.dtypes

#Checking inconsistency in Columns
for col in data.columns:
  print(col, " | ", data[col].unique())

data['Destination'] = np.where(data['Destination'] == 'New Delhi', 'Delhi', data['Destination'])
data['Additional_Info'] = np.where(data['Additional_Info'] == 'No Info', 'No info', data['Additional_Info'])

#Countplot
obj_columns = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info', 'Route']
fig, ax = plt.subplots(3, 2, figsize = (20, 45))
for i, j in enumerate(obj_columns):
  q, r = divmod(i, 2)
  g = sbn.countplot(data[j], ax = ax[q, r])
  g.set_xticklabels(g.get_xticklabels(), rotation = 45, fontsize = 9)
plt.show()

#Average Price of Airline Travelled from Source to Destination
data.groupby(['Airline', 'Source', 'Destination']).agg(['mean', 'max', 'min'])['Price']

#Total Airline Flights Travelled from Source to Destination
data_count = data.groupby(['Airline', 'Source', 'Destination']).agg({'Airline' : 'count'})
data_count.columns = ['# of Flights']
data_count

#Calculate the Duration in Minutes
splits = []
for text in data['Duration']:
  if text.find('h ') > 0:
    splits.append(text.split('h '))
  else:
    splits.append(text.split('h'))
Duration = []
for i in range(len(splits)):
  if splits[i][0].find('m') > 0:
    Duration.append(int(splits[i][0][:-1]))
  elif splits[i][1][:-1] != '':
    Duration.append((int(splits[i][0]) * 60) + int(splits[i][1][:-1]))
  else:
    Duration.append((int(splits[i][0]) * 60))
hr = []
min = []
for text in data['Dep_Time']:
  hr.append(text.split(':')[0])
  min.append(text.split(':')[1])
data['Duration'] = Duration
data['Dep_Time_hr'] = np.array(hr).astype('int')
data['Dep_Time_min'] = np.array(min).astype('int')

#Descriptive Statistics
data.describe()

#Average Duration taken by the Flights from Source to Destination
data.groupby(['Airline', 'Source', 'Destination']).agg(['mean', 'max', 'min'])['Duration']

#Impact of Source and Destination on Flight Price
model = ols('Price ~ Source + Destination', data = data).fit()
anova_result = st.stats.anova_lm(model, typ = 2)
print(anova_result)

print("Source and Destination doesn't have effect on the Price")

#Impact of Source and Destination on Flight Duration
model = ols('Duration ~ Source + Destination', data = data).fit()
anova_result = st.stats.anova_lm(model, typ = 2)
print(anova_result)

print("Source and Destination doesn't have effect on the Flight Duartion")

#Impact of route on Price
model = ols('Price ~ Route', data = data).fit()
anova_result = st.stats.anova_lm(model, typ = 2)
print(anova_result)

print("Flight Route have effect on the Flight Price")

#Impact of route on Duration
model = ols('Duration ~ Route', data = data).fit()
anova_result = st.stats.anova_lm(model, typ = 2)
print(anova_result)

print("Flight Route have effect on the the Flight Duration")

#Impact of Total_Stops on Duration
model = ols('Duration ~ Total_Stops', data = data).fit()
anova_result = st.stats.anova_lm(model, typ = 2)
print(anova_result)

print("Total Stops have effect on the the Flight Duration")

#Impact of Total_Stops on Price
model = ols('Price ~ Total_Stops', data = data).fit()
anova_result = st.stats.anova_lm(model, typ = 2)
print(anova_result)

print("Total Stops have effect on the the Flight Price")

#Impact of Airline Flights on Price
model = ols('Price ~ Airline', data = data).fit()
anova_result = st.stats.anova_lm(model, typ = 2)
print(anova_result)

print("Airline Flights have effect on the the Flight Price")

#Correlation Analysis
plt.figure(figsize = (5, 5))
sbn.heatmap(data[['Price', 'Duration']].corr(), annot = True)
plt.show()

print("There is a strong positive correlation between Duration and Price")

#Q-Q plots to visualize if the variable is normally distributed
columns = ['Price', 'Duration']
plt.figure(figsize = (10, 10))
for i, col in enumerate(columns):
  plt.subplot(2, 2, i + 1)
  sm.probplot(data[col], dist="norm", plot = pylab)
  plt.title(col)

#Distribution Analysis
q, r = divmod(len(columns), 2)
fig, ax = plt.subplots(q, 2, figsize = (10, 5))
print("skewness")
for i, col in enumerate(columns):
  print(col, " : ", data[col].skew())
  q, r = divmod(i, 2)
  sbn.histplot(data[col], kde = True, ax = ax[r], bins = 30)

#Detection of Outliers
plt.figure(figsize = (10, 5))
sbn.boxplot(data['Duration'])
plt.show()

#Detection of Missing Values
data.isna().sum()

#Bivariate Analysis - Scatter Plot
ex.scatter(data, x = 'Duration', y = 'Price', color = 'Airline')

#Bivariate Analysis - Box Plot
plt.figure(figsize = (10, 10))
sbn.boxplot(data = data, x = 'Total_Stops', y = 'Duration', hue = 'Airline', palette = 'rainbow')
plt.title('Box plot of Duration vs Total_Stops')
plt.show()

#Bivariate Analysis - Box Plot
plt.figure(figsize = (10, 10))
sbn.boxplot(data = data, x = 'Total_Stops', y = 'Price', hue = 'Airline', palette = 'rainbow')
plt.title('Box plot of Price vs Total_Stops')
plt.show()

data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], format = '%d/%m/%Y')

#Average Flight Fare over a period of time
data_Price = data.groupby(['Airline', 'Date_of_Journey']).agg({'Price' : 'mean'}).reset_index()
ex.line(data_Price, x = 'Date_of_Journey', y = 'Price', color = 'Airline', markers = True, title = 'Average Flight Fare over a period of time')

#Average Flight Duration over a period of time
data_Dur = data.groupby(['Airline', 'Date_of_Journey']).agg({'Duration' : 'mean'}).reset_index()
ex.line(data_Dur, x = 'Date_of_Journey', y = 'Duration', color = 'Airline', markers = True, title = 'Average Flight Duration over a period of time')

#4. Data Preprocessing
#Remove unwanted columns
data = data.drop(['Dep_Time', 'Arrival_Time'], axis = 1)

#Handling Outliers 
data['Duration'] = np.sqrt(data['Duration'])

#After Handling Outliers
plt.figure(figsize = (10, 5))
sbn.boxplot(data['Duration'])
plt.show()

#One Hot Encoding
one_hot = OneHotEncoder(drop = 'first').fit(data[['Airline', 'Source', 'Destination', 'Route', 'Total_Stops', 'Additional_Info']])
data_cat = one_hot.transform(data[['Airline', 'Source', 'Destination', 'Route', 'Total_Stops', 'Additional_Info']]).toarray()
data_cat_final = pd.DataFrame(data_cat)
data_cat_final.columns = one_hot.get_feature_names_out(['Airline', 'Source', 'Destination', 'Route', 'Total_Stops', 'Additional_Info'])
data.reset_index(drop = True, inplace = True)
#Saving the model
pkl.dump(one_hot, open('one_hot_model.pkl', 'wb'))
data_final = pd.concat([data, data_cat_final], axis = 1)

#Feature Creation
data_final['Month'] = data_final['Date_of_Journey'].dt.month
data_final['Date'] = data_final['Date_of_Journey'].dt.day

#Features and Target
Features = data_final.drop(['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route', 'Price', 'Total_Stops', 'Additional_Info'], axis = 1)
Target = data_final['Price']

#Train Test Split
x_train, x_test, y_train, y_test = train_test_split(Features, Target, test_size = 0.2, random_state = 10)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

#Feature Scaling and Transformation of the Target variabe
y_train_trans = np.cbrt(y_train)
y_test_trans = np.cbrt(y_test)
sc = StandardScaler().fit(x_train)
x_train_trans = sc.transform(x_train)
x_test_trans = sc.transform(x_test)
#Saving the model
pkl.dump(sc, open('standardscaler_model.pkl', 'wb'))

#5. Model Selection
#i. Linear Regression models
linear_models = [LinearRegression(), Ridge(), Lasso(), ElasticNet()]
for model in linear_models:
  train_model = model.fit(x_train_trans, y_train_trans)
  train_pred = model.predict(x_train_trans)
  test_pred = model.predict(x_test_trans)
  train_Adj_r2 = 1 - (1 - r2_score(y_train_trans, train_pred)) * (len(train_pred) - 1) / (len(train_pred) - x_train_trans.shape[1] - 1)
  test_Adj_r2 = 1 - (1 - r2_score(y_test_trans, test_pred)) * (len(test_pred) - 1) / (len(test_pred) - x_test_trans.shape[1] - 1)
  train_rootmean_square = np.sqrt(mean_squared_error(y_train_trans, train_pred))
  test_rootmean_square = np.sqrt(mean_squared_error(y_test_trans, test_pred))
  print(model.__class__.__name__, ":", "Training Adjusted R2 Score: ", train_Adj_r2, "Testing Adjusted R2 Score: ", test_Adj_r2)
  print("Training root mean square Error: ", train_rootmean_square, "Testing root mean square Error: ", test_rootmean_square)

#Based on Previous scores, Selecting Ridge Model
#Model Validation
model = Ridge().fit(x_train_trans, y_train_trans)
train_pred = model.predict(x_train_trans)
plt.subplot(121)
sbn.scatterplot(y_train_trans, train_pred)
plt.subplot(122)
sbn.distplot(y_train_trans - train_pred)

#ii. Tree Based Models
tree_models = [DecisionTreeRegressor(), RandomForestRegressor(), BaggingRegressor(), AdaBoostRegressor(), XGBRegressor(), LGBMRegressor()]
for model in tree_models:
  train_model = model.fit(x_train, y_train)
  train_pred = model.predict(x_train)
  test_pred = model.predict(x_test)
  train_Adj_r2 = 1 - (1 - r2_score(y_train, train_pred)) * (len(train_pred) - 1) / (len(train_pred) - x_train.shape[1] - 1)
  test_Adj_r2 = 1 - (1 - r2_score(y_test, test_pred)) * (len(test_pred) - 1) / (len(test_pred) - x_test.shape[1] - 1)
  train_rootmean_square = np.sqrt(mean_squared_error(y_train, train_pred))
  test_rootmean_square = np.sqrt(mean_squared_error(y_test, test_pred))
  print(model.__class__.__name__, ":", "Training Adjusted R2 Score: ", train_Adj_r2, "Testing Adjusted R2 Score: ", test_Adj_r2)
  print("Training root mean square Error: ", train_rootmean_square, "Testing root mean square Error: ", test_rootmean_square)

#iii. Other Regression models
other_models = [KNeighborsRegressor(), SVR()]
for model in other_models:
  train_model = model.fit(x_train_trans, y_train)
  train_pred = model.predict(x_train_trans)
  test_pred = model.predict(x_test_trans)
  train_Adj_r2 = 1 - (1 - r2_score(y_train, train_pred)) * (len(train_pred) - 1) / (len(train_pred) - x_train_trans.shape[1] - 1)
  test_Adj_r2 = 1 - (1 - r2_score(y_test, test_pred)) * (len(test_pred) - 1) / (len(test_pred) - x_test_trans.shape[1] - 1)
  train_rootmean_square = np.sqrt(mean_squared_error(y_train, train_pred))
  test_rootmean_square = np.sqrt(mean_squared_error(y_test, test_pred))
  print(model.__class__.__name__, ":", "Training Adjusted R2 Score: ", train_Adj_r2, "Testing Adjusted R2 Score: ", test_Adj_r2)
  print("Training root mean square Error: ", train_rootmean_square, "Testing root mean square Error: ", test_rootmean_square)

#Based on Adjusted R2 and root mean square Error, I am choosing XGBRegressor

#6. Hyperparameter tuning
kfold = KFold(n_splits = 10)
params = {'max_depth' : [6, 8, 9], 'reg_alpha' : [10, 100, 10e4]}
gs = GridSearchCV(XGBRegressor(learning_rate = 0.1), param_grid = params, scoring = ['r2','neg_root_mean_squared_error'], refit = 'r2', cv = kfold, n_jobs = -1)
gs.fit(x_train, y_train)

#Best model
XGB_tuned_model = gs.best_estimator_
print(XGB_tuned_model)

#Best score validation score
print(gs.best_score_)

#7. Model Building and Training
XGB_Model = XGB_tuned_model.fit(x_train, y_train)

#8. Model Evaluation
train_pred = XGB_Model.predict(x_train)
test_pred = XGB_Model.predict(x_test)
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
train_Adj_r2 = 1 - (1 - r2_score(y_train, train_pred)) * (len(train_pred) - 1) / (len(train_pred) - x_train.shape[1] - 1)
test_Adj_r2 = 1 - (1 - r2_score(y_test, test_pred)) * (len(test_pred) - 1) / (len(test_pred) - x_test.shape[1] - 1)
train_rootmean_square = np.sqrt(mean_squared_error(y_train, train_pred))
test_rootmean_square = np.sqrt(mean_squared_error(y_test, test_pred))
print("Training R2 Score: ", train_r2, "Testing R2 Score: ", test_r2)
print("Training Adjusted R2 Score: ", train_Adj_r2, "Testing Adjusted R2 Score: ", test_Adj_r2)
print("Training root mean square Error: ", train_rootmean_square, "Testing root mean square Error: ", test_rootmean_square)

#9. Summary Plot - Model Interpretation
explainer = shap.TreeExplainer(XGB_Model)
shap_values = explainer.shap_values(x_train)
shap.summary_plot(shap_values, x_train)

#10. Model Saving
pkl.dump(XGB_Model, open('XGBRegressor.pkl', 'wb'))

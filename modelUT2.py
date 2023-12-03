import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


newdata=pd.read_csv(r"C:\Users\Asus\Downloads\UtPredictor\ST2ml.csv", encoding='unicode_escape')
dataseteven=newdata
dataseteven.isna().sum()
df_filtered=dataseteven.dropna()
df_filtered.isna().sum()
columns_to_check = ['CHEMISTRY', 'MATHS', 'ELECTRONICS', 'MECHANICAL', 'SOFTSKILLS',
                    'TOTAL', 'PERCENTAGE', 'CHEMISTRYPUT', 'MATHSPUT',
                    'ELECTRONICSPUT', 'MECHANICALPUT', 'SOFTSKILLSPUT', 'TOTALPUT',
                    'PERCENTAGEPUT']
df_filtered.isna().sum()
df_filtered.dropna()
df=df_filtered
STfields = ['CHEMISTRY', 'MATHS', 'ELECTRONICS', 'MECHANICAL', 'SOFTSKILLS']
PUTfields = ['CHEMISTRYPUT', 'MATHSPUT', 'ELECTRONICSPUT', 'MECHANICALPUT', 'SOFTSKILLSPUT', 'TOTALPUT', 'PERCENTAGEPUT']
X = df[STfields]
y = df[PUTfields]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=30)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

pickle.dump(model, open('modelUT2.pkl','wb'))
modelUT2 = pickle.load(open('modelUT2.pkl','rb'))


#STnew_data = {
#    'CHEMISTRY': [25],
#    'MATHS': [34],
#    'ELECTRONICS': [36],
#    'MECHANICAL': [31],
#    'SOFTSKILLS': [31],
#   
#}
#STnew_data_df = pd.DataFrame(STnew_data)
#new_predictions = model.predict(STnew_data_df[STfields])
#print(f'Predictions for UT:\n{new_predictions}')
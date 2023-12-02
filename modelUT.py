import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
import seaborn as sns
df=pd.read_csv(r"C:\Users\Asus\Downloads\UtPredictor\ST2ml.csv", encoding='unicode_escape')
df=df.dropna()
df_filtered=df.copy()
df_filtered.WantsToUpgrade=0
df_filtered["WantsToUpgrade"]=0
num_rows_to_set_to_1 = int(0.8 * len(df_filtered))
indices_to_set_to_1 = np.random.choice(df_filtered.index, num_rows_to_set_to_1, replace=False)
df_filtered.loc[indices_to_set_to_1, 'WantsToUpgrade'] = 1
df_sorted=df_filtered.sort_values(by='PERCENTAGEPUT', ascending=False)
top_20_percent = int(0.2 * len(df_sorted))
df_sorted_top20percent = df_sorted.head(top_20_percent)
df_sorted_top20percent[df_sorted_top20percent["WantsToUpgrade"]==1]
applicants=df_sorted_top20percent[df_sorted_top20percent["WantsToUpgrade"]==1]
applicant_df=applicants.copy()
applicant_df["WantsToUpgradeTo"] = 0
applicant_df.iloc[:7, applicant_df.columns.get_loc('WantsToUpgradeTo')] = 'CSE'
applicant_df.iloc[7:14, applicant_df.columns.get_loc('WantsToUpgradeTo')] = 'CSE AI ML'
applicant_df.iloc[14:21, applicant_df.columns.get_loc('WantsToUpgradeTo')] = 'CSE DS'
applicant_df.iloc[21:28, applicant_df.columns.get_loc('WantsToUpgradeTo')] = 'CS'
applicant_df.iloc[28:35, applicant_df.columns.get_loc('WantsToUpgradeTo')] = 'CSIT'
applicant_df.iloc[35:, applicant_df.columns.get_loc('WantsToUpgradeTo')] = 'IT'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
label_encoder = LabelEncoder()
applicant_df['WantsToUpgradeToEncoded'] = label_encoder.fit_transform(applicant_df['WantsToUpgradeTo'])

X = applicant_df[['PERCENTAGEPUT']]
y = applicant_df['WantsToUpgradeToEncoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

pickle.dump(model, open('modelUT.pkl','wb'))
modelUT = pickle.load(open('modelUT.pkl','rb'))

#PUTPERCENTAGE = [[78.6]] 
#BRANCHPREDICTION = model.predict(PUTPERCENTAGE)
#PREDICTEDBRANCH = label_encoder.inverse_transform([int(BRANCHPREDICTION)])

#print(f"Sample Prediction: {PUTPERCENTAGE[0][0]}% will get you the branch: {PREDICTEDBRANCH[0]}")

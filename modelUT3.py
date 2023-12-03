import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
import seaborn as sns
import pickle

df=pd.read_csv(r"C:\Users\Asus\Downloads\UtPredictor\ST2ml.csv", encoding='unicode_escape')
df=df.dropna()
df_filtered=df.copy()
df_filtered["PreferredElective"]=0
numeric_columns = ['CHEMISTRY', 'MATHS', 'ELECTRONICS', 'MECHANICAL', 'SOFTSKILLS']
df_filtered[numeric_columns] = df_filtered[numeric_columns].apply(pd.to_numeric, errors='coerce')
print(df_filtered[numeric_columns].dtypes)
df_filtered[numeric_columns] = df_filtered[numeric_columns].fillna(0)  # Replace NaN with 0, adjust based on your use case
df_filtered['PreferredElective'] = df_filtered[numeric_columns].idxmax(axis=1)
subject_to_elective_mapping = {
    'CHEMISTRY': 'MaterialScience',
    'MATHS': 'DigitalElectronics',
    'ELECTRONICS': 'SensorInstrumentation',
    'MECHANICAL': 'MechanicsApplied',
    'SOFTSKILLS': 'EnergySciences'
}
df_filtered['PreferredElective'] = df_filtered['PreferredElective'].map(subject_to_elective_mapping)
np.random.seed(42)
num_rows_per_category = int(0.2 * len(df_filtered))
categories = ["DigitalElectronics", "MaterialScience", "EnergyScience", "Mechanics", "SensorInstrumentation"]
indices_per_category = []
for category in categories:
    indices = np.random.choice(df_filtered.index, num_rows_per_category, replace=False)
    indices_per_category.append(indices)
    df_filtered.loc[indices, 'PreferredElective'] = category

df_filtered = df_filtered[df_filtered['PreferredElective'] != 0]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

applicant_df=df_filtered.copy()

label_encoder = LabelEncoder()
applicant_df['PreferredElectiveEncoded'] = label_encoder.fit_transform(applicant_df['PreferredElective'])

X = applicant_df[['CHEMISTRY', 'MATHS', 'ELECTRONICS', 'MECHANICAL', 'SOFTSKILLS']]
y = applicant_df['PreferredElectiveEncoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#STMARKS = pd.DataFrame({'CHEMISTRY': [20], 'MATHS': [25], 'ELECTRONICS': [38], 'MECHANICAL': [35], 'SOFTSKILLS': [40]})
#ELECTIVEPREDICTION = model.predict(STMARKS)
#PREDICTEDELECTIVE = label_encoder.inverse_transform([int(ELECTIVEPREDICTION)])

#print(f"YOU SHOULD TAKE THE BRANCH: {PREDICTEDELECTIVE[0]}")

pickle.dump(model, open('modelUT3.pkl','wb'))
modelUT2 = pickle.load(open('modelUT3.pkl','rb'))
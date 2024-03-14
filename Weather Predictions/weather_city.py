# Utsav Anantbhat


import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Read data
data_labelled = pd.read_csv(sys.argv[1])
data_unlabelled = pd.read_csv(sys.argv[2])

# Access the data
X_lab = data_labelled.loc[:,'tmax-01':'snwd-12']
X_unlab = data_unlabelled.loc[:,'tmax-01':'snwd-12']
y = data_labelled['city']

# Training
X_train, X_test, y_train, y_test = train_test_split(X_lab, y)
training_model = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=15)) #use Standard Scaler to transform data
training_model.fit(X_train, y_train)

# Predictions
pred = training_model.predict(X_unlab)
print("Model Score: ", training_model.score(X_test, y_test))

# Dataframe
df = pd.DataFrame({'truth': y_test, 'prediction': training_model.predict(X_test)})
#print(df[df['truth'] != df['prediction']])
pd.Series(pred).to_csv(sys.argv[3], index=False, header=False)
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# random seed
seed = 42

#read original dataset
iris_df = pd.read_csv("Data/Iris.csv")
#iris_df.sample(frac=1, random_state=seed)

#selecting features and target data
X = iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalLengthCm']]
y = iris_df[['Species']]

#split data into train and test sets
#70% training 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

#create an instance of k neighbours classifier
clf = KNeighborsClassifier(n_neighbors=10)

#train the classifier on training data
clf.fit(X_train,y_train)

#predict on test set
y_pred = clf.predict(X_test)

#calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")     #accuracy: 0.91

# Save model to disk
joblib.dump(clf, "output_models/kn_model.sav")



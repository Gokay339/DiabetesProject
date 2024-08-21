import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from google.colab import files
uploaded = files.upload()

pd = pd.read_csv('diabetes.csv')
pd.head()

pd.rename(columns={'Pregnancies':'Hamilelik','Glucose':'Glikoz','BloodPressure':'Tansiyon','SkinThickness':'Cilt Kalınlığı','DiabetesPedigreeFunction':'DiyabetSoyağacıFonksiyon','Age':'Yaş','Outcome':'Sonuç'}, inplace=True)
pd.head()

pd.isnull().sum()   # BOŞ VERİ YOK
print(pd.dtypes)

sekerhastasi=pd[pd.Sonuç==1]
saglikli=pd[pd.Sonuç==0]

plt.scatter(saglikli.Yaş ,saglikli.Glikoz,color="green",label="Sağlıklı",alpha=0.4)
plt.scatter(sekerhastasi.Yaş ,sekerhastasi.Glikoz,color="red",label="sekerhastasi",alpha=0.4)
plt.xlabel("Yaş")
plt.ylabel("Glikoz")
plt.legend()
plt.show()

X = pd.drop(['Sonuç'], axis = 1)
y = pd['Sonuç']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=42)

from sklearn.preprocessing import StandardScaler       # StandardScaler ile model daha etkili bir şekilde eğitilebilir ve test edilebilir.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6, metric='minkowski')
knn.fit(X_train,y_train)


y_pred = knn.predict(X_test)    # X_test verisi üzerinde tahmin yapar ve y_pred değişkenine atar

from sklearn import metrics
print("Eğitim seti doğruluğu:: ", metrics.accuracy_score(y_train, knn.predict(X_train)))
print("Test seti doğruluğu: ", metrics.accuracy_score(y_test, y_pred))

 # Şöyle Birşey Yapalım

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

dogruluk = []

for i in range(1, 31):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train, y_train)
    dogruluk_orani = metrics.accuracy_score(y_test, knn2.predict(X_test))
    dogruluk.append((i, dogruluk_orani))
    print(f"i = {i}   Doğruluk = {dogruluk_orani}")  # Doğruluk oranını yüzde olarak göster

# En yüksek doğruluk oranını ve ona karşılık gelen i değerini bulma
max_dogruluk = max(dogruluk, key=lambda x: x[1])

print(f"\nEn yüksek doğruluk: i = {max_dogruluk[0]}   Doğruluk = {max_dogruluk[1]}")

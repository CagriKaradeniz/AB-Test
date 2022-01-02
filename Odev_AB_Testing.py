import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
import Utils_cagri as util

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#GÖREV1
#A/B testinin hipotezini tanımlayınız.
df_cntrl=pd.read_excel("ab_testing.xlsx",sheet_name="Control Group")
df_test=pd.read_excel("ab_testing.xlsx",sheet_name="Test Group")
df_test=df_test[["Impression","Click","Purchase","Earning"]]
df_cntrl=df_cntrl[["Impression","Click","Purchase","Earning"]]
    #Kontrol gurubu eski yöntem için örnek dataseti
    #Test grubu yeni yöntem için örnek dataseti.
print(f"Kontrol setindeki ortalama kazanç: {df_cntrl.Earning.mean()}  "
      f"\nTest setindeki ortalama kazanç: {df_test.Earning.mean()}")
    #Mean ortalamaları bakıldığında gözle görüür bir yükseliş var.
    #Bu yükselişin şans eserimi yoksa seçilen örneklerle mi alakalı olup olmadığını anlamak\
    #İçin A\B Testing yöntemi kullanılacaktır.



######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# İki grup ortalaması arasında karşılaştırma yapılmak istenildiğinde kullanılır.

# 1. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 2. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

############################
# Uygulama : İki yöntem arasında İst Ol An Fark var mı?
############################

############################
# Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

test_stat, pvalue = shapiro(df_cntrl["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df_test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#P0 iki data içinde 0.05'den büyük bu ndenle H0 reddedilemez.
#Yani iki sette Normal dağılımı sağlamaktadır.


############################
# Varyans Homojenligi Varsayımı
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df_cntrl["Purchase"],df_test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Çıkan değer yine çok yüksektir ve bu nedenle H0 reddedilemez.
#Varyans Homojendir.

############################
# Hipotezin Uygulanması
############################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

# Eğer normallik sağlanmazsa her türlü nonparametrik test yapacağız.
# Eger normallik sağlanır varyans homojenliği sağlanmazsa ne olacak?
# T test fonksiyonuna arguman gireceğiz.


# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

#GÖREV 2
#Çıkan test sonuçlarının istatistiksel olarak anlamlı olup olmadığını yorumlayınız.
    # Normallik ve Varyans sağlandığı için Parametrik hipotes testi yapılacaktır.
test_stat, pvalue = ttest_ind(df_cntrl["Purchase"],df_test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


    #pvalue değeri 0.05'den büyük olduğu için H0 reddedilemez.
    #İki gurup arasından istatislik olarak fark yoktur.
    #Yapılan yeni strateji işe yaramıştır. ortalamadaki artış istatislik olarkta kanıtlanamamıştır.


#Tek yapılan fonksiyon ile deneme

for i in df_test.columns:
    print(f"{i} değişkeni için:")
    util.A_B_Testing(df_test,df_cntrl,i)


#GÖREV 3
#Hangi testleri kullandınız? Sebeplerini belirtiniz.
    #Normallik ve Varyans kontrolü yaptık ve iki şartında sağladığını gördük.
    #Bu nenle parametrik hipotez testi ttest_ind yöntemini kullandık.

#Görev 4
#Görev 2’de verdiğiniz cevaba göre, müşteriye tavsiyeniz nedir?
    #Bu madde için harici bir powerpoint hazırlanarak teamse yüklenecektir.


plt.figure(1)
sns.barplot(x=df_cntrl.index,y=df_cntrl.Earning)
plt.figure(2)
sns.barplot(x=df_test.index,y=df_test.Earning)
plt.figure(3)
plt.title("Impression Comparing")
plt.xlabel("Impressions")
plt.ylabel("Count")
sns.barplot(x=["Impression_Control","Impression_Test"],y=[df_cntrl["Impression"].mean(),df_test["Impression"].mean()])
plt.figure(4)
plt.title("Click Comparing")
plt.xlabel("Clicks")
plt.ylabel("Count")
sns.barplot(x=["Click_Control","Click_Test"],y=[df_cntrl["Click"].mean(),df_test["Click"].mean()])
plt.figure(5)
plt.title("Purchase Comparing")
plt.xlabel("Purchases")
plt.ylabel("Count")
sns.barplot(x=["Purchase_Control","Purchase_Test"],y=[df_cntrl["Purchase"].mean(),df_test["Purchase"].mean()])
plt.figure(6)
plt.title("Earning Comparing")
plt.xlabel("Earnings")
plt.ylabel("$")
sns.barplot(x=["Earning_Control","Earning_Test"],y=[df_cntrl["Earning"].mean(),df_test["Earning"].mean()])



########################### SON YORUMUM ##########################################

# Facebook sadece satınalma sayfasında revizyon yaptı.
#BU bilgi ışığında control verilerindeki tıklama sayısını teste eşitlersek daha doğru sonuç alabiliriz.

df_test_son=df_test.copy()
df_test_son["Purchase"]=df_cntrl["Click"]*(df_test["Purchase"]/df_test["Click"])

util.A_B_Testing(df_test_son,df_cntrl,"Purchase")


plt.figure(7)
plt.title("Purchase Comparing")
plt.xlabel("Purchases")
plt.ylabel("Count")
sns.barplot(x=["Purchase_Control","Purchase_Test"],y=[df_cntrl["Purchase"].mean(),df_test_son["Purchase"].mean()])

#Bu durumda iki dağılım arasında fark olduğunu göstermektedir.7
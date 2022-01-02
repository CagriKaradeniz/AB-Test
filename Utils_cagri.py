import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

def dataset_yukle(dataset):
    return pd.read_csv(dataset+".csv")

def degisken_tiplerine_ayirma(data,cat_th,car_th):
   """
   Veri:data parametresi ili fonksiyona girilen verinin değişkenlerin sınıflandırılması.
   Parameters
   ----------
   data: pandas.DataFrame
   İşlem yapılacak veri seti

   cat_th:int
   categoric değişken threshold değeri

   car_th:int
   Cardinal değişkenler için threshold değeri

   Returns
   -------
    cat_deg:list
    categorik değişken listesi
    num_deg:list
    numeric değişken listesi
    car_deg:list
    categoric ama cardinal değişken listesi

   Examples
   -------
    df = dataset_yukle("breast_cancer")
    cat,num,car=degisken_tiplerine_ayirma(df,10,20)
   Notes
   -------
    cat_deg + num_deg + car_deg = toplam değişken sayısı

   """


   num_but_cat=[i for i in data.columns if data[i].dtypes !="O" and data[i].nunique() < cat_th]

   car_deg=[i for i in data.columns if data[i].dtypes == "O" and data[i].nunique() > car_th]

   num_deg=[i for i in data.columns if data[i].dtypes !="O" and i not in num_but_cat]

   cat_deg = [i for i in data.columns if data[i].dtypes == "O" and i not in car_deg]

   cat_deg = cat_deg+num_but_cat

   print(f"Dataset kolon/değişken sayısı: {data.shape[1]}")
   print(f"Dataset satır/veri sayısı: {data.shape[0]}")
   print("********************************************")
   print(f"Datasetin numeric değişken sayısı: {len(num_deg)}")
   print(f"Datasetin numeric değişkenler: {num_deg}")
   print("********************************************")
   print(f"Datasetin categoric değişken sayısı: {len(cat_deg)}")
   print(f"Datasetin categoric değişkenler: {cat_deg}")
   print("********************************************")
   print(f"Datasetin cardinal değişken sayısı: {len(car_deg)}")
   print(f"Datasetin cardinal değişkenler: {car_deg}")
   print("********************************************")

   return cat_deg,num_deg,car_deg

def categoric_ozet(data,degisken,plot=False,null_control=False):
    """
    Task
    ----------
    Datasetinde bulunan categoric değişkenlerin değişken tiplerinin sayısını ve totale karşı oranını bulur.
    Ayrıca isteğe bağlı olarak değişken dağılımının grafiğini ve değişken içinde bulunan null sayısını çıkartır.

    Parameters
    ----------
    data:pandas.DataFrame
    categoric değişkenin bulunduğu dataset.
    degisken:String
    Categoric değişken ismi.
    plot:bool
    Fonksiyonda categoric değişken dağılımının grafiğini çizdirmek için opsiyonel özellik.
    null_control:bool
    Fonksiyonda değişken içinde null değer kontolü için opsiyonel özellik

    Returns
    -------
    tablo:pandas.DataFrame
    Unique değişkenlerin ratio olarak oran tablosu
    Examples
    -------
    df=dataset_yukle("titanic")
    cat_deg,num_deg,car_deg=degisken_tiplerine_ayirma(df,10,20)
    for i in cat_deg:
        tablo=categoric_ozet(df,i,True,True)
    """

    print(pd.DataFrame({degisken: data[degisken].value_counts(),
                        "Ratio": 100 * data[degisken].value_counts() / len(data)}))
    tablo=pd.DataFrame({degisken: data[degisken].value_counts(),
                        "Ratio": 100 * data[degisken].value_counts() / len(data)})
    print("##########################################")
    if plot:
        sns.countplot(x=data[degisken], data=data)
        plt.show()
    if null_control:
        print(f"Null veri sayısı: {data[degisken].isnull().sum()}")

    return tablo
def dataset_ozet(data, head=5):
    print("##################### Shape #####################")
    print(f"Satır sayısı: {data.shape[0]}")
    print(f"Kolon sayısı: {data.shape[1]}")

    print("##################### Types #####################")
    print(data.dtypes)

    print("##################### Head #####################")
    print(data.head(head))

    print("##################### Tail #####################")
    print(data.tail(head))

    print("##################### NA Kontrolü #####################")
    print(data.isnull().sum())

    print("##################### Quantiles #####################")
    print(data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    print("##################### Describe Tablosu #####################")
    print(data.describe().T)

def outlier_threshold(data,degisken):
    Q1=data[degisken].quantile(0.01)
    Q3=data[degisken].quantile(0.99)
    Q_Inter_Range=Q3-Q1
    alt_limit=Q1-1.5*Q_Inter_Range
    ust_limit=Q3+1.5*Q_Inter_Range
    return alt_limit,ust_limit
def threshold_degisimi(data,degisken):
    alt_limit,ust_limit=outlier_threshold(data,degisken)
    data[data[degisken]<alt_limit]=alt_limit
    data[data[degisken]>ust_limit]=ust_limit
    return data

def data_hazirlama(data):
    data.dropna(inplace=True)
    data = data[~data["Invoice"].str.contains("C", na=False)]
    data = data[data["Quantity"] > 0]
    data = data[data["Price"] > 0]
    threshold_degisimi(data, "Quantity")
    threshold_degisimi(data, "Price")
    return data

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    #print(product_name)
    return product_name
def kural_olustur_kitap(data):
    frequent_itemsets = apriori(data, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

def kural_olustur(data, id=True,country="Germany"):
    data = data[data['Country'] == country]
    data = create_invoice_product_df(data, id)
    frequent_itemsets = apriori(data, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

def arl_recommender(rules_df, product_id, rec_count=1):

    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

def kullanıcıBased_dataolustur(data1,data2):

    df = data1.merge(data2, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

def item_based_recommender(film, data):
    movie_name = data[film]
    return data.corrwith(movie_name).sort_values(ascending=False).head(10)

def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

def A_B_Testing(data1,data2,degisken):
    """

    Parameters
    ----------
    data1
    data2
    degisken

    Returns
    -------

    """
    p_ths=0.05
    t_stat1,p_value1=shapiro(data1[degisken])
    t_stat2, p_value2 = shapiro(data2[degisken])
    t_stat3, p_value3 = levene(data1[degisken],data2[degisken])
    if p_value1 > p_ths and p_value2 > p_ths and p_value3 > p_ths:
        t_stat, p_value = ttest_ind(data1[degisken], data2[degisken], equal_var=True)
        if p_value > p_ths:
            print(f"{p_value} bu sayı {p_ths}'den büyüktür, H0 reddedilemez. İki data da aynı dağılımdadır")
        else:
            print(f"{p_value} bu sayı {p_ths}'den küçüktür, H0 red edilir. İki Data arasında dağılım farkı vardır")
    elif p_value1 > p_ths and p_value2 > p_ths and p_value3 < p_ths:
        t_stat, p_value = ttest_ind(data1[degisken], data2[degisken], equal_var=True)
        if p_value > p_ths:
            print(f"{p_value} bu sayı {p_ths}'den büyüktür, H0 reddedilemez. İki data da aynı dağılımdadır")
        else:
            print(f"{p_value} bu sayı {p_ths}'den küçüktür, H0 red edilir. İki Data arasında dağılım farkı vardır")
    elif (p_value1 > p_ths and p_value2 < p_ths) or (p_value1 < p_ths and p_value2 > p_ths) or (p_value1 < p_ths and p_value2 < p_ths):
        t_stat, p_value = mannwhitneyu(data1[degisken], data2[degisken])

        if p_value > p_ths:
            print(f"{p_value} bu sayı {p_ths}'den büyüktür, H0 reddedilemez. İki data da aynı dağılımdadır")
        else:
            print(f"{p_value} bu sayı {p_ths}'den küçüktür, H0 red edilir. İki Data arasında dağılım farkı vardır")



    
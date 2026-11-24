import pandas as pd


# df1 = pd.read_csv('D:\PROJECTS\BRAZALLIAN DATASET BY OLIST\olist_customers_dataset.xlsx.csv')
# df2 = pd.read_csv('D:\PROJECTS\BRAZALLIAN DATASET BY OLIST\olist_geolocation_dataset.xlsx.csv')
# df3 = pd.read_csv('D:\PROJECTS\BRAZALLIAN DATASET BY OLIST\olist_order_items_dataset.xlsx.csv')
# df4 = pd.read_csv('D:\PROJECTS\BRAZALLIAN DATASET BY OLIST\olist_order_payments_dataset.xlsx.csv')
# df5 = pd.read_csv('D:\PROJECTS\BRAZALLIAN DATASET BY OLIST\olist_order_reviews_dataset.xlsx.csv')
# df6 = pd.read_csv('D:\PROJECTS\BRAZALLIAN DATASET BY OLIST\olist_orders_dataset.xlsx.csv')
# df7 = pd.read_csv('D:\PROJECTS\BRAZALLIAN DATASET BY OLIST\olist_products_dataset.xlsx.csv')
# df8 = pd.read_csv('D:\PROJECTS\BRAZALLIAN DATASET BY OLIST\olist_sellers_dataset.csv')
# df9 = pd.read_csv('D:\PROJECTS\BRAZALLIAN DATASET BY OLIST\product_category_name_translation.csv')

# print(df1.isnull().sum())
# print(df2.isnull().sum())
# print(df3.isnull().sum())
# print(df4.isnull().sum())
# print(df5.isnull().sum())
# print(df6.isnull().sum())
# print(df7.isnull().sum())
# print(df8.isnull().sum())
# print(df9.isnull().sum())


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('D:\PROJECTS\BRAZALLIAN DATASET BY OLIST\MASTER_DATABASE.csv')
# data = pd.DataFrame(data)
# print(data.describe())
# print(data.dtypes)

df = data.copy()

df['order_approved_at_date'] = pd.to_datetime(df['order_approved_at'], errors = 'coerce')
df['order_delivered_to_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'], errors = 'coerce')
df['order_estimated_delivery_date_encoded'] = pd.to_datetime(df['order_estimated_delivery_date'], errors = 'coerce')
df['review_created_date'] = pd.to_datetime(df['review_creation_date'], errors = 'coerce')
df['price'] = pd.to_numeric(df['price'].astype(str).str.strip(), errors = 'coerce')
print('Encoding successful')
print(df['price'].sum())
df['actual_delivery_delay_day'] = (df['order_delivered_to_customer_date'] - df['order_estimated_delivery_date_encoded']).dt.days
# print(df.head(10))



# print(df['payment_type'].value_counts())

# plt.figure(figsize = (12, 8))
# sns.barplot(df['payment_type'].value_counts(), saturation = 0.75)
# plt.xlabel('Payment Mode')
# plt.ylabel('Total numbers of order')
# plt.title('Distribution of payments mode Barchart')
# plt.tight_layout()
# plt.show()

# # plt.figure(figsize = (12, 8))
# # sns.histplot(df['actual_delivery_delay_day'], element = 'bars', fill = True, lw = 2)
# # plt.xlabel('actual delivery delay days')
# # plt.ylabel('Total number of orders')
# # plt.title('Delivery delay distribution')
# # plt.xlim(-30, 30)
# # plt.tight_layout()
# # plt.show()


top_10_categories = df['product_category_name_english'].value_counts().head(10).index
# plt.figure(figsize = (10, 5))
# sns.countplot(df[df['product_category_name_english'].isin(top_10_categories)], width = 0.8, order = top_10_categories, y = 'product_category_name_english')
# plt.xlabel('total number of orders')
# plt.ylabel('Product Category')
# plt.title('Category wise distribution')
# plt.tight_layout()
# plt.show()


df['sentiment'] = (df['review_score'] >= 4).astype(int)
# print(f'Total number of positive reviews: {(df['sentiment'] == 1).sum()}')

nlp_data = df[df['review_comment_message'].str.strip() != 'No Comment'].reset_index(drop = True)
# print(f'Total number of reviews: {len(nlp_data)}')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X = nlp_data['review_comment_message']
y = nlp_data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

tfidf = TfidfVectorizer(max_features = 5000)
train_data = tfidf.fit_transform(X_train)
test_data = tfidf.transform(X_test)

# print('Text data successfully converted to numbers')
# print(f'Train data range: {train_data.shape}')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

model = LogisticRegression(max_iter = 1000)

model = model.fit(train_data, y_train)

y_pred = model.predict(test_data)

print('Model trained successfully')
print(f'Accuracy score: {accuracy_score(y_test, y_pred): .2f}')
print(f'Classification report: {classification_report(y_test, y_pred)}')


# print((df['review_comment_message']).head(5))
# print('------------------')
# print((df['sentiment']).head(5))

# last_purchase_date = df['order_approved_at_date'].max() + pd.Timedelta(days = 1)
# rfm = df.groupby('customer_unique_id').agg({
#     'order_approved_at_date' : lambda x : (last_purchase_date - x.max()).days,
#     'order_id' : 'count',
#     'price' : 'sum'
# }).reset_index()

# rfm.columns = ['customer_unique_id', 'recency', 'frequency', 'monetry']

# print('RFM table dimensions', rfm.shape)
# print(rfm.head())
# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_test, y_pred))
import joblib

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Files saved successfully!")

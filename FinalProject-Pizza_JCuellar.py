# Julia Cuellar
# DSC 550
# Final project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


# Display pizza place data
def read_file():
    pizza = pd.read_csv('pizzaplace.csv')
    print('Original pizza data:\n', pizza)


# Display described, summarized, and length of pizza place data
def des_sum_len():
    pizza = pd.read_csv('pizzaplace.csv')
    print('Described pizza data:\n', pizza.describe())
    print('Summarized pizza data:\n', pizza.describe(include=['O']))
    print('Length of pizza data:\n', len(pizza))


# Display bar chart of pizza name
def showBar_Pname():
    pizza = pd.read_csv('pizzaplace.csv')
    pizza['name'].value_counts().plot(kind='barh').invert_yaxis()
    plt.title('Pizza name')
    plt.show()


# Display bar chart of pizza size
def showBar_Psize():
    pizza = pd.read_csv('pizzaplace.csv')
    pizza['size'].value_counts().plot(kind='barh')
    plt.title('Pizza size')
    plt.show()


# Display pie chart of pizza type
def showPie_Ptype():
    pizza = pd.read_csv('pizzaplace.csv')
    plt.pie(pizza['type'].value_counts(), autopct=lambda p: f'{p:.2f}%', labels=['classic', 'supreme', 'veggie',
                                                                                 'chicken'])
    plt.title('Pizza type')
    plt.show()


# Display boxplot of pizza price
def showBoxplot_Pprice():
    pizza = pd.read_csv('pizzaplace.csv')
    sns.boxplot(pizza['price'])
    plt.title('Pizza price')
    plt.show()


# Check the nulls from pizza file
def check_null():
    pizza = pd.read_csv('pizzaplace.csv')
    print("Display pizza data with null:\n", pizza.isnull())
    print("Display counts of null from pizza data:\n", pizza.isnull().sum())


# Rename unname column then drop along with id and date
def rename_drop():
    pizza = pd.read_csv('pizzaplace.csv')
    pizza.rename(columns={'Unnamed: 0': 'num'}, inplace=True)
    pizza.drop(['num', 'id', 'date'], axis=1, inplace=True)
    print('Pizza data updated:\n', pizza.head(5))


# Check for outlier in pizza size column by counts
def size_count():
    pizza = pd.read_csv('pizzaplace.csv')
    pizza.rename(columns={'Unnamed: 0': 'num'}, inplace=True)
    pizza.drop(['num', 'id', 'date'], axis=1, inplace=True)
    print('Pizza size count:\n', pizza['size'].value_counts())


# Check for outlier in pizza price column by describe then remove and update
def price_out_r_up():
    pizza = pd.read_csv('pizzaplace.csv')
    pizza.rename(columns={'Unnamed: 0': 'num'}, inplace=True)
    pizza.drop(['num', 'id', 'date'], axis=1, inplace=True)
    print('Pizza price:\n', pizza['price'].describe())
    p_price = pizza[pizza['price'] >= 35].index
    pizza.drop(p_price, inplace=True)
    print('Described pizza price:\n', pizza['price'].describe())
    sns.boxplot(pizza['price'])
    plt.title('Pizza price updated')
    plt.show()


# Redisplay pizza place data with described, summarized, and length
def pizza_up():
    pizza = pd.read_csv('pizzaplace.csv')
    pizza.rename(columns={'Unnamed: 0': 'num'}, inplace=True)
    pizza.drop(['num', 'id', 'date'], axis=1, inplace=True)
    p_price = pizza[pizza['price'] >= 35].index
    pizza.drop(p_price, inplace=True)
    pizza.to_csv('pizza.csv')
    print('Pizza data updated:\n', pizza)
    print('Described pizza data updated:\n', pizza.describe())
    print('Summarized pizza data updated:\n', pizza.describe(include=['O']))
    print('Length of pizza data updated:\n', len(pizza))


# Create a multiple linear regression model for size of pizza vs type of pizza purchased
def reg_model_svt():
    pizza = pd.read_csv('pizzaplace.csv')
    pizza.rename(columns={'Unnamed: 0': 'num'}, inplace=True)
    pizza.drop(['num', 'id', 'date'], axis=1, inplace=True)
    p_price = pizza[pizza['price'] >= 35].index
    pizza.drop(p_price, inplace=True)
    fit = ols('price ~ C(size) + C(type)', data=pizza).fit()
    print("Multiple linear regression model for size of pizza vs type of pizza purchased:\n", fit.summary())
    res = fit.resid
    fig = sm.qqplot(res, fit=True, line="45")
    plt.title('Multiple linear regression plot')
    plt.show()


# Display frequency table for size of pizza vs type of pizza purchased
def showFT_svt():
    pizza = pd.read_csv('pizzaplace.csv')
    pizza.rename(columns={'Unnamed: 0': 'num'}, inplace=True)
    pizza.drop(['num', 'id', 'date'], axis=1, inplace=True)
    p_price = pizza[pizza['price'] >= 35].index
    pizza.drop(p_price, inplace=True)
    pizza_svt = pd.crosstab(pizza['size'], pizza['type'])
    print("Cross table of size of pizza vs type of pizza purchased:\n", pizza_svt)


# Create a simple linear regression model for name of pizza vs price of pizza purchased
def reg_model_nvp():
    pizza = pd.read_csv('pizzaplace.csv')
    pizza.rename(columns={'Unnamed: 0': 'num'}, inplace=True)
    pizza.drop(['num', 'id', 'date'], axis=1, inplace=True)
    p_price = pizza[pizza['price'] >= 35].index
    pizza.drop(p_price, inplace=True)
    fit = ols('price ~ C(name)', data=pizza).fit()
    print("Simple linear regression model for name of pizza vs price of pizza purchased:\n", fit.summary())
    res = fit.resid
    fig = sm.qqplot(res, fit=True, line="45")
    plt.title('Simple linear regression plot')
    plt.show()


# Display plot for name of pizza vs type of pizza purchased
def showPlot_nvp():
    pizza = pd.read_csv('pizzaplace.csv')
    pizza.rename(columns={'Unnamed: 0': 'num'}, inplace=True)
    pizza.drop(['num', 'id', 'date'], axis=1, inplace=True)
    p_price = pizza[pizza['price'] >= 35].index
    pizza.drop(p_price, inplace=True)
    sns.catplot(x='price', y='name', data=pizza)
    plt.title('Price vs Name')
    plt.show()
    pizza_nvp = pd.crosstab(pizza['price'], pizza['name'])
    print("Cross table of price of pizza vs name of pizza purchased:\n", pizza_nvp)
    pizza_name = pizza.groupby('name').count()
    print("Display count of pizza name:\n", pizza_name)
    pizza_price = pizza.groupby('price').count()
    print("Display count of pizza price:\n", pizza_price)


if __name__ == "__main__":
    read_file()
    des_sum_len()
    showBar_Pname()
    showBar_Psize()
    showPie_Ptype()
    showBoxplot_Pprice()
    check_null()
    rename_drop()
    size_count()
    price_out_r_up()
    pizza_up()
    reg_model_svt()
    showFT_svt()
    reg_model_nvp()
    showPlot_nvp()

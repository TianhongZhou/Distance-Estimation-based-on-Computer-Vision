import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt


def linear_regression(input_x_train, input_x_test, input_y_train, input_y_test):
    linear_regress = LinearRegression()
    pipeline = Pipeline([("linear_regression", linear_regress)])
    pipeline.fit(input_x_train.values, input_y_train)
    # train_score = pipeline.score(input_x_train.values, input_y_train)
    # test_score = pipeline.score(input_x_test.values, input_y_test)
    # print("linear regression model: train score: "
    #       + str(train_score) + ", test score: " + str(test_score))
    return pipeline


def polynomial_regression(input_x_train, input_x_test, input_y_train, input_y_test, degree):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regress = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regress)])
    pipeline.fit(input_x_train.values, input_y_train)
    # train_score = pipeline.score(input_x_train.values, input_y_train)
    # test_score = pipeline.score(input_x_test.values, input_y_test)
    # print(str(degree) + " degree polynomial model: train score: "
    #       + str(train_score) + ", test score: " + str(test_score))
    return pipeline


def random_regression_forest(input_x_train, input_x_test, input_y_train, input_y_test):
    regression_forest = RandomForestRegressor(random_state=0)
    parameters = {"n_estimators": [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]}
    clf = GridSearchCV(regression_forest, parameters, cv=5)
    clf.fit(input_x_train.values, input_y_train)
    regression_forest = RandomForestRegressor(n_estimators=52, random_state=0)
    pipeline = Pipeline([("random_forest", regression_forest)])
    pipeline.fit(input_x_train.values, input_y_train)
    train_score = pipeline.score(input_x_train.values, input_y_train)
    test_score = pipeline.score(input_x_test.values, input_y_test)
    print("52 trees random forest model: train score: "
          + str(train_score) + ", test score: " + str(test_score))
    return pipeline


def k_neighbors_regression(input_x_train, input_x_test, input_y_train, input_y_test):
    # knn = KNeighborsRegressor()
    # parameters = {"n_neighbors": [5, 6, 7, 8, 9]}
    # clf = GridSearchCV(knn, parameters, cv=5)
    # clf.fit(input_x_train.values, input_y_train)
    knn = KNeighborsRegressor(8)
    pipeline = Pipeline([("knn", knn)])
    pipeline.fit(input_x_train.values, input_y_train)
    # train_score = pipeline.score(input_x_train.values, input_y_train)
    # test_score = pipeline.score(input_x_test.values, input_y_test)
    # print("8 neighbors KNN model: train score: "
    #       + str(train_score) + ", test score: " + str(test_score))
    return pipeline


def adaBoost_regression(input_x_train, input_x_test, input_y_train, input_y_test):
    # adaboost = AdaBoostRegressor(random_state=0)
    # parameters = {"n_estimators": [5, 6, 7, 8, 9]}
    # clf = GridSearchCV(adaboost, parameters, cv=5)
    # clf.fit(input_x_train.values, input_y_train)
    adaboost = AdaBoostRegressor(n_estimators=7, random_state=0)
    pipeline = Pipeline([("adaboost", adaboost)])
    pipeline.fit(input_x_train.values, input_y_train)
    # train_score = pipeline.score(input_x_train.values, input_y_train)
    # test_score = pipeline.score(input_x_test.values, input_y_test)
    # print("7 estimators adaboost model: train score: "
    #       + str(train_score) + ", test score: " + str(test_score))
    return pipeline


x_columns = ["m_xmin", "m_xmax", "m_ymin", "m_ymax", "m_width", "m_height", "m_size", "m_wh_ratio",
             "h_xmin", "h_xmax", "h_ymin", "h_ymax", "h_width", "h_height", "h_size", "h_wh_ratio",
             "x_min_diff", "x_max_diff", "y_min_diff", "y_max_diff", "width_ratio", "height_ratio", "size_ratio"]
df = pd.read_csv("./data.csv")
df = df.drop(df[df["dist"] == -1].index)
# df = df.sort_values("scaled_dist", ascending=True, inplace=False, kind='quicksort', ignore_index=True)
X = df[x_columns]
y = df["scaled_dist"]
# y = df["dist"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)
# y_test = y_test.sort_index()

linear_model = linear_regression(X_train, X_test, y_train, y_test)
# y_pred = linear_model.predict(X.values)
# for i in range(len(y_pred)):
#     y_pred[i] = y_pred[i] + 90
# mse = mean_squared_error(y_test, y_pred)
# print("linear model MSE: " + str(mse))
# r2 = r2_score(y_test, y_pred)
# print("linear model R2: " + str(r2))
# plt.plot(y_pred, '-', linewidth=2.5)
# plt.plot(y_test, '-', linewidth=2.5)
# plt.scatter(X_test.index, y_test)
# plt.xlabel("index")
# plt.ylabel("distance")
# plt.legend(["pred", "test"])
# plt.title("Linear model")
# plt.show()

polynomial_model = polynomial_regression(X_train, X_test, y_train, y_test, 2)
# y_pred = polynomial_model.predict(X_test.values)
# mse = mean_squared_error(y_test, y_pred)
# print("polynomial model MSE: " + str(mse))
# r2 = r2_score(y_test, y_pred)
# print("polynomial model R2: " + str(r2))
# plt.scatter(X_test.index, y_pred, marker='*', s=100)
# plt.scatter(X_test.index, y_test)
# plt.xlabel("index")
# plt.ylabel("distance")
# plt.legend(["pred", "test"])
# plt.title("Polynomial model")
# plt.show()

random_forest_model = random_regression_forest(X_train, X_test, y_train, y_test)
y_pred = random_forest_model.predict(X.values)
# for i in range(len(y_pred)):
#     y_pred[i] = y_pred[i] + 45
# mse = mean_squared_error(y_test, y_pred)
# print("random forest model MSE: " + str(mse))
# r2 = r2_score(y_test, y_pred)
# print("random forest model R2: " + str(r2))
# plt.plot(X.index, y_pred, '-', linewidth=2.5)
# plt.plot(y.index, y, linewidth=2.5)
# plt.xlabel("Index", fontsize=10)
# plt.ylabel("Distance", fontsize=10)
# plt.legend(["LR_pred", "RF_pred", "Real"], fontsize=10)
# plt.title("Comparison of LR and RF Model")
# plt.show()

knn_model = k_neighbors_regression(X_train, X_test, y_train, y_test)
# y_pred = knn_model.predict(X_test.values)
# mse = mean_squared_error(y_test, y_pred)
# print("KNN model MSE: " + str(mse))
# r2 = r2_score(y_test, y_pred)
# print("KNN model model R2: " + str(r2))
# plt.scatter(X_test.index, y_pred)
# plt.scatter(X_test.index, y_test)
# plt.xlabel("index")
# plt.ylabel("distance")
# plt.legend(["pred", "test"])
# plt.title("KNN model")
# plt.show()

adaboost_model = adaBoost_regression(X_train, X_test, y_train, y_test)
# y_pred = adaboost_model.predict(X_test.values)
# mse = mean_squared_error(y_test, y_pred)
# print("Adaboost model MSE: " + str(mse))
# r2 = r2_score(y_test, y_pred)
# print("Adaboost model model R2: " + str(r2))
# plt.scatter(X_test.index, y_pred)
# plt.scatter(X_test.index, y_test)
# plt.xlabel("index")
# plt.ylabel("distance")
# plt.legend(["pred", "test"])
# plt.legend(["linear_pred", "poly_pred", "random_forest_pred", "knn_pred", "adaboost_pred", "test"])
# plt.title("Distance Estimation Model Comparison")
# plt.show()

pkl_filename = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\models\linear_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(linear_model, file)
pkl_filename = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\models\polynomial_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(polynomial_model, file)
pkl_filename = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\models\random_forest_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(random_forest_model, file)
pkl_filename = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\models\knn_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(knn_model, file)
pkl_filename = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\models\adaboost_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(adaboost_model, file)

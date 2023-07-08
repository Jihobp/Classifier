import xgboost
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


def input_data():
    input_file_name = input()
    return input_file_name

# 1. 데이터 불러오기
def import_data(input_file_name) :
    data = pd.read_csv(input_file_name)
    data= data.drop(columns=['Balance','RunOut'])
    data= data.replace({'OK':0, 'NG':1})
    return data

# 2. X,Y 데이터 규별 및 학습 데이터와 테스트 데이터 구별
def split_data(data) :
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) 
    col = []
    for i in range(0, len(data.columns)):
        col.append(data.columns[i])
    return x_train, x_test, y_train, y_test, col

# 3. 분류 모델 명시 및 학습하기
def train_model(x_train, x_test, y_train, y_test) :
    # 3-1. RandomForest_Classifier
    rf_clf = RandomForestClassifier(random_state=0)
    rf_clf.fit(x_train, y_train)
    
    # 3-2. DecisionTree Classifier
    dt_clf = DecisionTreeClassifier(random_state=156)
    dt_clf.fit(x_train, y_train)
    
    # 3-3. LightGBM_Classifier
    lgbm_clf = LGBMClassifier(n_estimators=200, learning_rate=0.06)
    lgbm_clf.fit(x_train, y_train)

    # 3-4. XGBClassifier
    xgb_clf = xgboost.XGBClassifier(n_estimators=200, learning_rate = 0.06, gamma=0, subsaple=0.75, colsample_bytree=1, max_depth=7)
    xgb_clf.fit(x_train, y_train, early_stopping_rounds = 100, eval_metric = "logloss", eval_set = [ (x_test, y_test) ], verbose=True)

    return rf_clf, dt_clf, lgbm_clf, xgb_clf

# 4. 예측하기
def prediction(rf_clf, dt_clf, lgbm_clf, xgb_clf, x_test, col):
    # 4-1. RandomForest Classifier
    rf_pred = rf_clf.predict(x_test)

    # 4-2. DecisionTree Classifier
    dt_pred = dt_clf.predict(x_test)
    
    # 4-3. LGBMClassifier
    lgbm_pred = lgbm_clf.predict(x_test)

    # 4-4. XGBClassifier
    xgb_pred = xgb_clf.predict(x_test)
    xgb_pred = pd.DataFrame(xgb_pred, columns=[col[-1]])
    xgb_pred = xgb_pred.set_index(x_test.index.values)
    return rf_pred, dt_pred, lgbm_pred, xgb_pred




# 5. 평가
def evaluation_model(y_test, rf_pred, dt_pred, lgbm_pred, xgb_pred):
    rf_confusion = confusion_matrix(y_test, rf_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_pred)

    dt_confusion = confusion_matrix(y_test, dt_pred)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    dt_precision = precision_score(y_test, dt_pred)
    dt_recall = recall_score(y_test, dt_pred)
    dt_f1 = f1_score(y_test, dt_pred)
    dt_auc = roc_auc_score(y_test, dt_pred)

    lgbm_confusion = confusion_matrix(y_test, lgbm_pred)
    lgbm_accuracy = accuracy_score(y_test, lgbm_pred)
    lgbm_precision = precision_score(y_test, lgbm_pred)
    lgbm_recall = recall_score(y_test, lgbm_pred)
    lgbm_f1 = f1_score(y_test, lgbm_pred)
    lgbm_auc = roc_auc_score(y_test, lgbm_pred)


    xgb_confusion = confusion_matrix(y_test, xgb_pred)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_precision = precision_score(y_test, xgb_pred)
    xgb_recall = recall_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_pred)

    print('Random_Forest_Classifier: \n 오차행렬')
    print(rf_confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC: {4:.4f}'.format(rf_accuracy, rf_precision, rf_recall, rf_f1, rf_auc))
    print()
    print('Decision_Tree_Classifier \n 오차행렬')
    print(dt_confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC: {4:.4f}'.format(dt_accuracy, dt_precision, dt_recall, dt_f1, dt_auc))
    print()
    print('LightGBM_Classifier \n 오차행렬')
    print(lgbm_confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC: {4:.4f}'.format(lgbm_accuracy, lgbm_precision, lgbm_recall, lgbm_f1, lgbm_auc))
    print()
    print('XGBoost_Classifier \n 오차행렬')
    print(xgb_confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC: {4:.4f}'.format(xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_auc))
    dict = {rf_accuracy : 'Random_Forest_Classifier', dt_accuracy : 'Decision_Tree_Classifier', lgbm_accuracy :'LightGBM_Classifier', xgb_accuracy :'XGBoost_Classifier'}
    max_rate = "%.4f"%max(dict)
    print('입력된 데이터에서 가장 정확도가 높은 분류모델은 ', dict.get(max(dict)), '이며 해당 모델의 정확도는 ',max_rate,'이다.')

def relation_map(data, col) :
    refine_data = data.drop(['Cyl5temp', 'oiltemp'], axis=1)
    plt.figure(figsize=(15, 15))
    sns.heatmap(data = refine_data.corr(), annot=True, fmt='.2f', linewidths=.5, cmap='Blues')
    sns.clustermap(refine_data.corr(), annot = True, cmap= 'RdYlBu_r', vmin = -1, vmax=1)
    plt.show()
    revise_col = col

    if 'Result' in revise_col :
        col.remove('Result')
    else :
        pass
    return revise_col

def rf_importance_graph(revise_col, rf_clf, dt_clf, lgbm_clf, xgb_clf) :
    
    # Random_Forest_feature 중요성
    rf_importance = rf_clf.feature_importances_
    rf_importance.sort()
    plt.bar(revise_col, rf_importance)
    plt.xticks(rotation=45)
    plt.title('Random Forest Feature Importances')
    plt.show()

    #Decision_tree_Classifier 중요성
    dt_importance = dt_clf.feature_importances_
    dt_importance.sort()
    plt.bar(revise_col, dt_importance)
    plt.xticks(rotation=45)
    plt.title('Decision Tree Feature Importances')
    plt.show()

    #LightGBM_Classifier 중요성
    lgbm_importance = lgbm_clf.feature_importances_
    lgbm_importance.sort()
    plt.bar(revise_col, lgbm_importance)
    plt.xticks(rotation=45)
    plt.title('LightGBM Classifier Feature Importances')
    plt.show()

    #XGBoost_Classifer 중요성
    xgb_importance = xgb_clf.feature_importances_
    xgb_importance.sort()
    plt.bar(revise_col, xgb_importance)
    plt.xticks(rotation=45)
    plt.title('XGBoost Classifier Feature Importances')
    plt.show()
   

def compare_pred_real (y_test, rf_pred, dt_pred, lgbm_pred, xgb_pred) :
    rf_idx = []
    for i in range(0, len(y_test)):
        rf_idx.append(i)

    plt.figure(figsize=(15,5))
    plt.plot(rf_idx[:100], rf_pred[:100], label='predict')
    plt.plot(rf_idx[:100], y_test[:100], label='realistic')
    plt.title('Random Forest comparing predict and realistic data')
    plt.legend()
    plt.show()


    dt_idx = []
    for i in range(0, len(y_test)):
        dt_idx.append(i)

    plt.figure(figsize=(15,5))
    plt.plot(dt_idx[:100], dt_pred[:100], label='predict')
    plt.plot(dt_idx[:100], y_test[:100], label='realistic')
    plt.title('Decision Tree Classifier comparing predict and realistic data')
    plt.legend()
    plt.show()


    lgbm_idx = []
    for i in range(0, len(y_test)):
        lgbm_idx.append(i)

    plt.figure(figsize=(15,5))
    plt.plot(lgbm_idx[:100], lgbm_pred[:100], label='predict')
    plt.plot(lgbm_idx[:100], y_test[:100], label='realistic')
    plt.title('LightGBM Classifier comparing predict and realistic data')
    plt.legend()
    plt.show()

    xgb_idx = []
    for i in range(0, len(y_test)):
        xgb_idx.append(i)

    plt.figure(figsize=(15,5))
    plt.plot(xgb_idx[:100], xgb_pred[:100], label='predict')
    plt.plot(xgb_idx[:100], y_test[:100], label='realistic')
    plt.title('XGBoost Classifier comparing predict and realistic data')
    plt.legend()
    plt.show()
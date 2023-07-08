import xgboost
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #비쥬얼라이징
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score, roc_auc_score #정확도 확인
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

#Man's notion >> 머신러닝 관련 정리내용 !!!!! 참고. 
# 정확도 기준으로 classifier 모델 추천하는 알고리즘 생성
# 어떤 사출 조건이 제품의 불량 유무에 큰 영향을 주는지 

def input_data():
    input_file_name = input()
    return input_file_name

# 1. 데이터 불러오기
def import_data(input_file_name) :
    data = pd.read_csv(input_file_name)
    data= data.drop(columns=['Balance','RunOut'])
    data= data.replace({'OK':0, 'NG':1}) #ok:정상품 ng:불량품 // 사출 조건에 따른 제품 불량 유무를 학습하여 예측 : 사출조건=제품 생성할때의 기초조건 온도, 습도 등
    return data

# 2. X,Y 데이터 규별 및 학습 데이터와 테스트 데이터 구별

def split_data(data) : # : 2차원 배열(행은 데이터가 있는데 기압, 온도, 습도. 마지막 열에는 result 즉 제품불량 유닛.
    x = data.iloc[:, :-1] #x=사출조건 데이터 (끝에서 끝이전까지)
    y = data.iloc[:, -1:] #y=제품불량 유무 (끝에것만) 0=ok, 1=ng, -1: 0에서 거꾸로 가서 끝, 즉 리스트의 제일 last data를 불러올수 있게 하는 것. 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)  
    # train > test (학습>정답예측) 0.3 // 전체에서 0.3만큼의 분량을 test data로 하겠다. 10개의 데이터가 있다면 3개만큼의 test data, 나머지 0.7은 학습
    col = [] #column 열, list 변수명
    #컴퓨터는 바보라 데이터를 명시해줘야 한다. 
    #행들은 데이터고, 열은 데이터를 명시 및 카테고리화 해서 학습시켜주는 것. 기압, 온도, 습도의 명시화. 
    # col이 필요할때가 있다. fit만 시키면 카테고리가 뭔지 모른다. 열 0번이 기압, 열 1번이 온도다 해서 명시시킨것. 
    for i in range(0, len(data.columns)):
        col.append(data.columns[i])
    return x_train, x_test, y_train, y_test, col # 학습시킨 결과를 예측시키는것. xtrain test를 통해 사출조건 데이터를 학습시킴.
    #test는 다 result가 있다. 
    # train 데이터는 0.7 
    # test 데이터는 0.3
    # x가 변수 y 가 결과값. x가 사출조건이면 y는 제품불량 유무.
    # fit :  학습 시키는 것. 모델에 맞춰보는 것.  = x_train & y_train 
    # (머신러닝) 좌표값이 있을경우 특정 좌표값을 점찍어서 있다면 "사람이다"로 있는걸로 인식. 특정패턴의 정확한 기준으로 뭐다 아니다 정의를 내리는 것.
    # (딥러닝) 답이 없다. 대신 패턴이 있는 것. 사람이라고 군집화(cluster)로 묶는다. 강아지, 고양이 등 비슷한 패턴으로 묶인 것. 정답이 아닐 확률이 높다. 패턴을 찾는 딥러닝이 그래서 어려운것.
    # 딥러닝은 군집화. 군집화 속의 패턴이 있고 그걸 찾는 것.
    # 머신러닝 define ? 딥러닝은 classify. 머신러닝과 딥러닝의 차이는 사람의 개입 유무.
    # 머신러닝은 인간이 결과를 주는 것. 데이터를 분석 및 축적. 패턴 추출 작업 유.
    # 딥러닝은 인간이 개입하지 않고 컴퓨터가 스스로 정답을 추론하도록 유도. 패턴 추출 작업 무.
    # 분류 : 종류 예측(어뷰징 검출)  /// 회귀 : 연속된 값을 예측. >>> 둘다 머신러닝과 딥러닝에 속함. 


# 3. 분류 모델 명시 및 학습하기
def train_model(x_train, x_test, y_train, y_test) :
    # 3-1. RandomForest_Classifier
    rf_clf = RandomForestClassifier(random_state=0) #random_state : 컴퓨터는 단순하다. 단순한 12345만 학습시키면 그것만 학습하고 찾음. 다양한 난수패턴을 학습시켜야한다. 
    rf_clf.fit(x_train, y_train) #fit : 패턴을 학습하세요 :D
    
    # 3-2. DecisionTree Classifier 결정트리 #man's notion 참고
    dt_clf = DecisionTreeClassifier(random_state=156)
    dt_clf.fit(x_train, y_train)
    
    # 3-3. LightGBM_Classifier
    lgbm_clf = LGBMClassifier(n_estimators=200, learning_rate=0.06)
    lgbm_clf.fit(x_train, y_train)

    # 3-4. XGBClassifier 
    xgb_clf = xgboost.XGBClassifier(n_estimators=200, learning_rate = 0.06, gamma=0, subsaple=0.75, colsample_bytree=1, max_depth=7)
    xgb_clf.fit(x_train, y_train, early_stopping_rounds = 100, eval_metric = "logloss", eval_set = [ (x_test, y_test) ], verbose=True)
    # 학습할때의 최적조건을 적어 놓은 것. 목적 : 정확도 향상 
    # 최적의 조건을 찾는 공식?????????????????? 
    # hyper parameter를 튜닝함. grid search를 통해서. 그 방식이 명시되어 있는 것. 일일히 환경값을 주면서 성능을 일일히하며 hyper parameter 값을 알려줌.
    # grid search 가 문제가 되지만 그래도 쓴다. 1,2,3,4 너무 세세해서 시간이 오래 걸림.
    # 대체로 베이지안 최적화. >> 책에 있지만 실제론 x
    
    return rf_clf, dt_clf, lgbm_clf, xgb_clf #해당 모델의 데이터를 입력해서 학습시킨 것. 즉, fit 한 것 ㅇㅅㅇ/ 


# 4. 예측하기 = 분류(정답을 이미 줌) 이런 조건이면 불량이다 판단 유.
def prediction(rf_clf, dt_clf, lgbm_clf, xgb_clf, x_test, col):
    # 4-1. RandomForest Classifier
    rf_pred = rf_clf.predict(x_test) #예측한 값

    # 4-2. DecisionTree Classifier
    dt_pred = dt_clf.predict(x_test) 
    
    # 4-3. LGBMClassifier
    lgbm_pred = lgbm_clf.predict(x_test)

    # 4-4. XGBClassifier 
    xgb_pred = xgb_clf.predict(x_test)
    xgb_pred = pd.DataFrame(xgb_pred, columns=[col[-1]]) # 예측한것의 컬럼명. 제일 끝의 배열
    xgb_pred = xgb_pred.set_index(x_test.index.values) # index 만들어 준것. xgb_classifier의 특성 그냥 루틴화된것. 어떤걸 순서대로 해야하는지 
    return rf_pred, dt_pred, lgbm_pred, xgb_pred




# 5. 평가
def evaluation_model(y_test, rf_pred, dt_pred, lgbm_pred, xgb_pred):
    # 평가 지표
    # 왜 정확도로? 정확도를 기준으로? 해당 프로젝트에선 다른 평가지표보다 단순하게 불량 유무를 체크하기 위해서기 때문에, 정확도가 가장 중요하다고 판단해서
    # Acuuracy 를 픽했다!!!!!!!!!!!!!!!~~~~~~~~ 
    # 늘 생각하는거지만 데이터와 코드엔 목적성이 중요하다. 
    # 이후에 목적이 달라지면 그것에 맞는 평가지표가 필요하기 때문에 활용을 했다. 
    # 평가지표의 구별방법은? = 프로젝트의 목적에 따라 다르고, 프로젝트 경험이 많이 없기 때문에 회사에서 많이 알려준다면 따라서 갈것입니다. 
    rf_confusion = confusion_matrix(y_test, rf_pred) #x_test predict(예측) 한 결과값과 y_test(실제) 결과값 비교. 
    rf_accuracy = accuracy_score(y_test, rf_pred) # 정확도
    rf_precision = precision_score(y_test, rf_pred)  # 정밀도
    rf_recall = recall_score(y_test, rf_pred) # 재현률
    rf_f1 = f1_score(y_test, rf_pred) # 정밀도와 재현율을 기반으로 명확하게 성능을 평가하는 것.
    rf_auc = roc_auc_score(y_test, rf_pred) # false positive rate가 변할 때 true positive rate가 어떻게 벼하는지를 나타내는 곡선. 
                                            # fpr : 컴퓨터가 positive 판단했는데 실제값는 negative. 그 값이 얼만지 비율로 된것. ex) 지오가 있지만 robot으로 판단한 것.
                                            # tpr : 컴퓨터가 positive 판단했는데 실제값도 positive. 그 값이 얼만지 비율로 된것. ex) 지오가 지오로 판단된 것. 
    #위의 것들 숫자가 높을수록 좋은 것. ex) 1이 나오는 순간 성능이 가장 높다.
    #분류 모델 명칭. dt, lgbm, xgb.
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
    # 정규식 : 데이터 표현형식. 이걸 잘하면 개발을 잘하는 것 ;ㅅ; use ai!
    # index값(0번째 배열의 index값이지만, 어떻게 표현할지 즉, 소수점 넷째자리까지 rf_accuracy를 표현하라.) 
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

# feature 와 result 의 상관관계 
def relation_map(data, col) :  
    refine_data = data.drop(['Cyl5temp', 'oiltemp'], axis=1) #삭제 코드 . 두 컬럼 값이 result랑 상관이 없다. 그래서, 그래프화가 안됨. 그래서 삭제.
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

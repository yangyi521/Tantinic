#使用RandomForestClassifier填补缺失的年龄属性
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
data_train = pd.read_csv("train.csv")
def set_missing_ages(df):#传进来的参数是整个的数据
    #age_df = df[['age','boat','room','ticket','survived']]
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    #将所有的数据根据年龄的有跟无，分成两类
    know_age = age_df[age_df.Age.notnull()].as_matrix()
    unknow_age = age_df[age_df.Age.isnull()].as_matrix()
    #将知道姓名的数据分成两部分 即将预测的年龄是一个，然后剩下的那一部分是一个
    y = know_age[:,0]
    X = know_age[:,1:]
    #创建randomForestRegerssion对象，并且将数据喂给他们
    #关于决策树的算法，n_estimators为决策树的算法，越多越好，但是相对应的性能就会越差，
    #                 n_job=1 并行job的个数，这个在ensemble算法中非常重要，
    #                 1=不并行 n = n个并行， -1 CPU有多少core，就启动多少job
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    #将数据喂进去
    rfr.fit(X,y)
    #用得到的模型进行未知年龄结果的预测
    predictedAges = rfr.predict(unknow_age[:,1::])
    #用预测到的数据填充原来缺失的数据
    df.loc[(df.Age.isnull()),'Age'] = predictedAges
#     print(df)
# def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    # 接下来我们要接着做一些数据预处理的工作，比如scaling，将一些变化幅度较大的特征化到[-1,1]之内
    # 这样可以加速logistic regression的收敛
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'])
    df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'])
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
    # 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()#得到训练模型
    # y即Survival结果
    y = train_np[:, 0]
    # X即特征属性值
    X = train_np[:, 1:]
    # fit到RandomForestRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)
    ##以上的代码标识训练结束
    ##以下的代码标识开始预测，测试
    data_test = pd.read_csv("test.csv")
    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
    # 接着我们对test_data做和train_data中一致的特征变换
    # 首先用同样的RandomForestRegressor模型填上丢失的年龄
    tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].as_matrix()
    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges
    data_test.loc[(data_test.Cabin.notnull()), 'Cabin'] = "Yes"
    data_test.loc[(data_test.Cabin.isnull()), 'Cabin'] = "No"
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
    df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)

    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test)
    result = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv("logistic_regression_predictionsvvv.csv", index=False)


    print(df)
#titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

set_missing_ages(data_train)



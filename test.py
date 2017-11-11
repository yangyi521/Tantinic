import pandas as pd
titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
#print(titanic)
#将pclass,age,sex等组成一个矩阵，参数矩阵
X=titanic[['pclass','age','sex']]
#将生存信息组成矩阵y
y=titanic['survived']
print(y)
#print(X['age'])
#age中的数据有所缺失，所以需要对age中的数据进行补充
X['age'].fillna(X['age'].mean(),inplace=True)
#print(X['age'])


#决策树模型预测乘客的生还情况
#对数据进行分割
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)
#使用scikit-learn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer
#进行转换器的定义
vec = DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
#对测试数据也需要进行转换
X_test=vec.fit_transform(X_test.to_dict(orient='record'))
print(vec.feature_names_)

#决策树
from sklearn.tree import DecisionTreeClassifier
#对决策树进行初始化
dtc = DecisionTreeClassifier()
#对数据进行训练
dtc.fit(X_train,y_train)
#用训练好的决策树模型对测试特征数据进行预测
y_predict=dtc.predict(X_test)
print(y_predict)

#决策树模型对泰坦尼克号乘客是否生还的预测性能
from sklearn.metrics import classification_report
#s输出预测的准确性
print(dtc.score(X_test,y_test))
#输出更加详细的分类性能
#关于该函数classification_report(y_predict,y_test,target_names=['died','survived']))的使用，
#显示主要的分类指标，返回每一个标签的精确，召回率及F1的值，主要参数说明：
#精度(precision) = 正确预测的个数(TP)/被预测正确的个数(TP+FP)
#召回率(recall)=正确预测的个数(TP)/预测个数(TP+FN)
#F1 = 2*精度*召回率/(精度+召回率
print(classification_report(y_predict,y_test,target_names=['died','survived']))
import csv

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
import numpy as np
from sklearn import metrics as skl
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.calibration import calibration_curve

values = np.arange(0.01, 0.51, 0.01)
# for ii in values:
train = pd.read_csv("Result/Tumor_L3_clinical/train1.csv", encoding='gb18030', header=None)
test = pd.read_csv("Result/ex_test/Tumor_L3_clinical.csv", encoding='gb18030', header=None)

train_data = train.iloc[:, 2:]
train_label = train.iloc[:, 1]
test_data = test.iloc[:, 2:]
test_label = test.iloc[:, 1]

# tumor_L3_clinical
smo1 = BorderlineSMOTE(random_state=60, kind='borderline-2', k_neighbors=18, m_neighbors=15)
train_data, train_label = smo1.fit_resample(train_data, train_label)
rf_classifier = RandomForestClassifier(n_estimators=5, random_state=44, oob_score=False, bootstrap=False,
                                       min_weight_fraction_leaf=0.05)
rf_classifier.fit(train_data, train_label)
y_pred = rf_classifier.predict(test_data)
clf = CalibratedClassifierCV(estimator=rf_classifier, method="sigmoid", cv=25)
clf.fit(train_data, train_label)
y_proba = clf.predict_proba(test_data)
threshold = 0.48


result = np.where(y_proba > threshold, 1, 0)
print(result[:, 1])
matrix = skl.confusion_matrix(test_label, result[:, 1], labels=[0, 1])  #
tn, fp, fn, tp = matrix.ravel()
auc1 = skl.roc_auc_score(test_label, y_proba[:, 1])  #
sen = tp / (tp + fn)
spe = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
acc = accuracy_score(test_label, result[:, 1])
list = []
print('auc: ', auc1)
print('acc: ', acc)  #
print('sensitivity: ', tp / (tp + fn))
print('specificity: ', tn / (tn + fp))
print('ppv:', ppv)
print('npv:', npv)

df_test_label = test_label.to_frame(name='true_label')

# 创建一个 DataFrame 存储 test_label 和 y_proba 的概率值
df_proba = pd.DataFrame({'probability': y_proba[:, 1]})  # 假设只取了正类的概率

# 将两个 DataFrame 合并为一个
df = pd.concat([df_test_label, df_proba], axis=1)

# 保存 DataFrame 到 CSV 文件
df.to_csv('ex-test-bad.csv', index=False)


predict_pro_np = np.array(y_proba[:, 1])
test_label_np = np.array(test_label)
np.save('./Result/visualized/dca/ex_Tumor_prepro', predict_pro_np)
np.save('./Result/visualized/dca/ex_Tumor_label', test_label_np)

# Bootstrap计算置信区间
n_iterations = 100
stats = []
for _ in range(n_iterations):
    indices = np.random.choice(len(test_data), len(test_data), replace=True)
    test_data_bootstrap = test_data.iloc[indices]
    test_label_bootstrap = test_label.iloc[indices]
    y_proba_bootstrap = clf.predict_proba(test_data_bootstrap)
    auc_bootstrap = skl.roc_auc_score(test_label_bootstrap, y_proba_bootstrap[:, 1])
    stats.append(auc_bootstrap)

# 计算置信区间
alpha = 0.95
p = ((1.0 - alpha) / 2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha + ((1.0 - alpha) / 2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print(f'95% Confidence Interval for AUC: [{lower:.3f}, {upper:.3f}]')
fpr,tpr,threshold = roc_curve(test_label, result[:,1])
roc_auc = auc(fpr,tpr)
print(type(fpr))
print(tpr)
print(roc_auc)

tra_ = '训练集'
val_ = '验证集'
test_ = '测试集'
log_path = './Result/Fat/resnet18.csv'
file = open(log_path, 'a+', encoding='utf-8', newline='')
csv_writer = csv.writer(file)
csv_writer.writerow([f'数据集', 'AUC', 'ACC', 'SEN', 'SPE', 'PPV', 'NPV'])
csv_writer.writerow(['训练集', "%.3f"%auc1, "%.3f"%acc, "%.3f"%sen, "%.3f"%spe, "%.3f"%ppv, "%.3f"%npv])
file.close()

fpr, tpr, threshold = roc_curve(test_label, y_proba[:, 1])
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='blue',
         lw=lw, label='ROC curve (area = %0.2f)' % auc1) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

np.save('./roc/fpr-ex-test-GBDT', fpr)
np.save('./roc/tpr-ex-test-GBDT', tpr)

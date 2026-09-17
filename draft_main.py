import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# 1. 读取数据
# ============================================================

def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')

    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values

    X_test_unlabeled = test_df['text'].astype(str).tolist()

    return X_train, y_train, X_test_unlabeled


X_train, y_train, X_test_unlabeled = load_data()


print("=" * 60)
print("数据加载成功")
print("=" * 60)
print(f"训练集样本数量：{len(X_train)}")
print(f"训练集标签数量：{len(y_train)}")
print(f"测试集样本数量：{len(X_test_unlabeled)}")
print("=" * 60)


# ============================================================
# 2. 划分训练集和验证集
# ============================================================

# 80% 用来训练，20% 用来验证
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"实际训练集数量：{len(X_train_part)}")
print(f"验证集数量：{len(X_val)}")
print("=" * 60)


# ============================================================
# 3. TF-IDF 特征提取
# ============================================================

# 注意：
# TF-IDF 只在真正的训练集上 fit
# 然后再把同样的规则应用到验证集

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train_part)
X_val_tfidf = vectorizer.transform(X_val)

print("TF-IDF 特征提取完成")
print(f"TF-IDF 特征数量：{X_train_tfidf.shape[1]}")
print("=" * 60)


# ============================================================
# 4. SVM 参数实验
# ============================================================

print("\n========== SVM 参数实验 ==========")

svm_results = []

for C in [0.1, 1.0, 10.0]:

    svm_model = SVC(
        kernel='linear',
        C=C,
        random_state=42
    )

    svm_model.fit(X_train_tfidf, y_train_part)

    val_pred = svm_model.predict(X_val_tfidf)

    accuracy = accuracy_score(y_val, val_pred)

    svm_results.append({
        'model': 'SVM',
        'C': C,
        'accuracy': accuracy
    })

    print(f"SVM   C={C:<4}  验证集准确率：{accuracy:.4f}")


# ============================================================
# 5. Logistic Regression 参数实验
# ============================================================

print("\n========== Logistic Regression 参数实验 ==========")

lr_results = []

for C in [0.1, 1.0, 10.0]:

    lr_model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=42
    )

    lr_model.fit(X_train_tfidf, y_train_part)

    val_pred = lr_model.predict(X_val_tfidf)

    accuracy = accuracy_score(y_val, val_pred)

    lr_results.append({
        'model': 'LogisticRegression',
        'C': C,
        'accuracy': accuracy
    })

    print(f"LR    C={C:<4}  验证集准确率：{accuracy:.4f}")


# ============================================================
# 6. 汇总实验结果
# ============================================================

results = svm_results + lr_results

results_df = pd.DataFrame(results)

print("\n========== 所有实验结果 ==========")
print(results_df.to_string(index=False))

# 保存实验结果，后面写实验报告可以直接使用
results_df.to_csv(
    'experiment_results.csv',
    index=False
)


# ============================================================
# 7. 找到验证集表现最好的模型
# ============================================================

best_result = max(
    results,
    key=lambda x: x['accuracy']
)

print("\n========== 最优模型 ==========")
print(f"模型：{best_result['model']}")
print(f"C：{best_result['C']}")
print(f"验证集准确率：{best_result['accuracy']:.4f}")


# ============================================================
# 8. 使用全部有标签训练数据重新训练 TF-IDF
# ============================================================

print("\n========== 开始最终训练 ==========")

final_vectorizer = TfidfVectorizer(
    max_features=5000
)

X_all_train_tfidf = final_vectorizer.fit_transform(X_train)
X_test_tfidf = final_vectorizer.transform(X_test_unlabeled)


# ============================================================
# 9. 根据验证集最优结果训练最终模型
# ============================================================

if best_result['model'] == 'SVM':

    final_model = SVC(
        kernel='linear',
        C=best_result['C'],
        random_state=42
    )

else:

    final_model = LogisticRegression(
        C=best_result['C'],
        max_iter=1000,
        random_state=42
    )


final_model.fit(
    X_all_train_tfidf,
    y_train
)

print("最终模型训练完成")


# ============================================================
# 10. 预测测试集
# ============================================================

predictions = final_model.predict(X_test_tfidf)

print(f"预测完成，共得到 {len(predictions)} 个预测结果")


# ============================================================
# 11. 保存 predictions.csv
# ============================================================

pd.DataFrame(predictions).to_csv(
    'predictions.csv',
    index=False,
    header=False
)

print("\n========== 实验完成 ==========")
print("预测结果已经保存到：predictions.csv")
print("实验参数结果已经保存到：experiment_results.csv")
print("=" * 60)
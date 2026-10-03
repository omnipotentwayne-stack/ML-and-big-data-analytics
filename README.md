# 心臟病預測：機器學習模型比較與外部驗證

以臨床生理指標預測患者是否罹患心臟病，比較 Logistic Regression、Decision Tree、XGBoost 與 Neural Network 四類模型的預測表現與泛化能力，並以獨立資料集進行外部驗證。

> 本專案為國立臺灣科技大學資訊管理研究所「機器學習與大數據分析技術」課程之分組作業，其中 XGBoost、Neural Network 與外部驗證部分由本人獨立完成。

## 資料集

- **訓練資料**：[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)（Fedesoriano, 2021），整合 5 個醫學資料集，共 918 筆、11 個臨床特徵
- **外部驗證**：[UCI Cleveland Heart Disease Dataset](https://doi.org/10.24432/C52P4X)（Janosi et al., 1989），共 303 筆

資料檔未包含在本 repo 中，請自行從上述連結下載。

## 分析流程

1. **探索性資料分析**：檢查缺失值、類別與數值特徵分布、相關性分析
2. **資料預處理**：RestingBP 與 Cholesterol 中不合理的 0 值視為缺失，以中位數填補；類別特徵 One-Hot Encoding；數值特徵 StandardScaler 標準化；80/20 切分訓練與測試集
3. **模型訓練與調參**：
   - Decision Tree：Baseline 與 GridSearchCV 調參比較
   - Logistic Regression：搭配 5-Fold Cross-Validation 檢驗穩定性
   - XGBoost：比較 Grid Search、Random Search、Bayesian Optimization（Optuna）
   - Neural Network：Keras MLP（64→32，Dropout 0.3），以 Optuna 進行兩階段調參
4. **模型解釋**：Feature Importance（Gain）與 SHAP
5. **模型部署與外部驗證**：匯出模型與 Scaler，載入後對 UCI Cleveland 資料進行預測

## 主要結果

**測試集表現**

| 模型 | 測試集準確率 | AUC | 泛化差距 |
|---|---|---|---|
| Decision Tree（Baseline） | 0.7935 | 0.7951 | 0.2065 |
| Decision Tree（GridSearch） | 0.7663 | 0.8390 | 0.1193 |
| Logistic Regression | 0.8587 | 0.9230 | **0.0051** |
| XGBoost（Baseline） | 0.8261 | 0.9114 | 0.1671 |
| XGBoost（Bayesian Opt） | 0.8859 | 0.9232 | 0.0882 |
| Neural Network | **0.8913** | **0.9339** | 0.0133 |

Logistic Regression 另以 5-Fold Cross-Validation 驗證穩定性，平均準確率 0.8556、標準差 0.0234。

**外部驗證（UCI Cleveland，303 筆）**

| 模型 | Accuracy | AUC |
|---|---|---|
| XGBoost（Bayesian Opt） | **0.8746** | **0.9404** |
| Neural Network | 0.8482 | 0.9224 |

## 主要發現

- Decision Tree Baseline 訓練集準確率達 1.0，有明顯 overfitting；以 GridSearchCV 限制樹深與葉節點樣本數後，泛化差距由 0.2065 降至 0.1193，但測試集準確率仍不及 Logistic Regression。
- Neural Network 在原始測試集表現最佳，但在外部驗證中 XGBoost 反而勝出，顯示測試集分數最高的模型，不一定在不同來源的資料上最穩健。
- 超參數調整對 XGBoost 效果明顯（測試集準確率 0.8261 → 0.8859），但對 Neural Network 未能超越 Baseline，推測與資料量有限（918 筆）有關。
- ST_Slope、ExerciseAngina 與 Sex 在 Logistic Regression 係數、XGBoost Feature Importance 與 SHAP 分析中，皆被一致識別為最具影響力的特徵。

## 使用技術

Python、pandas、scikit-learn、XGBoost、TensorFlow / Keras、Optuna、SHAP、matplotlib、seaborn

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `heart_disease_prediction.ipynb` | 完整分析流程，包含 EDA、預處理、模型訓練、調參、解釋與外部驗證 |

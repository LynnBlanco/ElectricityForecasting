# Electricity Forecasting
NCKU DSAI HW1 - Electricity Forecasting

We will implement an algorithm to predict the operating reserve of
electrical power. Given a time series electricity data to predict the value of the operating reserve value of each day during 2022/03/30 ~ 2022/04/13.

## Data Analysis
Dataset           | Requirement
------------------|:---------------------------------------------------------------------------------------------------
[training_data.csv](https://github.com/LynnBlanco/ElectricityForecasting/blob/59b43daf1194408a16e2520b09284c7bddb99c46/training_data.csv) | 2019、2020、2021年1、3月 & 2022年1月，台灣電力公司過去電力供需資訊 & 氣候資訊（每日台北市最高溫度 & 最低溫度）
[train_all.csv](https://github.com/LynnBlanco/ElectricityForecasting/blob/59b43daf1194408a16e2520b09284c7bddb99c46/train_all.csv)         | 2019、2020年1、3月 & 2021年1月~2022年1月，台灣電力公司過去電力供需資訊 & 氣候資訊（每日台北市最高溫度 & 最低溫度）

### ・Heatmap
使用 ```training_data.csv``` 跑出 Heatmap，可以看出 2019、2020、2021、2022 四年的春季用電狀況，藉此找出關聯度較高的特徵。

![image_1](https://github.com/LynnBlanco/ElectricityForecasting/blob/59b43daf1194408a16e2520b09284c7bddb99c46/images/Figure_1.png)

### ・Feature Selection
從 Heatmap 中選出關聯度較高的特徵：```備轉容量率```、```麥寮#2```、```通宵(#1-#6)```、```核三#1```，以這四個特徵作模型訓練。

雖然```林口#3```與```備轉容量```比較```核三#1```與```備轉容量```關聯性度高，但是```核三#1```對其他的特徵關聯度又較```林口#3```還要高，最後實驗結果顯示使用```核三#1```的 RMSE 表現較好，所以最後使用```核三#1```這個特徵。

### ・Feature Relationship
從```備轉容量率```、```麥寮#2```、```通宵(#1-#6)```、```核三#1```這些特徵中刪除2筆偏差值較高的數值。

![image_2](https://github.com/LynnBlanco/ElectricityForecasting/blob/59b43daf1194408a16e2520b09284c7bddb99c46/images/Figure_2.png)
</br></br></br>
![image_3](https://github.com/LynnBlanco/ElectricityForecasting/blob/59b43daf1194408a16e2520b09284c7bddb99c46/images/Figure_3.png)

## Model Training
本次使用的是 ```SVR``` 模型。
```
SVR(kernel='poly', C=1e1, gamma=0.01)
```
資料集使用 ```train_all.csv``` 全年度台電電力供需資料輸入 SVR 模型。

## Evaluation
使用 ```RMSE``` (均方根誤差) 來評估訓練完的模型。
```
np.sqrt(metrics.mean_squared_error(test_y, pred))
```
```test_y```：從 test 資料集中取最近15筆資料 </br>
```pred```：模型預測完之後的結果 </br>

## Conclusion
本次做法使用歷年春季(1、3月)的台電電力供需資料，並且使用這個資料集做 Heatmap，然後使用跑出來關連度較高的特徵作為模型訓練用的特徵。

模型訓練時使用的資料集則不同於 Heatmap 的資料集，使用的是全年度2021年1月~2022年1月的資料。

因為全年度的資料集做 Heatmap 後出來的關聯度都較歷年春季的資料集低。經過交叉測試後發現使用歷年春季的資料集關聯度高的特徵，再搭配全年度的資料集做訓練，模型有較好的表現。

由此結果推測使用同一季節的資料找出影響因素較大的因子，再使用較大量的資料下去訓練，可以得到較好的結果。

## Code Execution
Environment: ```Python 3.7.13``` </br>

- Install ```requirements.txt```
```
$ pip install -r requirements.txt
```
- Execute ```app.py```
```
$ python app.py --training training_data.csv --output submission.csv
```

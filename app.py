import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn import metrics
import seaborn as sns  # 繪圖

# 選擇春季的月份，包含2019、2020、2021年的1、3月 + 2022年1月的dataset做heatmap，找出關聯度最高的feature
# 使用2019、2020年的1、3月 + 2021年全年的dataset訓練SVR回歸模型，預測2022/03/30~04/13的備轉容量

def heatmap(train):
    # heatmap
    corr = train.corr()
    # The number of columns to be displayed in the heat map
    k = 7
    # Calculate for the top 5 columns with the highest correlation with operating reserve
    cols = corr.nlargest(k, 'operating reserve')['operating reserve'].index
    cm = np.corrcoef(train[cols].values.T)
    # Font size of the heatmap
    sns.set(font_scale=1.2)
    # View in a heat map
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols.values, xticklabels=cols.values, linewidths=1, linecolor='white')

    # 觀察資料分布狀況，發現有些特徵存在較大的偏差值
    sns.set()
    cols = ['operating reserve', 'rate', 'ele_pro', 'MaiLiao#2', 'TongXiao (#1-#6)', 'NuclearThree#1']
    sns.set(font_scale=1)
    sns.pairplot(train[cols], size=1.5)

    # 將顯示前一筆最大的數據，並將其刪除，試圖減少偏差值
    train = train.drop(index=train.sort_values(by='rate', ascending=False)['date'][:2].index)
    train = train.drop(index=train.sort_values(by='MaiLiao#2', ascending=False)['date'][:2].index)
    # train=train.drop(index=train.sort_values(by='ele_pro',ascending=False)['date'][:2].index)
    # train=train.drop(index=train.sort_values(by='people_use',ascending=False)['date'][:2].index)
    train = train.drop(index=train.sort_values(by='LinKou#3', ascending=False)['date'][:2].index)
    train = train.drop(index=train.sort_values(by='TongXiao (#1-#6)', ascending=False)['date'][:1].index)

    sns.set()
    cols = ['operating reserve', 'rate', 'ele_pro', 'MaiLiao#2', 'TongXiao (#1-#6)', 'NuclearThree#1']
    sns.set(font_scale=1)
    sns.pairplot(train[cols], size=1.5)
    plt.show()
    return train


def sub(pred):
    name = ['operating reserv(MW)']
    pred = pd.DataFrame(pred, columns=name)
    date = [['date'], ['20220330'], ['20220331'], ['20220401'],
            ['20220402'], ['20220403'], ['20220404'], ['20220405'],
            ['20220406'], ['20220407'], ['20220408'], ['20220409'],
            ['20220410'], ['20220411'], ['20220412'], ['20220413']]
    name = date.pop(0)
    date_df = pd.DataFrame(date, columns=name)
    res = pd.concat([date_df, pred], axis=1)
    res.to_csv(args.output, index=0)


def forecasting():
    # import data
    train = pd.read_csv(args.training)
    train_feature = heatmap(train)

    train_new = pd.read_csv('train_all.csv')
    # 建立training dataset
    X = train_new[['rate', 'ele_pro', 'MaiLiao#2', 'TongXiao (#1-#6)', 'NuclearThree#1']]
    Y = train_new[['operating reserve']]
    Y = Y.values.reshape(-1, 1)

    # Feature scaling
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    train_x = scaler_x.fit_transform(X)
    train_y = scaler_y.fit_transform(Y)
    # SVR
    regressor = SVR(kernel='poly', C=1e1, gamma=0.01)
    regressor.fit(train_x, train_y)

    test = pd.read_csv('test.csv')
    test_y = test[['operating reserve']][38:]
    pred_x = test[['rate', 'ele_pro', 'MaiLiao#2', 'TongXiao (#1-#6)', 'NuclearThree#1']][14:29]
    # print(pred_x)
    pred_x = scaler_x.fit_transform(pred_x)
    pred = regressor.predict(pred_x)
    pred = scaler_y.inverse_transform(pred)
    print("RMSE:", np.sqrt(metrics.mean_squared_error(test_y, pred)))
    sub(pred)


# main
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')
    parser.add_argument('--output', default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    forecasting()
from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

"""""""""""""""""""""""""""""""""
Correlated Features structure.
"""""""""""""""""""""""""""""""""
@dataclass
class CorrelatedFeatures:
    feature1:str
    feature2:str
    correlation:float
    lin_reg:LinearRegression
    threshold:float

"""""""""""""""""""""""""""""""""
Anomaly Report structure.
"""""""""""""""""""""""""""""""""
@dataclass
class AnomalyReport:
    description:str
    time_step:int

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
A class for an anomaly detector using linear regression.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class SimpleAnomalyDetector:
    def __init__(self, threshold:float=0.9) -> None:
        self.threshold = threshold
        self.cf = []

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Getting a data frame and setting the normal model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def learn_normal(self, df:pd.DataFrame) -> None:
        pearson_frame = df.corr(method='pearson')
        pearson_frame = pearson_frame.abs()
        high_pearsons = np.argwhere(pearson_frame.to_numpy()>=self.threshold).tolist()
        correlated_indexes = []
        for index in high_pearsons:
            if index[0] != index[1] and [index[1], index[0]] not in correlated_indexes:
                correlated_indexes.append(index)
        for colindex in correlated_indexes:
            X = df.iloc[:, colindex[0]].values.reshape(-1,1)
            Y = df.iloc[:, colindex[1]].values.reshape(-1,1)
            linear_reg = LinearRegression()
            linear_reg = linear_reg.fit(X, Y)
            max = 0
            y_pred = linear_reg.predict(X)
            for i in range(0, len(X)):
                if abs(Y[i] - y_pred[i]) > max:
                    max = abs(Y[i] - y_pred[i])
            max = max * 1.1
            self.cf.append(CorrelatedFeatures(colindex[0], colindex[1], pearson_frame.iloc[colindex[0], colindex[1]], linear_reg, max))

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Checking if there's an anomaly from the normal model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def detect(self, df:pd.DataFrame) -> List:
        anomaly_report = []
        for cor_feature in self.cf:
            x_vec = df.iloc[:, cor_feature.feature1].values.reshape(-1,1)
            y_vec = df.iloc[:, cor_feature.feature2].values.reshape(-1,1)
            Y_vec_pred = cor_feature.lin_reg.predict(x_vec)
            cf_col_name_1 = df.columns[cor_feature.feature1]
            cf_col_name_2 = df.columns[cor_feature.feature2]
            for rowindex in range (0, df.last_valid_index()):
                point_y = y_vec[rowindex][0]
                y_pred = Y_vec_pred[rowindex][0]
                if abs(y_pred - point_y) > cor_feature.threshold[0]:
                    desc = cf_col_name_1 + " - " +  cf_col_name_2
                    anomaly_report.append(AnomalyReport(desc, rowindex + 1))

        return anomaly_report

"""""""""""""""""""""""""""""""""""""""""""""
Printing the report in a nice format
"""""""""""""""""""""""""""""""""""""""""""""
def printReport(report:List) ->None:
    print('-' * 20, "Start of report", '-' * 20)
    for anomaly_report in report:
        print(anomaly_report)
    print('-' * 20, "End of report", '-' * 20)

def main():
    df = pd.read_csv("new_reg_flight.csv")
    anomaly_detector = SimpleAnomalyDetector(0.9)
    anomaly_detector.learn_normal(df)
    report = anomaly_detector.detect(pd.read_csv("anomaly_flight.csv"))
    printReport(report)

if __name__ == "__main__":
    main()
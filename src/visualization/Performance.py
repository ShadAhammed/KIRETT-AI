import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools
from sklearn import metrics
from sklearn.metrics import classification_report


class ModelPerformance:

    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    metrics_df = pd.DataFrame(columns=['Classification', 'Accuracy', 'Sensitivity', 'Specificity', 'f_1 Score'])

    # Creating a function to report confusion metrics
    def ConfMetrics(self, conf_matrix, classification):
        global metrics_df
        # save confusion matrix and slice into four pieces
        TP = conf_matrix[1][1]
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]
        print('True Positives:', TP)
        print('True Negatives:', TN)
        print('False Positives:', FP)
        print('False Negatives:', FN)

        # calculating accuracy
        accuracy = round((float(TP + TN) / float(TP + TN + FP + FN)), 4) * 100

        # calculating sensitivity
        sensitivity = round((TP / float(TP + FN)), 4) * 100
        # calculating specificity
        specificity = round((TN / float(TN + FP)), 4) * 100
        # calculating precision
        precision = round((TP / float(TP + FP)), 4) * 100
        # calculating f_1 score
        f1 = round(2 * ((precision * sensitivity) / (precision + sensitivity)), 4)

        print('-' * 50)
        print(f'Accuracy: {accuracy}')
        print(f'Sensitivity: {sensitivity}')
        print(f'Specificity: {specificity}')
        print(f'Precision: {precision}')
        print(f'f_1 Score: {f1}')
        metrics_df = self.metrics_df.append({'Classification': classification, 'Accuracy': accuracy,
                                             'Sensitivity': sensitivity, 'Specificity': specificity,
                                             'Precision': precision, 'f_1 Score': f1}, ignore_index=True)
        return metrics_df

    def draw_confusion_matrix(self, matrix):
        plt.clf()

        # place labels at the top
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')

        # plot the matrix per se
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)

        # plot colorbar to the right
        plt.colorbar()

        fmt = 'd'
        class_names = self.y_test.columns
        # write the number of predictions in each bucket
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            # if background is dark, use a white number, and vice-versa
            plt.text(j, i, format(matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.tight_layout()
        plt.ylabel('True label', size=14)
        plt.xlabel('Predicted label', size=14)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    @staticmethod
    def ClfReport(y_test, prediction, model):
        report = classification_report(y_test, prediction, output_dict=True, digits=1)
        report = pd.DataFrame(report)
        report = report.T
        acc = report.iloc[0, 2]
        report = report.iloc[0:2, 0:3]
        clf = [model, ' ']
        report['Clf'] = clf
        acc = [round(acc, 3), '']
        report['Accuracy'] = acc
        return report

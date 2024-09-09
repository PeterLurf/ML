import numpy as np
import torch as torch
import torch.nn as nn


def AccuracyClassification(y_pred,y_true) -> float:
    #accuracy is the ratio of the number of correct predictions to the total number of predictions
    return torch.sum(y_pred == y_true) / len(y_pred)

def PrecisionClassification(y_pred,y_true) -> float:
    #precision is the ratio of the number of true positive predictions to the number of true positive predictions and false positive predictions
    return torch.sum(y_pred * y_true) / torch.sum(y_pred)

def RecallClassification(y_pred,y_true) -> float:
    #recall is the ratio of the number of true positive predictions to the number of true positive predictions and false negative predictions
    return torch.sum(y_pred * y_true) / torch.sum(y_true)
    

def F1_scoreClassification(y_pred,y_true) -> float:
    #F1 score is the harmonic mean of precision and recall
    precision = PrecisionClassification(y_pred,y_true)
    recall = RecallClassification(y_pred,y_true)
    return 2 * precision * recall / (precision + recall)
    
def ROC_AUC_Classification(y_pred,y_true) -> float:
    #ROC AUC is the area under the receiver operating characteristic curve
    #it is a measure of the performance of a classification model at various thresholds
    #it is equal to the probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true,y_pred)

def ROC_graph(y_pred,y_true) -> float:
    #ROC graph is a graphical representation of the performance of a classification model at various thresholds
    #it is a plot of the true positive rate against the false positive rate
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true,y_pred)
    from matplotlib import pyplot as plt
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')  
    plt.show()
    plt.legend(['ROC Curve'])
    return 

def Precision_Recall_graph(y_pred,y_true) -> float:
    #Precision-Recall graph is a graphical representation of the performance of a classification model at various thresholds
    #it is a plot of the precision against the recall
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true,y_pred)
    from matplotlib import pyplot as plt
    plt.plot(recall,precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')  
    plt.show()
    plt.legend(['Precision-Recall Curve'])
    return

if __name__ == "__main__":
    #test the functions
    y_pred = torch.tensor([1,0,1,0,1])
    y_true = torch.tensor([1,1,1,0,1])
    print(AccuracyClassification(y_pred,y_true))
    print(PrecisionClassification(y_pred,y_true))
    print(RecallClassification(y_pred,y_true))
    print(F1_scoreClassification(y_pred,y_true))
    print(ROC_AUC_Classification(y_pred,y_true))
    ROC_graph(y_pred,y_true)
    Precision_Recall_graph(y_pred,y_true)
  

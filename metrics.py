from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def sklearn_Compatible_preds_and_targets(model_prediction_list, model_target_list):

    y_pred_list = []
    preds = []
    target_list = []
    tgts = []
    y_pred_list = [a.tolist() for a in model_prediction_list]
    preds = [item for sublist in y_pred_list for item in sublist]
    target_list = [a.tolist() for a in model_target_list]
    tgts = [item for sublist in target_list for item in sublist]
    return accuracy_score(preds, tgts)

def accuracy_score(prediction, target):

    TN, FP, FN, TP = confusion_matrix(target, prediction).ravel()
    print("TP: ", TP, "FP: ", FP, "TN: ", TN, "FN: ", FN)
    #TSS Computation also known as "recall"
    tp_rate = TP / float(TP + FN) if TP > 0 else 0  
    fp_rate = FP / float(FP + TN) if FP > 0 else 0
    TSS = tp_rate - fp_rate
    
    #HSS2 Computation
    N = TN + FP
    P = TP + FN
    HSS = (2 * (TP * TN - FN * FP)) / float((P * (FN + TN) + (TP + FP) * N))
    
    #F0.5 Score Computation
    prec,recall,fscore,_ = precision_recall_fscore_support(target, prediction, average='macro', beta=0.5)

    return TSS, HSS, fscore, TN, FP, FN, TP

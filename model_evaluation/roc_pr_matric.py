from sklearn.metrics import precision_recall_curve, roc_curve

def all(y_valid, yprob, focus_label=1):
    
    roc = roc_curve(y_valid, yprob, pos_label=focus_label)
    pr = precision_recall_curve(y_valid, yprob, pos_label=focus_label)
    value = {'roc: fpr-tpr-thresholds': roc, 'pr: precision-recall': pr}
	
    return value
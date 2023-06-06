import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import itertools 


def numeric_metrics(model,dataset,threshold=0.5,drop_carbon=True):
    for data, labels in dataset.take(1):
            y_pred = model.predict(data,verbose=0)
            y_true = labels
    for data, labels in dataset.skip(1).take(-1):
            y_pred = tf.concat([y_pred,model.predict(data,verbose=0)],axis = 0)
            y_true = tf.concat([y_true,labels],axis=0)
    
    predicted_labels = tf.cast(tf.math.greater_equal(y_pred,threshold),tf.float32)
    labels = list(range(80))
    if drop_carbon:
        labels.remove(2)
    precision = sklearn.metrics.precision_score(y_true,predicted_labels,average='weighted',labels=labels)
    recall = sklearn.metrics.recall_score(y_true,predicted_labels,average='weighted',labels=labels)
    F1score = sklearn.metrics.f1_score(y_true,predicted_labels,average='weighted',labels=labels)
    print(f'F1_score is {F1score} ,  Recall is {recall}, precision is {precision}')
    return F1score,precision,recall


class match_rate():
    def __init__(self, model, threshold = 0.5, drop_carbon=True):
        self.model = model
        self.threshold = threshold
        self.drop_carbon = drop_carbon


    def calculate(self, dataset):
        for data, labels in dataset.take(1):
            y_pred = self.model.predict(data,verbose=0)
            y_true = labels
        for data, labels in dataset.skip(1).take(-1):
            y_pred = tf.concat([y_pred,self.model.predict(data,verbose=0)],axis = 0)
            y_true = tf.concat([y_true,labels],axis=0)
            
        y_true = tf.cast(y_true,tf.float32)
        if self.drop_carbon:
            rem_col = 2
            y_pred = tf.cast(tf.math.greater_equal(y_pred,self.threshold),tf.float32)
            y_true = tf.concat([y_true[:,0:rem_col] , y_true[:,rem_col+1::]],1)
            y_pred = tf.concat([y_pred[:,0:rem_col] , y_pred[:,rem_col+1::]],1)
        else:
            y_pred = tf.cast(tf.math.greater_equal(y_pred,self.threshold),tf.float32)

        result = tf.math.equal(y_true, y_pred) # this compares elementwise
        result = tf.reduce_all(result, axis=1) # returns True for a row if all elements in that row are True
        result = tf.cast(result, 'float32') # change it to numeric for calculating accuracy
        result = tf.reduce_mean(result) # calculate the accuracy
        self.result = result
        return result
    
class confusionmatrix():

    def __init__(self, class_names,model,dataset):
        '''
        note that this only takes a single batch
        '''
        for data, labels in dataset.take(1):
            y_pred = tf.math.argmax(model.predict(data,verbose=0),axis=1)
            y_true = tf.math.argmax(labels,axis=1)
        
        self.cm = sklearn.metrics.confusion_matrix(y_true,y_pred)
        self.norm_cm = self.cm
        self.class_names = class_names

        
    def plot_confusion_matrix(self):
        plt.figure(figsize=(40, 40))
        plt.imshow(self.norm_cm, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, fontsize = 20)
        plt.yticks(tick_marks, self.class_names, fontsize = 20)
        
        
        for i, j in itertools.product(range(self.norm_cm.shape[0]), range(self.norm_cm.shape[1])):
            if i == j:
                plt.text(j, i, self.norm_cm[i, j], horizontalalignment="center", color='white')
            else:
                plt.text(j, i, self.norm_cm[i, j], horizontalalignment="center", color='black')

      
        plt.ylabel('True label',fontweight = 'bold', fontsize = 35)
        plt.xlabel('Predicted label',fontweight = 'bold', fontsize = 35)
        plt.tick_params(labeltop=True, labelright=True)
        plt.tight_layout()


    


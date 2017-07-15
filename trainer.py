import pickle
import numpy as np
from flask import Flask, render_template, abort, make_response, current_app
app = Flask(__name__, static_path='/static')
import os


filename = 'static/se_model_lr0.01epoch_20hlu_10.sav'

class EfficientLowNoiseSentimentalNeuralNetwork(object):
    def __init__(self, training_data, num_hidden_nodes=10, num_epochs=10, learning_rate=0.1, min_count=10,polar_cutoff=0.1):
        # set our random number generator 
        np.random.seed(1)
        # set our improvement parameters
        self.min_count = min_count
        # ToDo -cater different centers of the frquency ditribution
        self.polar_cutoff = polar_cutoff
        # pre-process data
        self.pre_process_data(training_data)
        
        # set network paramaters
        self.num_features = len(self.vocab)
        self.vocab_vector = np.zeros((1, len(self.vocab)))
        self.num_input_nodes = self.num_features
        self.num_hidden_nodes = num_hidden_nodes
        self.num_epochs = num_epochs
        self.num_output_nodes = 1
        self.learning_rate = learning_rate
        
        # for printing later
        self.losses = {'train':[], 'validation':[]}
        self.accuracy = {'train':[], 'validation':[]}
        self.confusion_matrix = np.zeros((1,4))
        # initialize weights
        self.weights_i_h = np.random.randn(self.num_input_nodes, self.num_hidden_nodes)
        self.weights_h_o = np.random.randn(self.num_hidden_nodes, self.num_output_nodes)
         # initialize weights
        self.bias_i_h = np.zeros(self.num_hidden_nodes)
        self.bias_h_o = np.zeros(self.num_output_nodes)
        # initialise the hidden layer with zeros
        self.hidden_layer = np.zeros((self.num_output_nodes, self.num_hidden_nodes))
        
    def forward_backward_propagate(self, text, label):
        ### Forward pass ###
        # Input Layer & Hidden layer operation
        self.hidden_layer *= 0
        for index in text:
            self.hidden_layer += self.weights_i_h[index]
        self.hidden_layer += self.bias_i_h
            
        # Output layer
        output_layer = self.sigmoid(self.hidden_layer.dot(self.weights_h_o) + self.bias_h_o)
        
        ### Backward pass ###
        # Output error
        output_layer_error = output_layer - self.get_target_for_label(label)
        output_layer_delta = output_layer_error * self.sigmoid_derivative(output_layer)

        # Backpropagated error - to the hidden layer
        hidden_layer_error = output_layer_delta.dot(self.weights_h_o.T)
        # hidden layer gradients - no nonlinearity so it's the same as the error
        hidden_layer_delta = output_layer_error 

        # update the weights and bias - with grdient descent
        self.weights_h_o -= self.hidden_layer.T.dot(output_layer_delta) * self.learning_rate 
        self.bias_h_o -= output_layer_delta[0] * self.learning_rate 
        # update only the weights and bias used in the forward pass
        for index in text:
            self.weights_i_h[index] -= hidden_layer_delta[0] * self.learning_rate
        self.bias_i_h -= hidden_layer_delta[0] * self.learning_rate
        
        if(output_layer >= 0.5 and self.get_target_for_label(label) == 1):
                self.correct_so_far += 1
        elif(output_layer < 0.5 and self.get_target_for_label(label) == 0):
                self.correct_so_far += 1
        
    def train(self):
        # process data to eliminate zero's
        training_data_text = list()
        for review in training_data.Text:
            indices = set()
            for word in review.split(" "):
                if(word in self.word_to_column.keys()):
                    indices.add(self.word_to_column[word])
            training_data_text.append(list(indices))
        
        # iterate through all epochs
        for epoch in range(self.num_epochs):
            self.correct_so_far = 0
            start = time.time()
            training_loss = 0
            validation_loss = 0
            training_accuracy = 0

            # train over all rows of training data
            for row in range(training_data.shape[0]):
                # Forward and Back Propagation
                self.forward_backward_propagate(training_data_text[row], training_data.Label[row])
                # calculate our speed
                elasped_time = float(time.time() - start + 0.001)
                samples_per_second = row / float(elasped_time)
                # calculate our accuracy
                training_accuracy = self.correct_so_far * 100 / float(row+1)
                # print progress of training
                sys.stdout.write("\rEpoch: "+ str(epoch)
                                 + " Progress: " + str(100 * row/float(training_data.shape[0]))[:4] + "%"
                                 + " Speed(samples/sec): " + str(samples_per_second)[0:5] 
                                 + " #Correct: " + str(self.correct_so_far) 
                                 + " #Trained: " + str(row+1) 
                                 + " Training Accuracy: " + str(training_accuracy)[:4] + "%")
            
            self.accuracy["train"].append(training_accuracy) 
                
            training_loss = self.run(training_data[0:7000])
            validation_loss = self.run(validation_data,  mode="validate")
            self.losses["train"].append(training_loss)
            self.losses["validation"].append(validation_loss)      
            print("")
            
    def run(self, input_data, mode="train"):
        # total losses for sample
        val_correct_so_far = 0
        val_accuracy = 0
        loss = 0
        # iterate through all training samples
        for row in range(0, input_data.shape[0]):
            # get prediction
            pred = self.predict(input_data.Text[row])
            # calculate the loss
            loss += np.mean((pred - self.get_target_for_label(input_data.Label[row]))**2)
            # Calculate our accuracy
            if(mode is "validate"):
                #calculate the accuracy
                if(pred >= 0.5 and self.get_target_for_label(input_data.Label[row]) == 1):
                    val_correct_so_far += 1
                elif(pred < 0.5 and self.get_target_for_label(input_data.Label[row]) == 0):
                    val_correct_so_far += 1
        if(mode is "validate"):
            val_accuracy = val_correct_so_far * 100 / float(input_data.shape[0])
            self.accuracy["validation"].append(val_accuracy) 
        return loss/float(input_data.shape[0]) 
            
            
    def test(self, test_data):
        # How many predictions are correct out of total training
        correct = 0
        # Reset cnfusion matrix 
        self.confusion_matrix = np.zeros((1,4))
        # start time of one epoch
        start = time.time()
        # iterate through all training samples
        for i in range(0, test_data.shape[0]):
            # get prediction
            pred = self.predict(test_data.Text[i])
            # count how many we validate as correct
            if(pred >= 0.5 and self.get_target_for_label(test_data.Label[i]) == 1):
                correct += 1
            elif(pred < 0.5 and self.get_target_for_label(test_data.Label[i]) == 0):
                correct += 1
            # create confusion matrix    
            self.confusion_matrix += self.calculate_confusion_matrix(np.rint(pred), self.get_target_for_label(test_data.Label[i]))
            # calculate our sampling rate
            reviews_per_second = i / float(time.time() - start + 0.001)
            # print out the validation metrics
            sys.stdout.write("\rProgress:" + str(100 * i/float(test_data.shape[0]))[:4] + "%"
                             + " Speed(reviews/sec):" + str(reviews_per_second)[0:5] 
                             + " #Correct:" + str(correct) 
                             + " #Tested:" + str(i+1) 
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
            
    def predict(self, text):
        # prepare the data
        indices = set()
        for word in text.split(" "):
            if(word in self.word_to_column.keys()):
                indices.add(self.word_to_column[word])
        
        ### Forward pass ###
        # Input Layer & Hidden layer operation
        self.hidden_layer *= 0
        for index in indices:
            self.hidden_layer += self.weights_i_h[index]
        self.hidden_layer += self.bias_i_h

        # output layer
        output_layer = self.sigmoid(self.hidden_layer.dot(self.weights_h_o) + self.bias_h_o)
        
        return output_layer.flatten()
    
    
    def visualise_training(self):
        plt.figure(1)
        plt.title('Training, LR: ' + str(self.learning_rate) + ' HLU: ' + str(self.num_hidden_nodes))
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.losses['train'], label='Training loss')
        plt.plot(self.losses['validation'], label='Validation loss')
        plt.legend()
        
        file_path = "training/plot_loss_lr" + str(self.learning_rate) + "epoch_" + str(self.num_epochs)+ "hlu_" + str(self.num_hidden_nodes) + ".png"
        self.save_plot(file_path)
        
        plt.figure(2)
        plt.title('Training, LR: ' + str(self.learning_rate) + ' HLU: ' + str(self.num_hidden_nodes))
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(self.accuracy['train'], label='Training Accuracy')
        plt.plot(self.accuracy['validation'], label='Validation Accuracy')
        plt.legend()
        
        file_path = "training/plot_acc_lr" + str(self.learning_rate) + "epoch_" + str(self.num_epochs) + "hlu_" + str(self.num_hidden_nodes) + ".png"
        self.save_plot(file_path)
        
    def save_plot(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(file_path, bbox_inches='tight')
                      
    def get_confusion_matrix(self):
        hyper_param_dict = {'EP':self.num_epochs, 'LR':self.learning_rate, 'HLU': self.num_hidden_nodes}
        df = pd.DataFrame(data=[hyper_param_dict], columns=['EP', 'LR', 'HLU'])
        
        tmp_df = pd.DataFrame(data=self.confusion_matrix, columns=['TP','FP', 'TN', 'FN'])
        # combine the rows, not columns i.e axis=1
        df = pd.concat([df, tmp_df], axis=1)
    
        TP = self.confusion_matrix[0][0]
        FP = self.confusion_matrix[0][1]
        TN = self.confusion_matrix[0][2]
        FN = self.confusion_matrix[0][3]

        recall = TP/(TP + FN)
        precision = TP/(TP + FP)
        f_one_score = (2*recall*precision)/(recall + precision)
        mcc_score = ((TP * TN) - (FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        test_accuracy = (TP + TN)/(TP +TN +FP +FN)
        
        df['RECALL'] = recall *100
        df['PRECISION'] = precision *100
        df['F1-S'] = f_one_score *100
        df['MCC-S'] = f_one_score *100
        df['TE-ACC'] = test_accuracy *100
        df['TR-ACC'] = None if not self.accuracy['train'] else self.accuracy['train'][self.num_epochs -1]
        df['VA-ACC'] = None if not self.accuracy['validation'] else self.accuracy['validation'][self.num_epochs -1]
        
        return df
    
    def pre_process_data(self, training_data):
        # frequency of words in positive reviews
        positive_counts = Counter()
        # frequency of words in negative reviews
        negative_counts = Counter()
        # frequency of words in all reviews
        total_counts = Counter()
        # affinity of words for being in positive/negative reviews
        positive_negative_ratios = Counter()
        
        # get the counts
        for i in range(training_data.shape[0]):
            if(training_data.Label[i] == 'POSITIVE'):
                for word in training_data.Text[i].split(" "):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            if(training_data.Label[i] == 'NEGATIVE'):
                for word in training_data.Text[i].split(" "):
                    negative_counts[word] += 1
                    total_counts[word] += 1
        # calculate positive-negative affinity
        for term, count in list(total_counts.most_common()):
            # consider only words that appear more than 50 times
            if(count >= 50):
                positive_negative_ratio = float(positive_counts[term]) / float(negative_counts[term]+1)
                positive_negative_ratios[term] = positive_negative_ratio

        for word, ratio in positive_negative_ratios.most_common():
            # normalise the ratio
            if(ratio > 1):
                positive_negative_ratios[word] = np.log(ratio)
            else:
                positive_negative_ratios[word] = -np.log((1 / (ratio + 0.01)))
        
        self.vocab = set()
        
        for review in training_data.Text:
            for word in review.split(" "):
                # eliminate low freqeuncy words
                if(total_counts[word] > self.min_count):
                    if(word in positive_negative_ratios.keys()):
                        # eliminate words with very high frequency on both sides of the spectrum
                        if((positive_negative_ratios[word] >= self.polar_cutoff) or (positive_negative_ratios[word] <= -self.polar_cutoff)):
                            self.vocab.add(word)
                    else:    
                        self.vocab.add(word)
        # convert to list so that we can access using indices        
        self.vocab = list(self.vocab)
        # create our vocab to column index mapping
        self.word_to_column = {}
        for i, word in enumerate(self.vocab):
            self.word_to_column[word] = i
    
    def calculate_confusion_matrix(self, y_predicted, y_actual):
        #True/False Positive and True/False Negative 
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        if y_actual==y_predicted==1:
               TP += 1
        if y_predicted==1 and y_actual!=y_predicted:
               FP += 1
        if y_actual==y_predicted==0:
               TN += 1
        if y_predicted==0 and y_actual!=y_predicted:
               FN += 1
        return np.array((TP, FP, TN, FN))

    def get_target_for_label(self, label):
        if(label == 'POSITIVE'):
            return 1
        elif(label == 'NEGATIVE'):
            return 0
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    
    def sigmoid_derivative(self,x):
        return x * (1 - x)

def hello(analyze):

    load_ = pickle.load(open(filename, 'rb'))

    result_ = load_.predict(analyze)

    perc = result_ * 100
    if perc >= 50:
        answer = "POSITIVE %s" %(perc)
    else:
        answer = "NEGATIVE %s" %(perc)

    print(answer)
    return answer

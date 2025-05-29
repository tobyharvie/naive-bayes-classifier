import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from collections import defaultdict
stopwords = stopwords.words('english')

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
classes = ['W', 'A', 'S', 'G']

class standard_naive_bayes:
    # standard naive bayes implementation
    def __init__(self, train_df=pd.read_csv('train.csv'), test_df=pd.read_csv('test.csv'), classes = ['W', 'A', 'S', 'G']):
        self.train_df = train_df
        self.test_df = test_df
        self.classes = classes
        self.create_vocabulary()
        self.create_freq_table()

    def get_tokens(self, desc):
        # gets tokens from a training instance
        return desc.split()

    def create_vocabulary(self):
        # creates a set of tokens from all instances of training data
        self.vocab = set()
        for index, row in self.train_df.iterrows():
            tokens = self.get_tokens(row['Description'])
            self.vocab.update(tokens)

    def create_freq_table(self):
        # creates a table of token freqencies for each class in the training data

        self.freq_table = {label : {} for label in self.classes}

        for index, row in self.train_df.iterrows():
            tokens = self.get_tokens(row['Description'])
            for token in tokens:
                # add or increment token count to frequency table
                self.freq_table[row['Class']][token] = self.freq_table[row['Class']].get(token,0) + 1
    
    def get_posterior(self, label, freqs, tokens):
        # gets the posterior probability P(label)P(w|label)

        tokens = Counter(tokens)
        # P(label). Prior probability
        p = np.log(len(self.train_df[self.train_df['Class']==label])/len(self.train_df))

        denom = sum(freqs.values())

        for token, count in tokens.items():
            # P(w|label). Likelihood
            if token in freqs.keys():
                p += count * np.log(freqs[token]/denom)
            else:
                p += count * np.log(1/denom)

        return p

    def classify(self, desc):
        # classifies a new instance
        probs = {}

        for label, freqs in self.freq_table.items():
            # iterate for each class label
            tokens = self.get_tokens(desc)
            probs[label] = self.get_posterior(label, freqs, tokens)
        
        # get the class corresponding to the maximum probability
        return max(probs, key=probs.get)        

    def predict(self, record_data=False):
        # computes classifications for each instance in the test set and returns the predictions

        preds = []
        for index, row in self.test_df.iterrows():
            preds.append(self.classify(row['Description']))

        if record_data:
            # write to csv
            preds_df = pd.DataFrame({'Class':preds})
            preds_df.index = np.arange(1, len(preds_df) + 1)
            preds_df.to_csv('v13.csv', index_label='Id')

        return preds


class extended_naive_bayes(standard_naive_bayes):
    # extended naive bayes model
    # extensions include:
    # including bigrams and removing stopwords
    # attribute selection using chi-squared correlation analysis
    # fine tuned Laplace smoothing
    def __init__(self,train_df=pd.read_csv('train.csv'), test_df=pd.read_csv('test.csv'), record_data=False):
        super().__init__(train_df, test_df)
        self.cutoff=100000
        self.attribute_selection()
        self.record_data = record_data
        self.alpha = 0.5

    def get_tokens(self, desc):
        # changes: implementing bigrams and removing stopwords
        words = desc.split()
        words = [word for word in words if word not in stopwords]
        bigrams= [words[i-1]+'_'+words[i] for i in range(1,len(words))]
        return words + bigrams
    
    def attribute_selection(self):
        # attribute selection using chi-squared
        # implemented using information from lectures, https://www.geeksforgeeks.org/ml-chi-square-test-for-feature-selection/ and wikipedia

        # total number of token positions in each class
        class_totals = {label: sum(freqs.values()) for label, freqs in self.freq_table.items()}
        total_instances = sum(class_totals.values())

        # total token frequency across classes
        attr_totals = defaultdict(int)
        for label, freqs in self.freq_table.items():
            for token, freq in freqs.items():
                attr_totals[token] += freq

        chi_squared_scores = {}
        for attr in attr_totals:
            # for each attribute compute Chi-squared
            chi_squared = 0
            for label in self.freq_table:
                # actual number of attribute instances in class
                observed = self.freq_table[label].get(attr, 0)
                # expected number of attribute instances based on ealier
                expected = (class_totals[label] * attr_totals[attr]) / total_instances
                if expected > 0:
                    # basic chi squared formula
                    chi_squared += ((observed - expected) ** 2) / expected
            chi_squared_scores[attr] = chi_squared

        # sort attributes by score
        sorted_attrs = sorted(chi_squared_scores.items(), key=lambda x: x[1], reverse=True)
        # select top correlating attributes
        sorted_attrs = sorted_attrs[:self.cutoff]
        # get just a list of attribute names
        sorted_attrs = [x[0] for x in sorted_attrs]

        # remove uncorrelated features from frequency table
        for label in self.freq_table.keys():
            self.freq_table[label] = {attr: self.freq_table[label][attr] for attr in sorted_attrs if attr in self.freq_table[label].keys()}
    
    def get_posterior(self, label, freqs, tokens):
        # changes: Laplace smoothing. scale with parameter alpha

        tokens = Counter(tokens)
        # P(label). Prior probability
        p = np.log(len(self.train_df[self.train_df['Class']==label])/len(self.train_df))

        # denominator used in likelihood calculation
        denom = sum(freqs.values()) + self.alpha * len(self.vocab)

        # add log likelihood of each token to total probability
        for token, count in tokens.items():
            # P(w|label). Likelihood
            if token in freqs.keys():
                p += count * np.log((freqs[token] + self.alpha) / denom)
            else:
                p += count * np.log(1/denom)

        return p         
    
def get_preds(model, cross_val_train, cross_val_test):
    # standard method of getting predictions for the cross_validate method. Can adjust depending on experiment
    return model(train_df=cross_val_train, test_df=cross_val_test).predict()

def tune_alpha():
    # find the best alpha value for Laplace smoothing using the extended model
    for a in [0.25*i for i in range(1,6)]:
        # iterate through alpha values

        # modified prediction function using a specified alpha value
        def get_preds_adjusted_alpha(model, cross_val_train, cross_val_test):
            nb = model(train_df = cross_val_train, test_df = cross_val_test, record_data=False)
            nb.alpha = a
            return nb.predict()

        print(f"Beginning validation for alpha={a}")
        cross_validate(extended_naive_bayes, get_preds_adjusted_alpha, k=5, verbalize=False)

def cross_validate(model, get_preds=get_preds, k=10, verbalize=True):
    # we use kfold validation as discussed in the report

    ins = len(train_df)
    accuracies = []

    for i in range(k):
        # split data
        cross_val_train = pd.concat((train_df.iloc[:int(i*ins//k)], train_df.iloc[int((i+1)*ins//k):]))
        cross_val_test = train_df.iloc[int(i*ins//k):int((i+1)*ins//k)]

        # get predictions from model based on train/test split
        preds = get_preds(model, cross_val_train, cross_val_test)

        # compute accuracy
        matches = sum(1 for i in range(len(cross_val_test)) if preds[i] == cross_val_test.iloc[i]['Class'])
        accuracy = (matches / len(cross_val_test)) * 100 
        accuracies.append( accuracy )
        if verbalize: print(f"Fold {i}: accuracy {accuracy}")

    print(f"Mean accuracy: {np.mean(accuracies)}")
    return np.mean(accuracies)
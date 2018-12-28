
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 15:50:01 2018

@author: Rathi Bapat Makashir
"""


# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids: Rohit Bapat rbapat, Amit Makashir abmakash, Akshay Rathi arathi
#
# (Based on skeleton code by D. Crandall)
#
#
####
'''
 Initially, we train our data with respect to the following 3 models -
 Simple - Naive Bayes, Complex MCMC and HMM.
 For the same, we form dictionaries of the following type -
 
 P(W1|S1), P(W2|S2),...and so on - Emission Probability (variable name => )
 P(S2|S1), P(S3|S2),...and so on - Transition Probability
 P(S3|S2,S1), P(S4|S3,S2),...and so on - MCMC Transition Probability
 P(S1) for all sentences  - Priors
 P(S1),P(S2),....P(S12) for all 12 types of part of speech - Figure of Speech (fos) Probability

 Bayes Formula - P(S|W) = P(W|S)*P(S)/P(W)

 Now, that our model is trained with values for all types of probabilities,
 we take the test file input, sentence by sentence and break it into words.

 For each word, we predict the corresponding part of speech with respect to
 each model. This is written in the functions - simplified(), complex_mcmc() and hmm_viterbi()
 
 These predicted parts of speech are now sent as "Label" to posterior() function.
 The posterior function, depending on the model selected, employs the appropriate posterior
 calculation formula and calculates the log of the posterior probability.

 Thus, we finally get the matrix consisting of posterior probabilities and number of words correctly tagged.
 The results of the POS Tagging done by three methods i.e Simple,HMM and Complex MCMC are as follows. We are
 also including the time required for tagging of all 2000 test sentences for 3 models.

 ==> So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
         1. Simple:       93.92%               47.50%
            2. HMM:       95.29%               56.35%
        3. Complex:       93.77%               47.45%
----

 real    6m0.470s
 user    5m51.187s
 sys     0m3.973s

'''

import random
import math
import operator

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
       
class Solver:
    priors = {}
    emission_probability = {}
    word_count = {}
    word_prob = {}
    emission_count={}
    fos={}
    transition={}
    mcmc_transition={}
    emission={}
    
    # Calculates the log of the posterior probability of output sentence (label).
    # Has different formulae for different models.  
    def posterior(self, model, sentence, label):
        # We assume this low probability for all those words which exist in test file but not in training file.We assume the same low probaability for missing transitional probabilities.
        low_prob=0.0000001
        
        # Posterior for simple => P(S|W) = P(W|S)*P(S).
        # We ignore the denominator prior P(W).
        if model == "Simple":
            # prod_simple stores the final posterior value (log) for Simple model
            prod_simple=0
            
            for i in range(len(sentence)):              
                try:
                    prod_simple+=math.log(self.emission[label[i]][sentence[i]]) + math.log(self.fos[label[i]])
                except:
                    # We assign a very low probability if the word does not exist in any of the dictionaries formed wrt our training data.
                    prod_simple+=math.log(low_prob)
            return prod_simple
                
        # Posterior for complex => P(S1,S2,S3,.....,Sn|W1,W2,W3,.....,Wn) = P(S1)*P(W1|S1)*P(S2|S1)*P(W2|S2)*P(S3|S1,S2)*P(W3|S3)........P(Wn|Sn)*P(Sn|Sn-1,Sn-2)
        # Thus, the formula depends on the index of the word (word_ind). For S1 - Prior only. S2 - Transition only P(S2|S1), same as HMM. S3 and beyond - P(S3|S2,S1).
        elif model == "Complex":
            # prod_complex stores the final posterior value (log) for Complex model
            prod_complex=0
            
            for word_ind in range(len(sentence)):
                try:
                    emission_prob = math.log(self.emission[label[word_ind]][sentence[word_ind]])
                except:
                    emission_prob = math.log(low_prob)
                prod_complex+=emission_prob
                
                # S1 needs only calculation of priors
                if word_ind==0:
                    try:
                        prior_val=math.log(self.priors[label[word_ind]])
                    except:
                        prior_val=math.log(low_prob)
                    prod_complex+=prior_val
                    
                # S2 depends only on S1 and W1.
                elif word_ind==1:
                    try:
                        trans_val=math.log(self.transition[label[word_ind-1]][label[word_ind]])
                    except:
                        trans_val=math.log(low_prob)
                    prod_complex+=trans_val
                # S3 and beyond employs the complete posterior formula mentioned above.
                else:
                    try:
                        trans_mcmc_val=math.log(self.mcmc_transition[label[word_ind-2]][label[word_ind-1]][label[word_ind]])
                    except:
                        trans_mcmc_val=math.log(low_prob)
                    prod_complex+=trans_mcmc_val
                
            return  prod_complex
        
        # Posterior calculation for HMM model => P(S1,S2,S3|W1,W2,W3) = P(W1|S1)*P(S1)*P(W2|S2)*P(S2|S1)*P(W3|S3)*P(S3|S2)
        # Thus, unlike MCMC, in HMM, S3 is independent of S1 and is only dependent on S2.
        elif model == "HMM":
            # prod_hmm stores the final posterior value (log) for HMM model
            prod_hmm=0
            
            for word_ind in range(len(sentence)):
                # Emission probability - P(W|S) calculation.
                try:
                    emission_prob = math.log(self.emission[label[word_ind]][sentence[word_ind]])
                except:
                    emission_prob = math.log(low_prob)
                prod_hmm+=emission_prob
                
                # S1 needs only calculation of priors.
                if word_ind == 0:
                    try:
                        prior_val=math.log(self.priors[label[word_ind]])
                    except:
                        prior_val=math.log(low_prob)
                    prod_hmm+=prior_val
                    
                # S2 depends on S1, S3 depends on S2 and so on - Transition Probability calculation.
                else:
                    try:
                        trans_val=math.log(self.transition[label[word_ind-1]][label[word_ind]])
                    except:
                        trans_val=math.log(low_prob)
                    prod_hmm+=trans_val
        
            return prod_hmm
        else:
            print("Unknown algo!")

    # Do the training!
    #
    
    def train(self, data):
        
        # Forming dictionary with prior probability calculation for 1st word of each sentence - P(S1)
        for line in data:
            for words,pos_list in zip(line[::2],line[1::2]):
                if pos_list[0] not in self.priors.keys():
                    self.priors[pos_list[0]]=0
                self.priors[pos_list[0]]+=1       
        total_priors = sum(self.priors.values(), 0.0)
        self.priors = {k: v / total_priors for k, v in self.priors.items()}
        
        # Forming dictionary with HMM transition probabilites - P(S2|S1), P(S3|S2),...and so on.
        for i in range(len(data)):
            for pos in range(len(data[i][1])-1):
                if data[i][1][pos] not in self.transition.keys():
                    self.transition[data[i][1][pos]]={}
                if data[i][1][pos+1] not in self.transition[data[i][1][pos]].keys():
                    self.transition[data[i][1][pos]][data[i][1][pos+1]]=0
                self.transition[data[i][1][pos]][data[i][1][pos+1]]+=1
        for key_val in self.transition.keys():
           total_trans = sum(self.transition[key_val].values(), 0.0)
           self.transition[key_val] = {k: v / total_trans for k, v in self.transition[key_val].items()}        
        
        # Forming dictionary with MCMC transition probabilites - P(S3|S2,S1), P(S4|S3,S2),...and so on.
        for i in range(len(data)):
            for pos in range(len(data[i][1])-2):
                if data[i][1][pos] not in self.mcmc_transition.keys():
                    self.mcmc_transition[data[i][1][pos]]={}
                if data[i][1][pos+1] not in self.mcmc_transition[data[i][1][pos]].keys():
                    self.mcmc_transition[data[i][1][pos]][data[i][1][pos+1]]={}
                if data[i][1][pos+2] not in self.mcmc_transition[data[i][1][pos]][data[i][1][pos+1]].keys():
                    self.mcmc_transition[data[i][1][pos]][data[i][1][pos+1]][data[i][1][pos+2]]=0
                self.mcmc_transition[data[i][1][pos]][data[i][1][pos+1]][data[i][1][pos+2]]+=1

        for k1 in self.mcmc_transition.keys():
            for k2 in self.mcmc_transition[k1].keys():
                total_mcmc = sum(self.mcmc_transition[k1][k2].values(), 0.0)
                self.mcmc_transition[k1][k2] = {key: v / total_mcmc for key, v in self.mcmc_transition[k1][k2].items()}

        # Forming dictionary with probabilities of occurence of each of the 12 parts of speech
        # Thus, "fos" consists of probabilities of noun, conj, verb,... and so on.
        for i in range(len(data)):
            for j in range(len(data[i][1])-1):
                if data[i][1][j] not in self.fos.keys():
                    self.fos[data[i][1][j]]=0
                self.fos[data[i][1][j]]+=1
                    
        for key_val in self.fos.keys():
            total_fos = sum(self.fos.values(), 0.0)
            self.fos = {key: v / total_fos for key, v in self.fos.items()}
        
        # Forming dictionary with emission probabilities - P(W1|S1), P(W2|S2),...and so on.        
        for i in range(len(data)):
            for word in range(len(data[i][1])):
                if data[i][1][word] not in self.emission.keys():
                    self.emission[data[i][1][word]]={}
                if data[i][0][word] not in self.emission[data[i][1][word]].keys():
                    self.emission[data[i][1][word]][data[i][0][word]]=0            
                self.emission[data[i][1][word]][data[i][0][word]]+=1

        for k in self.emission.keys():
            total_emis = sum(self.emission[k].values(), 0.0)
            self.emission[k] = {key: v / total_emis for key, v in self.emission[k].items()}
    
    # Generating samples, forming a random distribution of sample - Gibbs' Sampling
    # Discussed the implementation with Srinithish Kandagadla skandag@iu.edu for iteration definition for sample generation
    def sample_generate(self,sample,sentence):
        sentence_size=len(sentence)
        for word_ind in range(sentence_size):
            random_distribution = {}
            for pos in self.fos.keys():
                sample[word_ind]=pos
                prod=1
                prod=prod*(self.emission[sample[word_ind]][sentence[word_ind]] if sentence[word_ind] in self.emission[sample[word_ind]].keys() else 0.000000001)
                
                if word_ind==0:
                    prod=prod*(self.priors[sample[word_ind]] if sample[word_ind] in self.priors.keys() else 0.00000001)
                elif word_ind==1:
                    prod=prod*(self.transition[sample[word_ind-1]][sample[word_ind]] if sample[word_ind] in self.transition[sample[word_ind-1]].keys() else 0.000000001)
                else:
                    prod=prod*(self.mcmc_transition[sample[word_ind-2]][sample[word_ind-1]][sample[word_ind]] if sample[word_ind-1] in self.mcmc_transition[sample[word_ind-2]].keys() and sample[word_ind] in self.mcmc_transition[sample[word_ind-2]][sample[word_ind-1]].keys() else 0.00000001)
                        
                random_distribution[pos] = float(prod)
                
            distribution=float(sum(random_distribution.values()))
            if distribution != 0:
                for dis_key in random_distribution.keys():
                    random_distribution[dis_key]=float(random_distribution[dis_key]/distribution)
            else:
                random_distribution['noun']=1
            number=random.random()
            for dis_key,dis_val in random_distribution.items():
                if dis_val>number:
                    draw=dis_key
                    break
                else:
                    draw='noun'
            sample[word_ind]=draw
        return sample
                
                    
    # We predict sequence using the emission probabilites and fos probabilites found earlier
    # For simple - P(S|W) is proportional to  P(W|S)*P(S) We neglect the denominator
    def simplified(self, sentence):
        predict=[]
        for words in sentence:
            final={}
            for pos in self.emission.keys():
                if words in self.emission[pos].keys():
                    final[pos] =(self.emission[pos][words]*(self.fos[pos]))
                else:
                    final[pos]=(0.00000000000001*self.fos[pos])
            predict.append(max(final.items(), key=operator.itemgetter(1))[0]) # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        return predict

    # Here, we take a arbitary count of 500 we start with most probable sequence of all nouns. 
    # When iterating over the sequence we put the first 100 iterations as burn ins and consider results for 400 samples genrated by sample generator function
    # We freeze 1 element each time and iterate over the rest and every sample found
    # acts as the initial sample for the next iteration for the particular sample sequence at the end for loop for length of sentence we get a sample sequence,
    # chosen randomly.
    # After getting all such 400 samples we check the maximum ocurrences of POS at each positions and we finalize it.
    # For complex - we include all factors on which the prediction is dependent as per the figure 1.(c)
    def complex_mcmc(self, sentence):
        
        sample = ["noun"] * len(sentence)
        sentence_list=[]
        
        # Number of iterations taken = 500. Burn-in size = 100.
        for i in range(500):
            sample=self.sample_generate(sample,sentence)
            if i >= 100:
                sentence_list.append(list(sample))
                
        sequence=[]
        count_words = {}
        for sentence_seq in sentence_list:
            for i in range(len(sentence_seq)):
                if i not in count_words.keys():
                    count_words[i]={}
                if sentence_seq[i] not in count_words[i].keys():
                    count_words[i][sentence_seq[i]]=0
                count_words[i][sentence_seq[i]]+=1
        for pos_key in count_words.keys():
            sequence.append(max(count_words[pos_key].items(), key=operator.itemgetter(1))[0]) 
        return sequence    
    
    # We predict the sequence in this model by taking into consideration all
    # dependent factors as in figure 1.(b)
    # For viterbi we consider just one state before the current iterable sequence.
    def hmm_viterbi(self, test_sample):
        costs = []
        low_prob=0.00000001
        for index, observed in enumerate(test_sample):
            if index == 0:
                #Initial probability = emission*prior
                arr = {}
		# we consider the intital probability calculation for 1st pos ocirrence probability
                for q in self.fos:
                    initial = -math.log(self.fos[q])
		    # try catch for taking low probability if a complete new word ocurrs
                    try:
                        emission_prob = -math.log(self.emission[q][observed])
                    except:
                        emission_prob = -math.log(low_prob)
                    arr[q] = [emission_prob + initial, q]
                costs.append(arr)
            #for all index later than one we also maximize the product of previous probability and transition probability to new state.
            else:
                costs.append({})
                for q in self.fos:
                    min_arg = {}
                    for i in self.fos:
                        try:
                            trans_prob = -math.log(self.transition[i][q])
                        except:
                            trans_prob = -math.log(low_prob)
    
                        min_arg[i] = costs[index - 1][i][0] + trans_prob
    		    #we are taking -math.log hence we minimize over the function
                    key = min(min_arg, key=lambda x: min_arg[x])  # Find the key with min values in the dictionary
                    try:
                        emission_prob = -math.log(self.emission[q][observed])
                    except:
                        emission_prob = -math.log(low_prob)
                    costs[index][q] = [emission_prob + min_arg[key], key]

        #Backtracking the best sequence
    	# For backtracking we again take a minimum value of argument for each steps we have stored in the costs data
	# we store this sequence of pos at each step
        sequence = []
        i = len(costs) - 1
        key = min(costs[i], key=lambda x: costs[i][x][0])
        while i >= 0:
            sequence.insert(0, key)
            key = costs[i][key][1]
            i -= 1
    
        return sequence

       
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

class Viterbi:
    def __init__(self, ps, cs, val):
        self.prev_state = ps
        self.curr_state = cs
        self.viterbiValue = val


class Solver:
    def __init__(self):
        self.priors = dict()  # prior probabilities of words P(Wi)
        self.pos_priors = dict()  # prior probabilities of tags P(Si)
        self.e_p = dict()  # emission probabilities P(Wi|Si)
        self.t_p = dict()  # transition probabilities P(Si+1 | Si)

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        # Calculate P(S1,S2,...,Sn | W1,W2,...,Wn) for this one instance

        logBase = 10  # another assumption
        wordList = list(sentence)
        # assert len(wordList) == len(label)
        N = len(wordList)

        result = 0
        for idx in range(0, N, 1):
            if wordList[idx] in self.e_p[label[idx]].keys():
                x = self.e_p[label[idx]][wordList[idx]]
            else:
                x = 0.00000000000000000000000000000001  # assumption :P
            result += math.log(x, logBase)

        result += math.log(self.pos_priors[label[0]], logBase)
        prev_label = label[0]
        for idx in range(1, N, 1):
            if label[idx] in self.e_p[prev_label].keys():
                x = self.t_p[prev_label][label[idx]]
            else:
                x = 0.00000000000000000000000000000001  # assumption :P
            result += math.log(x, logBase)
            prev_label = label[idx]

        return result

    # Do the training!
    #
    def train(self, data):
        # P(O): prior words O=Wi
        # P(S): prior lablels S=Si
        # E-S (O): emission of O=Wi in state S=Si
        # T (S1, S2): Transition from past S1=Si to current S2=Sj

        total_word_count = 0
        for d in data:
            total_word_count += len(d[0])
            for i in xrange(len(d[0])):
                word = d[0][i]
                pos = d[1][i]
                # calculate prior for word
                if word in self.priors:
                    self.priors[word] += 1
                else:
                    self.priors[word] = 1
                # calculate prior for pos
                if pos in self.pos_priors:
                    self.pos_priors[pos] += 1
                else:
                    self.pos_priors[pos] = 1
                if pos not in self.e_p: # get actual probability P(Wi | Si)
                    self.e_p[pos] = {word: 1}
                elif word not in self.e_p[pos]:
                    self.e_p[pos][word] = 1
                else:
                    self.e_p[pos][word] += 1
                # calculate first order transition probabilities
                if i > 0:
                    prev_pos = d[1][i - 1]
                    if prev_pos not in self.t_p:
                        self.t_p[prev_pos] = {pos: 1}
                    elif pos not in self.t_p[prev_pos]:
                        self.t_p[prev_pos][pos] = 1
                    else:
                        self.t_p[prev_pos][pos] += 1

        for word in self.priors:
            self.priors[word] = float(self.priors[word]) / float(total_word_count)
        for pos in self.pos_priors:
            self.pos_priors[pos] = float(self.pos_priors[pos]) / float(total_word_count)

        for pos in self.e_p:
            total = 0
            for word in self.e_p[pos]:
                total += self.e_p[pos][word]
            for word in self.e_p[pos]:
                self.e_p[pos][word] = \
                    float(self.e_p[pos][word]) / float(total)
        for prev_pos in self.t_p:
            total = 0
            for pos in self.t_p[prev_pos]:
                total += self.t_p[prev_pos][pos]
            for pos in self.t_p[prev_pos]:
                self.t_p[prev_pos][pos] = \
                    float(self.t_p[prev_pos][pos]) / float(total)

    # Functions for each algorithm.
    #
    def naive(self, sentence):
        # P(Sj | Wj) = [ P(Wj | Sj) * P(Sj) / P(Wj) ]
        # P(Sj | Wj) = [ P(Wj | Sj) * P(Sj) ]

        globalResultList = []
        for aWord in list(sentence):
            globalMaxProb = 0
            globalPOS = 'noun'

            for pos in self.e_p.keys():

                if aWord in self.e_p[pos].keys():
                    localPW = self.e_p[pos][aWord]  # get actual probability P(Wj | Sj)
                    localProb = localPW * self.pos_priors[pos]

                    if (globalMaxProb < localProb):
                        globalMaxProb = localProb
                        globalPOS = pos

            globalResultList.extend([globalPOS])

        # assert len(list(sentence)) == len(globalResultList)
        return [[globalResultList], []]

    def generate_sample(self, sentence, sample):
        sentence_len = len(sentence)
        tags = self.t_p.keys()
        for index in xrange(sentence_len):
            word = sentence[index]
            probabilities = [0] * len(self.t_p)

            s_1 = sample[index - 1] if index > 0 else " "
            s_3 = sample[index + 1] if index < sentence_len - 1 else " "

            for j in xrange(len(self.t_p)):  # try by assigning every tag
                s_2 = tags[j]

                ep = self.e_p[s_2][word] if s_2 in self.e_p and word in self.e_p[s_2] else .000000001
                j_k = self.t_p[s_2][s_3] if s_2 in self.t_p and s_3 in self.t_p[s_2] else .000000001
                i_j = self.t_p[s_1][s_2] if s_1 in self.t_p and s_2 in self.t_p[s_1] else .000000001

                if index == 0:
                    probabilities[j] = j_k * ep * self.pos_priors[s_2]
                elif index == sentence_len - 1:
                    probabilities[j] = i_j * ep * self.pos_priors[s_1]
                else:
                    probabilities[j] = i_j * j_k * ep * self.pos_priors[s_1]

            s = sum(probabilities)
            probabilities = [x / s for x in probabilities]
            rand = random.random()
            p_sum = 0
            for i in xrange(len(probabilities)):
                p = probabilities[i]
                p_sum += p
                if rand < p_sum:
                    sample[index] = tags[i]
                    break

        return sample

    def mcmc(self, sentence, sample_count):
        sample = ["noun"] * len(sentence)  # initial sample, all noun
        for i in xrange(1000):  # ignore first 1000 samples
            sample = self.generate_sample(sentence, sample)
        samples = []
        for p in xrange(sample_count):
            sample = self.generate_sample(sentence, sample)
            samples.append(sample)
        return [samples, []]

    def best(self, sentence):
        return self.viterbi(sentence)
    
    def max_marginal(self, sentence):
        sample_count = 3000
        samples = self.mcmc(sentence, sample_count)[0]
        probabilities = []
        final_sample = []

        for i in xrange(len(sentence)):
            tag_count = dict.fromkeys(self.e_p.keys(), 0)
            for sample in samples:
                tag_count[sample[i]] += 1
            final_sample.append(max(tag_count, key=tag_count.get))
            probabilities.append(tag_count[final_sample[i]] / sample_count)

        return [[final_sample], [probabilities]]

    def viterbi(self, sentence):
        wordList = list(sentence)
        N = len(wordList)
        checkList = [0] * N
        resultPOS = ['noun'] * N

        idx = -1
        for eachWord in wordList:
            idx += 1
            for eachPOS in self.e_p.keys():
                if eachWord in self.e_p[eachPOS].keys():
                    checkList[idx] = 1
                    break

        startIdx = 0
        while (startIdx < N):
            if (checkList[startIdx] == 1):
                endIdx = startIdx
                while ((endIdx < N) and (checkList[endIdx] == 1)):
                    endIdx += 1
                subResult = self.viterbiCalculator(wordList[startIdx:endIdx:1])
                i = startIdx
                j = 0
                # assert len(subResult) == (endIdx-startIdx)
                while (i < endIdx):
                    resultPOS[i] = subResult[j]
                    i += 1
                    j += 1
            startIdx += 1

        # assert len(resultPOS) == len(wordList)
        return [[resultPOS[::1]], []]

    def viterbiCalculator(self, sentence):
        """
            Next, implement the Viterbi Algorithm to find the most likely sequence of state variables,
            (s*1,...,s*N)   =   arg max (s1,...,sN) P(Si = si | W):
        """

        wordList = list(sentence)
        globalResultList = []  # viterbi sequence (reverse list)
        posStates = self.e_p.keys()

        tData = {}
        localMaxPrevState = ''
        for eachStateCurr in posStates:
            if wordList[0] in self.e_p[eachStateCurr].keys():
                tData[eachStateCurr] = Viterbi(localMaxPrevState, eachStateCurr,
                                               self.pos_priors[eachStateCurr] * self.e_p[eachStateCurr][wordList[0]])
            else:
                tData[eachStateCurr] = Viterbi(localMaxPrevState, eachStateCurr, self.pos_priors[
                    eachStateCurr] * 0.00000000000000000000000000000001)  # assumption :P
        firstHalf = [tData]
        secondHalf = list(globalResultList)
        firstHalf.extend(secondHalf)
        globalResultList = firstHalf

        for eachWord in wordList[1:]:
            tData = {}
            for eachStateCurr in posStates:
                localMax = 0.00000000000000000000000000000001  # assumption :P
                localMaxPrevState = '.'  # assumption :P

                for eachStatePrev in posStates:
                    vi = globalResultList[0][eachStatePrev].viterbiValue
                    # print(vi)
                    pij1 = self.t_p[eachStatePrev]
                    if (eachStateCurr in pij1.keys()):
                        pij2 = pij1[eachStateCurr]
                    else:
                        pij2 = 0.00000000000000000000000000000001  # assumption :P
                    # print(pij2)
                    temp = vi * pij2
                    if localMax < temp:
                        localMax = temp
                        localMaxPrevState = eachStatePrev

                if eachWord in self.e_p[eachStateCurr].keys():
                    tData[eachStateCurr] = Viterbi(localMaxPrevState, eachStateCurr,
                                                   localMax * self.e_p[eachStateCurr][eachWord])
                else:
                    tData[eachStateCurr] = Viterbi(localMaxPrevState, eachStateCurr,
                                                   localMax * 0.00000000000000000000000000000001)  # assumption :P

            firstHalf = [tData]
            secondHalf = list(globalResultList)
            firstHalf.extend(secondHalf)
            globalResultList = firstHalf

        # assert len(globalResultList) == len(wordList)

        # return most likely sequence
        resultStates = []
        globalMax = 0
        globalMaxState = ''
        globalMaxPrevState = ''
        for viterbiState in globalResultList[0].keys():
            if globalMax < globalResultList[0][viterbiState].viterbiValue:
                globalMax = globalResultList[0][viterbiState].viterbiValue
                globalMaxState = globalResultList[0][viterbiState].curr_state
                globalMaxPrevState = globalResultList[0][viterbiState].prev_state
        resultStates.extend([globalMaxState])

        if globalMaxPrevState != '':
            resultStates.extend([globalMaxPrevState])
            for eachLevel in globalResultList[1:]:
                temp = eachLevel[resultStates[-1]].prev_state
                if temp != '':
                    resultStates.extend([temp])

        # assert len(resultStates) == len(wordList)
        return resultStates[::-1]

    # The solve() method returns a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #    Most algorithms only return a single labeling per sentence, except for the
    #    mcmc sampler which is supposed to return 5.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for max_marginal() and is the marginal probabilities for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 5)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algo!"

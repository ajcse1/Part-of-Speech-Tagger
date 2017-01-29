#ALGORITHMS

**TRAINING:**            def train(self, data)

We are keeping the trained meta-data in the following data structures:

    self.priors = dict()        # OBSERVED VARIABLES:       prior probabilities of words P(Wi); We tried using this information for best method implementation but shunned it later!

    self.pos_priors = dict()    # UNOBSERVED VARIABLES:     prior probabilities of POS-labels P(Si), or call it probabilities of POS-tags or states Si

    self.e_p = dict()           # EMISSION PROBABILITIES:   P(Wi | Si), i.e., Emission of observation Wi in state Si
    
    self.t_p = dict()           # TRANSITION PROBABILITIES: P(Sj | Si), i.e., Transition from past POS state Si to current POS state Sj


**NAIVE BAYES:**         def naive(self, sentence)
The usage of notations Si and Wi are just explained above.

Here, first, we simply calculated the formula:
P(Si | Wi) = [ P(Wi | Si) * P(Si) / P(Wi) ]

Because the denominator will remain same for all, we ignore the P(Wi) for better computational performance. As we simply calculate following:
P(Si | Wi) = [ P(Wi | Si) * P(Si) ]

For each word Wi, the pos-tag Si with the highest probability P(Si | Wi) is considered for classifying the word Wi.



**MCMC SAMPLER:**        def mcmc(self, sentence, 5)
First we generated a random sample. From this one the subsequent samples were generated using generate_sample method. To
generate a sample, the probability of a pos tag for a word is calculated, given all the other words and their corresponding
tags observed, that is P(S_i | (S - {S_i}), W1,W2,...,Wn). If any emission or transition sequence is not learned from
training data, a small probability (.000000001) is assigned to them. The first 1000 samples were discarded to pass
the warming period and improve sampling accuracy.



**MCMC MAX MARGINAL:**   def max_marginal(self, sentence)
To calculate marginal distribution from samples, we first generated 3000 samples using mcmc (again after discarding the
first 1000 samples). From these samples we calculate max probability of each tag corresponding to each word in the test
sentence. We assign the pos tag which has the maximum probability for a word. And combining them we get the pos tags
for the whole sentence.



**VITERBI MAP:**         def viterbi(self, sentence)
Based on the training data set, we have learned following relevant parameters:
    INITIALS:    pos_priors = dict()    # UNOBSERVED VARIABLES:     prior probabilities of POS-labels P(Si), or call it probabilities of POS-tags or states Si
    EMISSIONS:   e_p = dict()           # EMISSION PROBABILITIES:   P(Wi | Si), i.e., Emission of observation Wi in state Si
    TRANSITIONS: t_p = dict()           # TRANSITION PROBABILITIES: P(Sj | Si), i.e., Transition from past POS state Si to current POS state Sj


Key Idea: We need to calculate the posterior probability of a state sequence, P(S1,...,Sn | W1,...,Wn);
it's an efficient algorithm based on dynamic programming paradigm!

    Computation Assumption For Algorithm Implementation: Below text is taken from slides!
    -- denominator of  bayes expansion of P(S1,...,Sn | W1,...,Wn), depends only on only on Wi (**observed variable**)
    -- Wi depends only on Si, as per the bayes net from assignment question
    -- Markov property "Si+1" depends only on "Si"
    -- So we ignore the denominator during computations of intermediate viterbi values. This is a harmless assumption and simplifies the calculations!


Handling New Words: For new words a small emission probability is assumed which is equal to 0.00000000000000000000000000000001
and such words are labelled with "noun" pos tag

Handling New Transition Edges: Just in case, there happens to be a new transition edge between two existing POS-states, a small transition
probability is assumed which is equal to 0.00000000000000000000000000000001 and such words are labelled with "noun" pos tag;

Final Implementation of Viterbi Sequence Decoding: With above setting at hand we realized that assuming small probabilities for viterbi decoding is not so much effective. So we created
a wrapper over this core algorithm implementation. Instead of sending a complete sentence to viterbi sequence calculator, we send only
those sub-sequence of words, for which there already exits at least one emission probability in our training data structure. The words that
are completely new, for them we simply tag them as "noun" POS-label. This made the viterbi decoding work in a predictable fashion and the
accuracy of this decoder is around 95% in almost all of the cases.



**BEST CLASSIFIER:**     def best(self, sentence)
Considering the restrictions imposed on the scope of programming and resources we can use,
plus, based on the multiple tweaks and test-trials of above 4 algorithms, we realized that viterbi decoding works best for us.
Here we are making a call to viterbi algorithm. Believe that a dynamic programming implementation will result in the most
number of correct sentence tags, i.e., the most likely POS-tag sequence for the given sequence of words.



**POSTERIOR:**           def posterior(self, sentence, label)
This method required us to calculate  probability P(S1,S2,...,Sn | W1,W2,...,Wn) for which we referred emission
probability table and assumed that current observed variable ("word") is dependent only on current unobserved variable ("POS-state").

Going by this, we just need to multiply individual elements. Instead of multiplying all entities first and then taking logarithm,
we preferred to take the logarithm of individual entities first and then sum them all in sequence. This way we did not lose
floating point precision for very low exponents.



#EVALUATION

**NAIVE BAYES:**         def naive(self, sentence)
93.92% of the test words     are correctly classified with POS labels;
47.45% of the test sentences are correctly classified with POS labels sequence;


**MCMC SAMPLER:**        def mcmc(self, sentence, 5)
93.83% of the test words     are correctly classified with POS labels;
47.80% of the test sentences are correctly classified with POS labels sequence;


**MCMC MAX MARGINAL:**   def max_marginal(self, sentence)
93.69% of the test words     are correctly classified with POS labels;
47.40% of the test sentences are correctly classified with POS labels sequence;


**VITERBI MAP:**         def viterbi(self, sentence)
95.26% of the test words     are correctly classified with POS labels;
53.90% of the test sentences are correctly classified with POS labels sequence;


**BEST CLASSIFIER:**     def best(self, sentence)
95.26% of the test words     are correctly classified with POS labels;
53.90% of the test sentences are correctly classified with POS labels sequence;


**COMPUTATIONAL PERFORMANCE:** 
On average, for a test sentence with 15 words, it takes approximately 2.4 seconds for following:
        all the 5 classifiers to predict the POS-labels
        + logarithmic posterior probability to be computed
        + Score class to compute and print the scores


**COMMENTS:**
1. Viterbi decoding is awesome: better sequence prediction + fast computation
2. Naive Bayes is cranky!
3. Gibbs Sampling takes more time as the sample size limit is increased!
4. Max Marginal suffers in performance and accuracy because of above sampler()



#ASSUMPTIONS
If we encounter a new word in test set, then we take priors and emissions as very small probability number: 0.00000000000000000000000000000001;
If we encounter a new word in test set, then we take associated POS tag as "noun"; based on multiple test runs, it comes out to be the most correct guess;
If we encounter a new POS tag in test set, then we take priors, transitions and emissions as very small probability number: 0.00000000000000000000000000000001;


#PROBLEMS FACED
It is very difficult to correctly classify a word that has never been in our training set. Further, since we are implementing the
statistical classification techniques, we need more and more and more training data for better results. "bc.train" file contains only
44204 sentences. Resulting accuracies of classifiers are not practical figures. Limited training data resulted in less mature POS tagging
system; And limited scope of experiments allowed around the traning data set constrained us w.r.t. to gradable algorithms to implement :-)


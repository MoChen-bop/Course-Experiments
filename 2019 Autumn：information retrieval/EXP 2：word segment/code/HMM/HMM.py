class Model(object):
    
    def __init__(self, states, observation, phi, trans_prob, conf_prob):
        self._states = states
        self._observation = observation
        self._phi = phi
        self._trans_prob = trans_prob
        self._conf_prob = conf_prob
    
    def states_length(self):
        return len(self._states)

    def _forward(self,observations):
        s_len = self.states_length
        o_len = len(observations)
        
        alpha = [[] for i in range(o_len)]
        
        alpha[0] = {}
        for state in self._states:
            alpha[0][state] = self._conf_prob[state][observations[0]]*self._phi[state]
        
        for index in range(1,o_len):
            alpha[index] ={}
            for state_to in self._states:
                prob = 0
                for state_from in self._states:
                    prob += alpha[index-1][state_from]*self._trans_prob[state_from][state_to]
                alpha[index][state_to]=self._conf_prob[state_to][observations[index]]*prob
        return alpha
        
    def _backward(self,observations):
        s_len = self.states_length
        o_len = len(observations)
        
        beta = [[] for i in range(o_len)] 
        beta[o_len-1] = {}
        
        for state in self._states:
            beta[o_len-1][state] = 1

        index = len(observations)-1
        while index > 0:
            beta[index-1] = {}
            for state_from in self._states:
                prob = 0
                for state_to in self._states:
                    prob += self._trans_prob[state_from][state_to] * \
                        self._conf_prob[state_to][observations[index]]* \
                        beta[index][state_to]
                beta[index-1][state_from] = prob
            index -= 1
        return beta
        
    def _viterbi(self,observations):
        s_len = self.states_length
        o_len = len(observations)
        
        beta = [[] for i in range(o_len)]
        beta[0] = {}
        
        for state in self._states:
            beta[0][state] = self._conf_prob[state][observations[0]]*self._phi[state]
            
        for index in range(1,o_len):
            beta[index] = {}
            for state_to in self._states:
                prob = []
                for state_from in self._states:
                    temp = beta[index-1][state_from]*self._trans_prob[state_from][state_to]*self._conf_prob[state_to][observations[index]]
                    prob.append(temp)
                prob =sorted(prob,reverse = True)
                beta[index][state_to] = prob[0]
        return beta
    
    def _backward_point(self,beta,observations,state):
        index = len(observations)-1
        theta =[0 for i in range(len(observations))]
        theta[index] = state
        while index >0:
            prob = {}
            for state_from in self._states:
                prob[state_from] = beta[index-1][state_from]*self._trans_prob[state_from][state]
            state = sorted(prob,key=prob.get,reverse=True)[0]
            index -= 1
            theta[index] = state
        return theta
        
    def _inverse(self,beta):
        result = [0 for i in range(len(beta))] 
        length = len(beta)
        for i in range(len(beta)):
            result[i] = beta[length-i-1]
        return result
    
    def _intial_par(self):
        '''
        phi,trans_prob,conf_prob = {},{},{}
        N = len(self._states)
        M = len(self._observation)
        for state in self._states:
            phi[state] = 1.0/N
            trans_prob[state] = {}
            for state_to in self._states:
                trans_prob[state][state_to] = 1.0/N
            conf_prob[state] = {}
            for ob in self._observation:
                conf_prob[state][ob] = 1.0/M
        '''
        phi = self._phi
        trans_prob = self._trans_prob
        conf_prob = self._conf_prob
        return (phi,trans_prob,conf_prob)

    def _cal_gamma(self,alpha,beta,observations):
        T = len(observations)
        gamma = [[] for x in range(T)]
        for t in range(T):
            gamma[t] = {}
            sum_prob = 0
            for state in self._states:
                prob = alpha[t][state]*beta[t][state]
                sum_prob += prob
                gamma[t][state] = prob
            for state in self._states:
                if gamma[t][state] == 0:
                    continue
                else:
                    gamma[t][state] /= sum_prob
        return gamma
        
    def _cal_espi(self,alpha,beta,trans_prob,conf_prob,observations):
        T = len(observations)
        espi = [[] for x in range(T-1)]
        for t in range(T-1):
            espi[t] = {}
            sum_prob = 0
            for state_i in self._states:
                espi[t][state_i] = {}
                for state_j in self._states:
                    prob = alpha[t][state_i]*trans_prob[state_i][state_j]*conf_prob[state_j][observations[t+1]]*beta[t+1][state_j]
                    espi[t][state_i][state_j] = prob
                    sum_prob += prob
            for i in self._states:
                for j in self._states:
                    if espi[t][i][j] == 0:
                        continue
                    else:
                        espi[t][i][j] /= sum_prob
        return espi
        
    def _evaluate_par(self,gamma,espi,observations):
        T = len(observations)
        phi = gamma[0]
        trans_prob,conf_prob = {},{}
        for state in self._states:
            trans_prob[state] = {}
            conf_prob[state] = {}
        for i in self._states:
            for j in self._states:
                gamma_t,espi_t = 0,0
                for t in range(T-1):
                    espi_t += espi[t][i][j]
                    gamma_t += gamma[t][i]
                trans_prob[i][j] = espi_t/gamma_t
        for state in self._states:
            for o in self._observation:
                gamma_con_t ,gamma_t = 0,0
                for t in range(T):
                    if observations[t] == o:
                        gamma_con_t += gamma[t][state]
                    gamma_t = gamma[t][state]
                conf_prob[state][o] = gamma_con_t/gamma_t
        return (phi,trans_prob,conf_prob)
                
    def _iteration(self,phi,trans_prob,conf_prob,observations):
        for index in range(self._iter_num):
            print('第%d次迭代'%(index+1))
            self._phi = phi
            self._trans_prob = trans_prob
            self._conf_prob = conf_prob
            alpha = self._forward(observations)
            beta = self._backward(observations)
            gamma = self._cal_gamma(alpha,beta,observations)
            espi = self._cal_espi(alpha,beta,trans_prob,conf_prob,observations)
            
            phi,trans_prob,conf_prob = self._evaluate_par(gamma,espi,observations)
           
    
    def evaluate(self,observations):
        length = len(observations)
        if length == 0:
            return 0
        
        alpha = self._forward(observations)
        prob = sum(alpha[length-1].values())
        return prob
        
    def decode(self,observations):
        length = len(observations)
        if length == 0 :
            return 0
        beta = self._viterbi(observations)
        
        sequence = beta[length-1]
        state = sorted(sequence,key=sequence.get,reverse=True)[0]
        theta = self._backward_point(beta,observations,state)
        return theta

    def learn(self,observations):        
        phi,trans_prob,conf_prob = self._intial_par()
        
        self._iteration(phi,trans_prob,conf_prob,observations)

import numpy as np 
from GridWord import GridWorldEnv

env=GridWorldEnv()

def value_iteration(env,theta=0.0001,discount_factor=1.0):
    def one_step_lookahead(state,V):
        A=np.zeros(env.nA)
        for a in range(env.nA):
            for prob,next_state,reward,done in env.P[state][a]:
                A[a]+=prob*(reward+discount_factor*V[next_state])
        return A 
    V=np.zeros(env.nS)
    while True:
        delta=0
        for s in range(env.nS):
            #Do a one-step lookahead to find the best action
            A=one_step_lookahead(s,V)
            best_action_value=np.max(A)
            delta=max(delta,np.abs(best_action_value-V[s]))
            V[s]=best_action_value
        if delta<theta:
            break
    policy=np.zeros([env.nS,env.nA])
    for s in range(env.nS):
        A=one_step_lookahead(s,V)
        best_action=np.argmax(A)
        policy[s,best_action]=1.0
    return policy,V
policy,v=value_iteration(env)
print("Policy Probability Distribution:")
print(policy)
print("")
print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

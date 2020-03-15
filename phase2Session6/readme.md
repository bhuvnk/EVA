# Assignment P2S6 : Q-learning Agents for Gridworld


#### Question 1 (50 pts) <br/>
Upload the image of the program (with graphics) running on your computer
**Ans:** ![](../phase2Session6/imgs/screenshot.png)


#### Question 2 (25 pts) <br/>
Write a the pseudo-code (if you paste direct code, will be awarded 0) for `__init__`<br/>

**Ans:**

    INITIALIZE Q-values Counter

#### Question 3 (25 pts) <br/>
Write a the pseudo-code (if you paste direct code, will be awarded 0) for `getQValue` <br/>
**Ans:**

    FUNCTION Q(state,action)
      IF we have never seen a state RETURN 0.0 
      ELSE the Q node value otherwise
      
#### Question 4 (25 pts) <br/>
Write a the pseudo-code (if you paste direct code, will be awarded 0) for `computeValueFromQValue` <br/>
**Ans:**

    COMPUTE legal actions for each state
      IF no legal actions RETURN  0.0
    RETURN max of qvalues of all actions at states
      
#### Question 5 (25 pts) <br/>
Write a the pseudo-code (if you paste direct code, will be awarded 0) for `computeActionFromQValues` <br/>
**Ans:**
    
    COMPUTE Legal Actions for each state
       IF no legal actions RETURN  0.0
    GET Qvalue for all actions and their states
      RETURN action for max Qvalue

#### Question 6 (25 pts) <br/>
Write a the pseudo-code (if you paste direct code, will be awarded 0) for `getAction` <br/>
**Ans:** 

    COMPUTE Legal Actions for each state
      IF action legal
        THEN IF ExplorationProb MORETHAN Random choice
          THEN random Action
        ELSE COMPUTE Action from computeActionFromQValues()
      
      ELSE RETURN action


#### Question 7 (25 pts) <br/>
Write a the pseudo-code (if you paste direct code, will be awarded 0) for `update` <br/>
**Ans:** 
USE ![Q_t\left(s,\:a\right)=\:Q_{t-1}\left(s,\:a\right)\:+\:\alpha TD_t\left(a,\:s\right)](https://render.githubusercontent.com/render/math?math=Q_t%5Cleft(s%2C%5C%3Aa%5Cright)%3D%5C%3AQ_%7Bt-1%7D%5Cleft(s%2C%5C%3Aa%5Cright)%5C%3A%2B%5C%3A%5Calpha%20TD_t%5Cleft(a%2C%5C%3As%5Cright)) to update q value

    Temporal difference = reward +(discount * optimum future value) - Q_old
    
    Q_new = Q_old +Learning rate*temporal difference


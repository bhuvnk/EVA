# Assignment P2S7 : Making The Dabba Delivery 


#### Question 1 (145 pts) <br/> 
Either Embed the video here or share the link to your solution's YouTube Video 
**Ans:** 

`[![https://www.youtube.com/watch?v=F20iX-A3bm4](https://img.youtube.com/vi/F20iX-A3bm4/0.jpg)]()








#### Question 2 (25 pts) <br/>
Paste your Model Architecture here (your __init__ and forward functions)

**Ans:**

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, 50)
        self.fc3 = nn.Linear(50, nb_action)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

#### Question 3 (10 pts)
What happens when "boundary-signal" is weak when compared to the last reward? <br/>
**Ans:** Agent doesn't bounce back from boundary.

#### Question 4 (10 pts) 
What happens when Temperature is reduced?  <br/>
**Ans:** Confidence and Explorations  goes low.

#### Question 5 (10 pts)
What is the effect of reducing What is the effect of reducing ![\gamma](https://render.githubusercontent.com/render/math?math=%5Cgamma) (gamma)? <br/>
**Ans:** Reducing discount factor ![\gamma](https://render.githubusercontent.com/render/math?math=%5Cgamma) makes model, near-sighted by only considering current rewards.

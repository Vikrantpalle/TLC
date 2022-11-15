from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
import numpy as np

class Brain:
    def __init__(self,input_size,action_space) -> None:
        self.input_size = input_size
        self.action_space = action_space
        self.lr = 0.01
        self.model = self.build_model()
        self.model_ = self.build_model()


    def build_model(self):
        model =  Sequential([
            Dense(12, activation = 'relu', input_dim = self.input_size),
            Dense(12, activation='relu'),
            Dense(self.action_space, activation='linear')
        ])

        model.compile(loss = 'mse', optimizer=Adam(learning_rate = self.lr))  

        return model  

    def train(self,x,y,epochs = 1):
        self.model.fit(x,y,batch_size=len(x),epochs = epochs, verbose=0)    

    def predict(self,state,target = False):
        if(target):
            return self.model_.predict(state)
        else:    
            return self.model.predict(state)    

    def update_target_model(self):
        self.model_.set_weights(self.model.get_weights())           


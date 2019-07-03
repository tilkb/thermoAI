import json

class PID:
    def __init__(self, P, I, D):
        self.p=P
        self.i=I
        self.d=D
        self.prev_target=None
    
    def update(self,current_value, target):
        assert(target!=None)
        if target!=self.prev_target:
            self.prev_target=target
            self.Integrator=0
            self.Derivator=0

        error = target - current_value

        P_value = self.p * error
        D_value = self.d * (error - self.Derivator)
        self.Derivator = error
        self.Integrator = self.Integrator + error
        I_value = self.Integrator * self.i

        PID = P_value + I_value + D_value
        return max(0,PID)


    def tuning(self,p,i,d):
        self.i=i
        self.p=p
        self.d=d

    @property
    def components(self):
        return self.p, self.i, self.d



class PIDController:
    def __init__(self):
        self.pid=PID(2000, 0, 0.0)
        self.response_step_count=1

    def control(self,future_required_temperatures, future_outside_temperatures, future_energy_cost, previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption):
        #watch the future for controlling
        target=max(future_required_temperatures[0][0],future_required_temperatures[min(self.response_step_count,len(future_required_temperatures)-1)][0])
        current=previous_inside_temperatures[-1]
        if current>=target+1.0:
            return 0.0
        power = self.pid.update(current,target+0.03*(target-future_outside_temperatures[0]))
        return power

    def train(self,simulator, response_step_count=10):
        #TODO: better parametertuning method
        self.response_step_count=response_step_count
        simulator.reset()
        _, _ ,(_, _, _, _, temp0, _) = simulator.step(0)

        temp0=temp0[-1]
        maxpower = simulator.heat_model.get_max_heating_power()
        for i in range(response_step_count):
            _,_, (_, _, _, _, temp, _) = simulator.step(maxpower)
        temp=temp[-1]    
        max_temp_gain=temp-temp0
        #reach the half of the reachable temp gain in given time
        stepsize=0.01
        target=max_temp_gain*0.5 +temp0
        D=-100.0
        P=0.85*maxpower
        I= 20
        self.pid.tuning(P,I,D)
        for epoch in range(100):
            simulator.reset()
            temp=temp0
            for i in range(response_step_count):
                power=self.pid.update(temp,target)
                _,_, (_, _, _, _, temp, _) = simulator.step(power)
                temp=temp[-1]
            I +=(target-temp)*I*stepsize
            self.pid.tuning(P,I,D)
    
    def save(self,path):
        p, i, d = self.pid.components
        data={"P":p,
              "I":i,
              "D":d}
        with open(path, 'w') as f:  
            json.dump(data, f)

    def load(self,path):
        with open(path) as f:
            param = json.load(f)
        self.pid.tuning(param["P"],param["I"],param["D"])
        
if __name__== "__main__":
    pid = PIDController()
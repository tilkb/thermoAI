import plotly.offline as py
import plotly.graph_objs as go

class Statistics:
    def __init__(self, simulator, controllers, temperature_bound=0.05):
        self.sim = simulator
        self.stat=[]
        self.temperature_bound = temperature_bound
        for name, controller in controllers:
            self.sim.reset()
            for t in range(self.sim.prev_states_count):
                self.sim.step(0)
            
            done=False
            single_controller=[]
            power=0
            while not(done):
                done,reward, (future_required_temperatures, future_outside_temperatures,future_energy_cost,
                previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption) = self.sim.step(power)
                power = controller.control(future_required_temperatures, future_outside_temperatures,future_energy_cost,
                previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption)
                measure={"inside_temperature": previous_inside_temperatures[-1],
                "outside_temperature": previous_outside_temperatures[-1],
                "heating_power": self.sim._last_heating_power,
                "energy_cost": future_energy_cost[0],
                "lower_limit": future_required_temperatures[0][0],
                "upper_limit": future_required_temperatures[0][1],
                "reward": reward}
                single_controller.append(measure)
            self.stat.append((name, single_controller)) 

            
    def print_result(self):
        for name, measures in self.stat:
            total=0
            out=0
            consumption=0.0
            cost=0.0
            mse_error=0.0
            reward=0.0
            for measure in measures:
                total += self.sim.time_step_size_minute
                if measure["inside_temperature"]-self.temperature_bound>measure["upper_limit"] or measure["inside_temperature"]+self.temperature_bound<measure["lower_limit"]:
                    out += self.sim.time_step_size_minute
                    if measure["inside_temperature"]-self.temperature_bound>measure["upper_limit"]:
                        mse_error+= ((measure["inside_temperature"] - measure["upper_limit"]) ** 2) * (1.0 / self.sim.time_step_size_minute)
                    else:
                        mse_error+= ((measure["inside_temperature"] - measure["lower_limit"]) ** 2) * (1.0 / self.sim.time_step_size_minute)
                current_consumption= measure["heating_power"]/1000 * (1 /60 * self.sim.time_step_size_minute)
                consumption+=current_consumption
                cost += current_consumption * measure["energy_cost"]
                reward+= measure["reward"]
            
            print("________________________________________________")
            print("Results for " + name)
            print("Time spent out of range: {0} minute /{1} minute".format(out, total))
            print("Total consumption [kWh]: {0}".format(consumption))
            print("Total cost [HUF]: {0}".format(cost))
            print("Weighted MSE error: {0}".format(mse_error))
            print("Total reward: {0}".format(reward))
            print("________________________________________________")
        
    
    def plot(self):
        time =[t*self.sim.time_step_size_minute/1440 for t in range(len(self.stat[0][1]))]
        low = [x["lower_limit"] for x in self.stat[0][1]]
        high = [x["upper_limit"] for x in self.stat[0][1]]
        outside = [x["outside_temperature"] for x in self.stat[0][1]]

        bound_low = go.Scatter(
        x = time,
        y = low,
        mode = 'lines',
        name = 'Lower limit')

        bound_high = go.Scatter(
        x = time,
        y = high,
        mode = 'lines',
        name = 'Higher limit',
        fill='tonexty')

        out = go.Scatter(
        x = time,
        y = outside,
        mode = 'lines',
        name = 'Outside temperature')
        
        plots=[]
        for name, measures in self.stat:
            power = [m['heating_power'] for m in measures]
            inside = [m['inside_temperature'] for m in measures]
            reward = [m['reward'] for m in measures]
    
            r = go.Scatter(
            x = time,
            y = reward,
            mode = 'lines',
            name = 'Reward: ' + name,
            yaxis = 'y2')
            plots.append(r)

            p = go.Scatter(
            x = time,
            y = power,
            mode = 'lines',
            name = 'Heating power: ' + name,
            yaxis = 'y2')
            plots.append(p)

            ins = go.Scatter(
            x = time,
            y = inside,
            mode = 'lines',
            name = 'Inside temperature: ' + name)
            plots.append(ins)

        data = [bound_low, bound_high, out] + plots
        layout = go.Layout(
            title='Heating plot',
            yaxis=dict(
                title='temperature[°C]'
            ),
            yaxis2=dict(
                title='power[W]',
                titlefont=dict(
                color='rgb(148, 103, 189)'
            ),
            tickfont=dict(
                color='rgb(148, 103, 189)'
            ),
            overlaying='y',
            side='right'
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename='heating_plot.html')


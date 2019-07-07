# Simulator
The main purpose of this module is to create a controlled sytem for creating proof of concepts.
## Assumptions of the simulator
* Controlling is done in predefined discrete timesteps
* The controller module provides the heating power for the next time step. This heating power is determined in [W].
* Temperatures are determined in [°C]

## Cost
Cost of the energy is often not constant(eg. daytime/nightime price). Further plan for improve this modul for reneweble energy flexiible pricing.


## House thermal model
This model is a very basic heating system model.
Consisting of one thermal capacity and one resistance, this model is derived from the
    hourly dynamic model of the ISO 13790. It models heating energy demand only.

Assumptions:
* optimal heating
* only the outside temperature is used

## Statistics
The main purpose of this module to simulate the heating for a heating season and collect the interesting metrics.




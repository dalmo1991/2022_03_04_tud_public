import pandas as pd
from utils.m01 import m01
import spotpy
import numpy as np

# Collect and organize data
data = pd.read_csv('data/Maimai.csv')
data.set_index(pd.to_datetime((data.year*10000+data.month*100+data.day).apply(str),format='%Y%m%d'), inplace=True)
data.drop(columns=['year', 'month', 'day'], inplace=True)

# Set input to the model and time-step
# We do not simulate the whole time series. Only until ts 830
m01.set_input([
    data['P(mm/d)'].values[:830],
    data['E(mm/d)'].values[:830],
])
m01.set_timestep(1.0)

# Model definition for spotpy
class SpotpyHydroModel():

    def __init__(self, model, observations, parameters, parameter_names, output_index, log_transformed, end_warmup=0):
        """
        Model interface for SuperflexPy to SpotPy.

        Parameters
        ----------
        model : superflexpy model
            Initialized SuperflexPy model with inputs and time step already
            defined.
        observations : np.array
            Array of observed data to use in calibration
        ...... TODO: conclude
        """
        self._model = model

        self._parameters = parameters
        self._parameter_names = parameter_names
        self._observarions = observations
        self._output_index = output_index
        self._end_warmup = end_warmup
        self._log_transformed = log_transformed

    def parameters(self):
        return spotpy.parameter.generate(self._parameters)

    def simulation(self, parameters):

        named_parameters = {}
        for i, (p_name, p) in enumerate(zip(self._parameter_names, parameters)):
            named_parameters[p_name] = np.exp(p) if self._log_transformed[i] else p

        self._model.set_parameters(named_parameters)
        self._model.reset_states()
        output = self._model.get_output()

        return output[self._output_index]

    def evaluation(self):
        return self._observarions

    def objectivefunction(self, simulation, evaluation):

        obj_fun = spotpy.objectivefunctions.nashsutcliffe(
            evaluation=evaluation[self._end_warmup:],
            simulation=simulation[self._end_warmup:]
        )

        return -obj_fun # Negative because the calibration algorithm MINIMIZES the obj function

spotpy_model = SpotpyHydroModel(
    model=m01,
    observations=data['Q_obs(mm/d)'].values[:830],
    parameters=[
        spotpy.parameter.Uniform('m01_FR_k', np.log(1e-8), np.log(1.0)),
        spotpy.parameter.Uniform('m01_FR_alpha', np.log(1.0), np.log(10.0)),
        spotpy.parameter.Uniform('m01_FR_Ce', np.log(0.7), np.log(1.3)),
    ],
    parameter_names=['m01_FR_k', 'm01_FR_alpha', 'm01_FR_Ce'],
    output_index=0,
    log_transformed=[True]*3,
    end_warmup=100
)

sampler = spotpy.algorithms.sceua(spotpy_model, dbname='calibration', dbformat='csv')
sampler.sample(repetitions=5000)

# m01.reset_states()
# m01.set_parameters({'m01_FR_k': 0.186034})
# Q_cal = m01.get_output()[0]

# import matplotlib.pyplot as plt
# plt.plot(Q_cal)
# plt.plot(data['Q_obs(mm/d)'].values[:830])
# plt.show()
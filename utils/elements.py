from superflexpy.framework.element import ODEsElement
from superflexpy.framework.element import ParameterizedElement
import numpy as np
import numba as nb

class PowerReservoirET(ODEsElement):

    def __init__(self, parameters, states, approximation, id):
        """
        This element implements a power reservoir with evapotranspiration.

        Q=kS^alpha
        E=Ce*PET*(1-exp(-(S/m)))

        PARAMETERS
        ----------
        parameters : dict
            Parameters of the model. Keys: 'k', 'alpha', 'Ce', 'm
        states : dict
            State of the model. Key: 'S0'
        approximation : superflexpy.utils.numerical_approximation.NumericalApproximator
            Numerial method used to approximate the differential equation
        id : str
            Itentifier of the element. All the elements of the framework must
            have an id.
        """

        ODEsElement.__init__(self, parameters, states, approximation, id)

        self._fluxes_python = [self._fluxes_function_python]  # Used by get fluxes, regardless of the architecture

        if approximation.architecture == 'numba':
            self._fluxes = [self._fluxes_function_numba]
        elif approximation.architecture == 'python':
            self._fluxes = [self._fluxes_function_python]

    def set_input(self, input):
        """
        Set the input of the element.

        Parameters
        ----------
        input : list(numpy.ndarray)
            List containing the input fluxes of the element. It contains 1
            flux:
            1. Rainfall
            2. PET
        """

        self.input = {'P': input[0], 'PET': input[1]}

    def get_output(self, solve=True):
        """
        This method solves the differential equation governing the routing
        store.

        Returns
        -------
        list(numpy.ndarray)
            Output fluxes in the following order:
            1. Streamflow (Q)
        """

        if solve:
            self._solver_states = [self._states[self._prefix_states + 'S0']]
            self._solve_differential_equation()

            # Update the state
            self.set_states({self._prefix_states + 'S0': self.state_array[-1, 0]})

        fluxes = self._num_app.get_fluxes(fluxes=self._fluxes_python,  # I can use the python method since it is fast
                                          S=self.state_array,
                                          S0=self._solver_states,
                                          dt=self._dt,
                                          **self.input,
                                          **{k[len(self._prefix_parameters):]: self._parameters[k] for k in self._parameters},
                                          )

        return [- fluxes[0][1]]

    def get_AET(self):
        """
        This method calculates the actual evapotranspiration

        Returns
        -------
        numpy.ndarray
            Array of actual evapotranspiration
        """

        try:
            S = self.state_array
        except AttributeError:
            message = '{}get_AET method has to be run after running '.format(self._error_message)
            message += 'the model using the method get_output'
            raise AttributeError(message)

        fluxes = self._num_app.get_fluxes(fluxes=self._fluxes_python,
                                          S=S,
                                          S0=self._solver_states,
                                          dt=self._dt,
                                          **self.input,
                                          **{k[len(self._prefix_parameters):]: self._parameters[k] for k in self._parameters},
                                          )

        return [- fluxes[0][2]]

    @staticmethod
    def _fluxes_function_python(S, S0, ind, P, PET, k, alpha, Ce, m, dt):

        if ind is None:
            return (
                [
                    P,
                    - k * S**alpha,
                    - Ce * PET * (1 - np.exp(-(S/m))),
                ],
                0.0,
                S0 + P * dt
            )
        else:
            return (
                [
                    P[ind],
                    - k[ind] * S**alpha[ind],
                    - Ce[ind] * PET[ind] * (1 - np.exp(-(S/m[ind]))),
                ],
                0.0,
                S0 + P[ind] * dt[ind]
            )

    @staticmethod
    @nb.jit('Tuple((UniTuple(f8, 3), f8, f8))(optional(f8), f8, i4, f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])',
            nopython=True)
    def _fluxes_function_numba(S, S0, ind, P, PET, k, alpha, Ce, m, dt):
        # This method is used only when solving the equation

        return (
            (
                P[ind],
                - k[ind] * S**alpha[ind],
                - Ce[ind] * PET[ind] * (1 - np.exp(-(S/m[ind]))),
            ),
            0.0,
            S0 + P[ind] * dt[ind]
        )

class ParameterizedSingleFluxSplitter(ParameterizedElement):
    _num_downstream = 2
    _num_upstream = 1

    def __init__(self, parameters, id):
        """
        parameters : dict
            Parameters of the element. The keys must be:
            - split-par : Parameter used for splitting; split-par goes to the
                          first downstream element, (1 - split-par) goes to the
                          second downstream element
        id : str
            Itentifier of the element. All the elements of the framework must have
            an id.
        """

        ParameterizedElement.__init__(self, parameters, id)

    def set_input(self, input):

        self.input = {'Q_in': input[0]}

    def get_output(self, solve=True):

        split_par = self._parameters[self._prefix_parameters + 'split-par']

        return [
            [self.input['Q_in'] * split_par],
            [self.input['Q_in'] * (1 - split_par)]
        ]

from superflexpy.framework.unit import Unit
from .elements import PowerReservoirET
from superflexpy.implementation.root_finders.pegasus import PegasusPython
from superflexpy.implementation.numerical_approximators.implicit_euler import ImplicitEulerPython
# from superflexpy.implementation.root_finders.pegasus import PegasusNumba as PegasusPython # Warning: dangerous but effective
# from superflexpy.implementation.numerical_approximators.implicit_euler import ImplicitEulerNumba as ImplicitEulerPython # Warning: dangerous but effective

PARAMETERS_MAIMAI_M01 = {
    'm01_FR_k': 1e-8,
    'm01_FR_alpha': 5.30,
    'm01_FR_Ce': 1.074082613,
}

PARAMETERS_HUEWELERBACH_M01 = {
    'm01_FR_k': 0.003935576215328,
    'm01_FR_alpha': 1.0,
    'm01_FR_Ce': 0.98493884385023,
}

PARAMETERS_WOLLEFSBACH_M01 = {
    'm01_FR_k': 1e-8,
    'm01_FR_alpha': 5.57990675511199,
    'm01_FR_Ce': 1.42312151778512,
}

PARAMETERS_WEIERBACH_M01 = {
    'm01_FR_k': 1e-8,
    'm01_FR_alpha': 4.34329024018006,
    'm01_FR_Ce': 0.77318384036332,
}

root_finder = PegasusPython(iter_max=1000)
num_app = ImplicitEulerPython(root_finder=root_finder)

fast_reservoir = PowerReservoirET(
    parameters={'k': 0.1, 'alpha': 1.0, 'Ce': 1.0, 'm': 0.5},
    states={'S0': 20.0},
    approximation=num_app,
    id='FR'
)

m01 = Unit(
    layers=[[fast_reservoir]],
    id='m01'
)

# if __name__ == '__main__':
#     import pandas as pd
#     data = pd.read_csv('data/Maimai.csv')
#     data.set_index(pd.to_datetime((data.year*10000+data.month*100+data.day).apply(str),format='%Y%m%d'), inplace=True)
#     data.drop(columns=['year', 'month', 'day'], inplace=True)
#     m01.set_parameters(PARAMETERS_MAIMAI)
#     START_SIMULATE = 50
#     START_VISUALIZE = 50 # Relative to START_SIMULATE
#     END_SIMULATE = 830 # Absolute
#     P = data['P(mm/d)'].values[START_SIMULATE:END_SIMULATE]
#     E = data['E(mm/d)'].values[START_SIMULATE:END_SIMULATE]
#     Q_obs = data['Q_obs(mm/d)'].values[START_SIMULATE:END_SIMULATE]

#     m01.set_input([P, E])
#     m01.set_timestep(1.0)
#     m01.set_parameters({'m01_FR_k': 1e-7})

#     out = m01.get_output()
#     aet = m01.call_internal('FR', 'get_aet')
#     pass
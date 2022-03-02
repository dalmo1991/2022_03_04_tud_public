from superflexpy.framework.unit import Unit
from superflexpy.implementation.elements.hbv import PowerReservoir, UnsaturatedReservoir
from superflexpy.implementation.root_finders.pegasus import PegasusPython
from superflexpy.implementation.numerical_approximators.implicit_euler import ImplicitEulerPython

PARAMETERS_MAIMAI_M02 = {
    'm02_FR_k': 0.002008225846719,
    'm02_FR_alpha': 2.95777066867277,
    'm02_UR_Ce': 0.998628191394559,
    'm02_UR_Smax': 68.8232043684795,
    'm02_UR_beta': 5.499704486484,
}

PARAMETERS_HUEWELERBACH_M02 = {
    'm02_FR_k': 1e-8,
    'm02_FR_alpha': 5.67905153949512,
    'm02_UR_Ce': 2.58940043581888,
    'm02_UR_Smax': 7928.16810807161,
    'm02_UR_beta': 0.37569024417246,
}

PARAMETERS_WOLLEFSBACH_M02 = {
    'm02_FR_k': 0.000850704165825,
    'm02_FR_alpha': 3.97122477852227,
    'm02_UR_Ce': 1.1563965025296,
    'm02_UR_Smax': 62.054665864139,
    'm02_UR_beta': 9.27038361675023,
}

PARAMETERS_WEIERBACH_M02 = {
    'm02_FR_k': 0.002139433659078,
    'm02_FR_alpha': 2.16587447669636,
    'm02_UR_Ce': 0.763629137970332,
    'm02_UR_Smax': 115.798257318686,
    'm02_UR_beta': 10.0,
}

root_finder = PegasusPython(iter_max=1000)
num_app = ImplicitEulerPython(root_finder=root_finder)

unsaturated_reservoir = UnsaturatedReservoir(
    parameters={'Smax': 200.0, 'Ce': 1.0, 'm': 0.01, 'beta': 0.02},
    states={'S0': 100.0},
    approximation=num_app,
    id='UR'
)

fast_reservoir = PowerReservoir(
    parameters={'k': 0.1, 'alpha': 1.0},
    states={'S0': 2.0},
    approximation=num_app,
    id='FR'
)

m02 = Unit(
    layers=[[unsaturated_reservoir], [fast_reservoir]],
    id='m02'
)
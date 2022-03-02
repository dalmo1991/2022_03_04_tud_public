from superflexpy.framework.unit import Unit
from superflexpy.implementation.elements.hbv import PowerReservoir, UnsaturatedReservoir
from superflexpy.implementation.elements.thur_model_hess import HalfTriangularLag
from superflexpy.implementation.root_finders.pegasus import PegasusPython
from superflexpy.implementation.numerical_approximators.implicit_euler import ImplicitEulerPython

PARAMETERS_MAIMAI_M03 = {
    'm03_FR_k': 0.002008225846719,
    'm03_FR_alpha': 2.95777066867277,
    'm03_UR_Ce': 0.998628191394559,
    'm03_UR_Smax': 68.8232043684795,
    'm03_UR_beta': 5.499704486484,
    'm03_lag_lag-time': 1.0,
}

PARAMETERS_HUEWELERBACH_M03 = {
    'm03_FR_k': 1e-8,
    'm03_FR_alpha': 5.67905153949512,
    'm03_UR_Ce': 2.58940043581888,
    'm03_UR_Smax': 7928.16810807161,
    'm03_UR_beta': 0.37569024417246,
    'm03_lag_lag-time': 1.0,
}

PARAMETERS_WOLLEFSBACH_M03 = {
    'm03_FR_k': 1e-8,
    'm03_FR_alpha': 9.27724341538998,
    'm03_UR_Ce': 1.16154970781699,
    'm03_UR_Smax': 63.4744786006809,
    'm03_UR_beta': 8.84829450076127,
    'm03_lag_lag-time': 1.147729869,
}

PARAMETERS_WEIERBACH_M03 = {
    'm03_FR_k': 1e-8,
    'm03_UR_Ce': 0.772235647042724,
    'm03_FR_alpha': 5.21310614901282,
    'm03_UR_Smax': 116.630704329363,
    'm03_UR_beta': 10.0,
    'm03_lag_lag-time': 2.117041512,
}

root_finder = PegasusPython(iter_max=1000)
num_app = ImplicitEulerPython(root_finder=root_finder)

unsaturated_reservoir = UnsaturatedReservoir(
    parameters={'Smax': 200.0, 'Ce': 1.0, 'm': 0.01, 'beta': 0.02},
    states={'S0': 100.0},
    approximation=num_app,
    id='UR'
)

lag = HalfTriangularLag(
    parameters={'lag-time': 1.0},
    states={'lag': None},
    id='lag'
)

fast_reservoir = PowerReservoir(
    parameters={'k': 0.1, 'alpha': 1.0},
    states={'S0': 2.0},
    approximation=num_app,
    id='FR'
)

m03 = Unit(
    layers=[[unsaturated_reservoir], [lag], [fast_reservoir]],
    id='m03'
)
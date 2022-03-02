from superflexpy.framework.unit import Unit
from superflexpy.implementation.elements.hbv import PowerReservoir, UnsaturatedReservoir
from .elements import ParameterizedSingleFluxSplitter
from superflexpy.implementation.elements.structure_elements import Junction
from superflexpy.implementation.root_finders.pegasus import PegasusPython
from superflexpy.implementation.numerical_approximators.implicit_euler import ImplicitEulerPython

PARAMETERS_MAIMAI_M04 = {
    'm04_FR_k': 0.001,
    'm04_FR_alpha': 3.27613672791991,
    'm04_UR_Ce': 0.894997932365674,
    'm04_UR_Smax': 57.836578528951,
    'm04_UR_beta': 10.0,
    'm04_SR_k': 1e-6,
    'm04_split_split-par': 0.123107044240795,
}

PARAMETERS_HUEWELERBACH_M04 = {
    'm04_FR_k': 0.002128839849116,
    'm04_FR_alpha': 4.4212935509882,
    'm04_UR_Ce': 1.99192496796301,
    'm04_UR_Smax': 2153.01822789773,
    'm04_UR_beta': 0.390898552324631,
    'm04_SR_k': 0.004617020489129,
    'm04_split_split-par': 0.597893083260982,
}

PARAMETERS_WOLLEFSBACH_M04 = {
    'm04_FR_k': 0.004070749686149,
    'm04_FR_alpha': 3.45823609934515,
    'm04_UR_Ce': 1.1702127655603,
    'm04_UR_Smax': 61.4514073761881,
    'm04_UR_beta': 10.0,
    'm04_SR_k': 0.000916739466627,
    'm04_split_split-par': 0.0,
}

PARAMETERS_WEIERBACH_M04 = {
    'm04_FR_k': 0.003454773102112,
    'm04_FR_alpha': 2.06272140965805,
    'm04_UR_Ce': 0.771711960349972,
    'm04_UR_Smax': 117.195201845032,
    'm04_UR_beta': 10.0,
    'm04_SR_k': 0.00157523385576,
    'm04_split_split-par': 0.0,
}

root_finder = PegasusPython(iter_max=1000)
num_app = ImplicitEulerPython(root_finder=root_finder)

unsaturated_reservoir = UnsaturatedReservoir(
    parameters={'Smax': 200.0, 'Ce': 1.0, 'm': 0.01, 'beta': 0.02},
    states={'S0': 100.0},
    approximation=num_app,
    id='UR'
)

splitter = ParameterizedSingleFluxSplitter(
    parameters={'split-par': 0.3},
    id='split'
)
fast_reservoir = PowerReservoir(
    parameters={'k': 0.1, 'alpha': 1.0},
    states={'S0': 2.0},
    approximation=num_app,
    id='FR'
)

slow_reservoir = PowerReservoir(
    parameters={'k': 0.001, 'alpha': 1.0},
    states={'S0': 100.0},
    approximation=num_app,
    id='SR'
)

junction = Junction(
    direction=[
        [0, 0]
    ],
    id='junction'
)

m04 = Unit(
    layers=[
        [unsaturated_reservoir],
        [splitter],
        [slow_reservoir, fast_reservoir],
        [junction]
    ],
    id='m04'
)
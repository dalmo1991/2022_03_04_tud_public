import matplotlib.pyplot as plt

def parameters_sensitivity(model, default_parameters, par_name, par_values, output_type='figure'):
    """
    This function takes a model and runs it with different parameter values and
    returns either a figure or the raw data.

    Parameters
    ----------
    model : superflexpy
        SuperflexPy model
    default_parameters : dict
        Set of default parameters to run the model with. Follow format
        indications of set_parameters method of SuperflexPy
    par_name : str
        Name of the parameter to change
    par_values : list
        List of parameters values to try. Recommend to not exceed 5-7 values.
    output : str
        'figure' or 'data' to specify output type
    """

    if not isinstance(par_values, list):
        par_values = [par_values]

    model.set_parameters(default_parameters)

    output = []

    # Find the UR S0

    state_name, parameter_name = None, None

    for s in model.get_states_name():
        if 'UR' in s:
            state_name = s
            break

    for p in model.get_parameters_name():
        if 'Smax' in p:
            parameter_name = p
            break

    for par_v in par_values:
        model.reset_states()
        model.set_parameters({par_name: par_v})
        if parameter_name is not None:
            s_max_val = model.get_parameters([parameter_name])[parameter_name]
            model.set_states({state_name: 0.2*s_max_val})

        output.append(model.get_output())

    if output_type == 'figure':
        fig, ax = plt.subplots(1, 1)
        for out, par_v in zip(output, par_values):
            ax.plot(out[0], label='{}={}'.format(par_name, par_v))
        ax.legend()
        return fig, ax

    elif output_type == 'data':
        return output
    else:
        raise ValueError('output={} not valid'.format(output_type))
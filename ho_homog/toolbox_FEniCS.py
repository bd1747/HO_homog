# coding: utf-8
"""
Created on 17/01/2019
@author: Baptiste Durand, baptiste.durand@enpc.fr

Collection of tools designed to help users working with FEniCS objects.

"""
import numpy as np
import dolfin as fe
import matplotlib.pyplot as plt

def get_MeshFunction_val(msh_fctn):
    """
    Get information about the set of values that a given MeshFunction outputs.

    Parameters
    ------
    msh_fctn : instance of MeshFunction

    Returns
    -------
    Tuple (nb, vals)
    nb : int
        number of different values
    vals : list
        list of the different values that the MeshFunction outputs on its definition mesh.
    """

    val = np.unique(msh_fctn.array())
    nb = len(val)
    return (nb, val)

def facet_plot2d(facet_func,mesh, mesh_edges=True, markers=None, exclude_val=(0,), **kargs):
    """
    Source : https://bitbucket.org/fenics-project/dolfin/issues/951/plotting-facetfunctions-in-matplotlib-not
    """
    x_list, y_list = [],[]
    if markers == None:
        for facet in fe.facets(mesh):
            mp = facet.midpoint()
            x_list.append(mp.x())
            y_list.append(mp.y())
        values = facet_func.array()
    else:
        i = 0
        values = []
        for facet in fe.facets(mesh):
            if facet_func[i] in markers:
                mp = facet.midpoint()
                x_list.append(mp.x())
                y_list.append(mp.y())
                values.append(facet_func[i])
            i+=1
    if exclude_val:
        filtered_data = [], [], []
        for x, y, val in zip(x_list, y_list, values):
            if val in exclude_val:
                continue
            filtered_data[0].append(x)
            filtered_data[1].append(y)
            filtered_data[2].append(val)
        x_list, y_list, values = filtered_data

    plots = [plt.scatter(x_list, y_list, s=30, c=values, linewidths=1, **kargs)]
    if mesh_edges:
        plots.append(fe.plot(facet_func.mesh()))
    return plots


def function_from_xdmf(function_space, function_name, xdmf_path):
    """Read a finite element function from a xdmf file with checkpoint format.

    Parameters
    ----------
    function_space : dolfin.FunctionSpace
        Function space appropriate for the previously saved function.
        The mesh must be identical to the one used for the saved function.
    function_name : str
        The name of the saved function.
    xdmf_path : pathlib.Path
        Path of the xdmf file. It can be a Path object or a string.
        Extension must be '.xmdf'

    Returns
    -------
    dolfin.Function
        New function that is identical to the previously saved function.
    """
    f = fe.Function(function_space)
    file_in = fe.XDMFFile(str(xdmf_path))
    file_in.read_checkpoint(f, function_name)
    file_in.close()
    return f



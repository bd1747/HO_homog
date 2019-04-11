# coding: utf8
"""
Created on 08/04/2019
@author: baptiste
"""

import logging
import dolfin as fe

logger = logging.getLogger(__name__)

#* For mechanical fields reconstruction
MACRO_FIELDS_NAMES = ['U', 'E', 'EG', 'EGG']


def reconstruction(localization_tensors:dict, macro_kinematic:dict, function_spaces:dict, localization_rules:dict={}, trunc_order:int=0, translation_microstructure=None, output_request=('u','eps')):
    """
    One argument among localization_rules and trunc_order must be used.

    Parameters
    ----------
    localization_tensors : dictionnary
        Format : {
            [Macro_field] : {
                [micro_field] : 
                    [list of lists : for each Macro_field component a list contains the components of the localization tensor field.]
            }
        }
    macro_kinematic : dictionnary of fields
        The macroscopic kinematic fields.
        Keys : 'U', 'EG', 'EGG',\dots
        Values : lists of components. 
        None can be used to indicate a component that is equal to 0 or irrelevant. 
    function_spaces : dictionnary
        Function space into which each mechanical field have to be built.
        keys : 'u', 'eps' or 'sigma'
        values : FEniCS function space
    localization_rules : dict, optional
        Indicates the rules that have to be followed for the construction of the fluctuations.
        (the default is {}, which means that the trunc_order argument will be used)
    trunc_order : int, optional
        Order of truncation for the reconstruction of the displacement field 
        following the higher order homogenization scheme defined in ???.
    translation_microstructure: np.array, optional
        Vector, 1D array (shape (2,) or (3,)), position origin used for the description of the RVE with regards to the macroscopic origin.
    output_request : tuple of strings, optional
        Which fields have to be calculated.
        This can contain : 'u', eps' and 'sigma'
        (the default is ('u', 'eps'), displacement and strain fields will be reconstructed)

    Return
    ------
    Dictionnary
        Mechanical fields with microscopic fluctuations. Keys are "eps", "sigma" and "u" respectively for the strain, stress and displacement fields.
    """
    #? Changer les inputs ? 
        #? Remplacer le dictionnaire de functionspace et output_request par un seul argument : list de tuples qui contiennent nom + function_space ?
        #? Comme ça on peut aussi imaginer reconstruire le même champs dans différents espaces ?

    #! La construction de champs de localisation périodiques doit être faite en dehors de cette fonction.

    #TODO : récupérer topo_dim à partir des tenseurs de localisation, ou mieux, à partir des espaces fonctionnels
    #TODO : choisir comment on fixe la len des listes correspondantes aux composantes de u et de epsilon.

    #Au choix, utilisation de trunc_order ou localization_rules dans les kargs
    if localization_rules:
        pass
    elif trunc_order:
        localization_rules = {
            'u': [(MACRO_FIELDS_NAMES[i], MACRO_FIELDS_NAMES[i]) for i in range(trunc_order+1)],
            'eps': [(MACRO_FIELDS_NAMES[i], MACRO_FIELDS_NAMES[i]) for i in range(1, trunc_order+1)]
        }
    #* Ex. for truncation order = 2:
    #* localization_rules = {
    #*    'u': [('U','U'), ('E','E'), ('EG','EG')],
    #*    'eps': [('E','E'), ('EG', 'EG')]
    #*}

    reconstructed_fields = dict()

    for mecha_field in output_request:
        #* Prepare function spaces and assigner
        fspace = function_spaces[mecha_field]
        value_shape = fspace.ufl_element()._value_shape
        if len(value_shape) == 1:
            vector_dim = value_shape[0]
            mesh = fspace.mesh()
            element = fe.FiniteElement(
                fspace._ufl_element._family,
                mesh.ufl_cell(),
                fspace._ufl_element._degree
                )
            constrain = fspace.dofmap().constrained_domain
            scalar_fspace = fe.FunctionSpace(mesh, element,constrained_domain= constrain)
            assigner = fe.FunctionAssigner(fspace, [scalar_fspace]*vector_dim)
            logger.debug(f"for reconstruction of {mecha_field}, vector_dim = {vector_dim}")
        elif len(value_shape) == 0:
            logger.warning("The value_shape attribute has not been found for the following function space. It is therefore assumed to be a scalar function space for the reconstruction : %s", fspace)
            scalar_fspace = fspace
            vector_dim = 1
            assigner = fe.FunctionAssigner(fspace, scalar_fspace)
        else:
            raise NotImplementedError("Only vector fields are supported by the reconstruction function for now.")

        #* Reconstruction proper
        contributions = [list() for i in range(vector_dim)]
        for macro_key, localization_key in localization_rules[mecha_field]:
            macro_f = macro_kinematic[macro_key]
            loc_tens = localization_tensors[localization_key][mecha_field]
            for macro_comp, loc_tens_comps in zip(macro_f, loc_tens):
                if not macro_comp:
                    continue
                for i in range(vector_dim):
                    contributions[i].append(macro_comp * loc_tens_comps[i])
        components = [sum(compnt_contrib) for compnt_contrib in contributions]

        #* Components -> vector field
        field = fe.Function(fspace)
        components_proj = list()
        for scl_field in components:
            components_proj.append(fe.project(scl_field, scalar_fspace))
        if len(value_shape) == 0:
            components_proj = components_proj[0]
        assigner.assign(field, components_proj)
        reconstructed_fields[mecha_field] = field
    return reconstructed_fields



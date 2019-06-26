# coding: utf-8
"""
Created on 09/01/2019
@author: baptiste

"""

from ho_homog import materials as mat
from ho_homog import part
from ho_homog import homog2d as hom
import dolfin as fe
from pathlib import Path
import numpy as np
import logging
from logging.handlers import RotatingFileHandler


#* Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s', "%H:%M")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


fe.set_log_level(20)




key_conversion = {"U": "u", "Epsilon": "eps", "Sigma": "sig"}



# * Step 2 : Defining the material mechanical properties for each subdomain
E1, nu1 = 1., 0.3
E2, nu2 = E1/100., nu1
E_nu_tuples = [(E1, nu1), (E2, nu2)]

subdo_tags = (1, 2)
material_dict = dict()
for coeff, tag in zip(E_nu_tuples, subdo_tags):
    material_dict[tag] = mat.Material(coeff[0], coeff[1], 'cp')

gen_vect =np.array([[4., 0.], [0., 8.]])

meshes = [Path("panto_coarse.xdmf"), Path("panto_fine.xdmf")]

for mesh in meshes:
    # * Step 3 : Creating the Python object that represents the RVE
    rve = part.Fenics2DRVE.file_2_Fenics_2DRVE(mesh, gen_vect, material_dict)

    # * Step 4 : Initializing the homogemization model
    hom_model = hom.Fenics2DHomogenization(rve)

    # * Step 5 : Computing the homogenized consitutive tensors
    hom_model.homogenizationScheme('E')
    loc_root = mesh.parent.joinpath(mesh.stem+'_loc')
    if not loc_root.exists():
        loc_root.mkdir()
    # test=loc_root.joinpath("test.txt")
    # test.touch()
    for key, scd_dict in hom_model.localization.items():
        if key not in ("E",):
            continue
        for scd_key, localztn_fields in scd_dict.items():
            if scd_key not in ("Sigma", "U"):
                continue
            scd_key = key_conversion[scd_key]
            # * 1 field per component of U, E, EG and EGG
            for i, field in enumerate(localztn_fields):
                loc_path = loc_root.joinpath(f"loc_{key}{i}_{scd_key}.xdmf")
                with fe.XDMFFile(loc_path.as_posix()) as f_out:
                    f_out.write_checkpoint(field, loc_path.stem, 0.0, append=False)

    # chkpt_u = case_data["full_scale"]["checkpoint_u"]
    # full_u = toolbox_FEniCS.function_from_xdmf(displ_fspace, **chkpt_u)

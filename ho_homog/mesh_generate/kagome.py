#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:12:32 2020

@author: manon_thbaut
"""
import numpy as np
import logging
import ho_homog.geometry as geo
import gmsh
import ho_homog.mesh_tools as msh


from . import Gmsh2DRVE, logger

model = gmsh.model
factory = model.occ

# * Logging
logger = logging.getLogger(__name__)  # http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.INFO)


formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s", "%H:%M")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

xdmf_parameters = dict(
    flush_output=False, functions_share_mesh=True, rewrite_function_mesh=False
)


def kagome_RVE(alpha, r, a=None, b=None, nb_cells=(1, 1), offset=(0.0, 0.0), name=""):
    """[summary]

    Parameters
    ----------
    alpha : float
        Paramètre d'ouverture, intervalle [0, 0.5]
        0: configuration refermé
        0.5: configuration complétement ouverte
    name : str

    r : float
        junction thinness, rayon de jonction / côté d'un triangle

    Returns
    -------
    Instance of the Gmsh2DRVE class.

    La cellule est de taille constante,
    par contre les triangles ont une taille qui dépend de l'angle d'ouverture.
    """

    logger.info("Start defining the geometry")
    name = name if name else "kagome"

    if (not a) and (not b):
        raise ValueError("a or b (exclusive OR) must be imposed")
    elif a and b:
        raise ValueError("a and b cannot be imposed simultaneously")
    elif b:
        # si on choisit la taille des triangles b
        a = kagome_triangle_size_2_cell_size(alpha, b)
    elif a:
        b = kagome_cell_size_2_triangle_size(alpha, a)

    gen_vect = np.array(((a, a / 2), (0.0, np.sqrt(3) / 2 * a)))
    nb_cells, offset = np.asarray(nb_cells), np.asarray(offset)

    a0 = np.array((a, 0.0, 0.0))
    a1 = np.array((a / 2, np.sqrt(3) / 2 * a, 0.0))
    phi1 = geo.Point(alpha * a0)
    psi1 = geo.Point((1 - alpha) * a1)
    z_axis = [0, 0, 1]
    angle = 2 * np.pi / 3
    phi2 = geo.rotation(phi1, angle, z_axis)
    psi2 = geo.rotation(psi1, angle, z_axis)
    phi3 = geo.rotation(phi2, angle, z_axis)
    psi3 = geo.rotation(psi2, angle, z_axis)

    star_pts = [phi1, psi1, phi2, psi2, phi3, psi3]
    pattern_ll = [geo.LineLoop(star_pts, explicit=False)]
    trans = [a0, a1, a0 + a1]
    pattern_ll += [geo.translation(pattern_ll[0], t) for t in trans]

    pattern_ll = geo.remove_duplicates(pattern_ll)
    logger.info("Removing duplicate pattern line-loops: Done")
    logger.info(f"Number of pattern line-loops: {len(pattern_ll)}")
    for ll in pattern_ll:
        ll.round_corner_kagome(r * b, a, alpha)
    logger.info("Rounding all corners of pattern line-loops: Done")
    fine_pts = [pt for ll in pattern_ll for pt in ll.vertices]
    fine_pts = geo.remove_duplicates(fine_pts)

    return Gmsh2DRVE(pattern_ll, gen_vect, nb_cells, offset, fine_pts, False, name,)


def kagome_sym_RVE(alpha, name, r, a, b):

    logger.info("Start defining the geometry")
    if a == 0.0 and b == 0.0:
        raise ValueError("a or b must be imposed")
    elif a != 0.0 and b != 0.0:
        raise ValueError("a and b cannot be imposed simultaneously")
    if b == 0.0:
        gen_vect = np.array(((a, -a / 2), (0.0, np.sqrt(3) * a / 2)))

        O = np.zeros((3,))
        a0 = np.array((a, 0.0, 0.0))
        a1 = np.array((a / 2, np.sqrt(3) / 2 * a, 0.0))
        a2 = np.array((-a / 2, np.sqrt(3) / 2 * a, 0.0))
        A0 = geo.Point(a0)
        A1 = geo.Point(a1)
        O_pt = geo.Point(O)

        phi1 = geo.Point(alpha * a0)
        psi1 = geo.Point((1 - alpha) * a1)

        phi2 = geo.rotation(phi1, 2 * np.pi / 3, [0, 0, 1])
        psi2 = geo.rotation(psi1, 2 * np.pi / 3, [0, 0, 1])

        phi3 = geo.rotation(phi2, 2 * np.pi / 3, [0, 0, 1])
        psi3 = geo.rotation(psi2, 2 * np.pi / 3, [0, 0, 1])

        star_pts = [phi1, psi1, phi2, psi2, phi3, psi3]
        pattern_ll = [geo.LineLoop(star_pts, explicit=False)]
        trans = [a0, a1, a2]
        pattern_ll += [geo.translation(pattern_ll[0], t) for t in trans]

        pattern_ll = geo.remove_duplicates(pattern_ll)
        logger.info(
            f"Done removing of the line-loops duplicates. pattern_ll length : {len(pattern_ll)}"
        )
        # 13 no-redundant LineLoop to define the pantographe microstructure geometry in one cell.
        # [pt for pt in constr_pts if (pt.coord[0] % 1 < p[0]/2. or pt.coord[0] % 1 > 1. - p[0]/2.)]
        # fine_pts=[]
        for ll in pattern_ll:

            ll.round_corner_kagome(r, a, alpha)

            # fine_pts+=[f for f in fine_pts_ll]
        fine_pts = [pt for ll in pattern_ll for pt in ll.vertices]
        fine_pts = geo.remove_duplicates(fine_pts)
        logger.info("Done rounding all corners of pattern line-loops")

        macro_vtcs = [
            O,
            gen_vect[:, 0],
            gen_vect[:, 0] + gen_vect[:, 1],
            gen_vect[:, 1],
        ]
        macro_ll = geo.LineLoop([geo.Point(c) for c in macro_vtcs])
        macro_s = geo.PlaneSurface(macro_ll)

        logger.info("Start boolean operations on surfaces")
        pattern_s = [geo.PlaneSurface(ll) for ll in pattern_ll]
        # arc=geo.round_corner(phi1,psi1,phi2,1/20,False,False)
        # pattern_ll+=[arc_ll]
        for ll in pattern_ll:
            ll.add_gmsh()
        A0.add_gmsh()
        A1.add_gmsh()
        O_pt.add_gmsh()
        macro_s.add_gmsh()
        factory.synchronize()
        # gmsh.fltk.run()

        rve_s = geo.surface_bool_cut(macro_s, pattern_s)
        # cell_s = geo.surface_bool_cut(macro_s, rve_s)
        # rve_s = rve_s[0]
        # cell_s=cell_s[0]
        logger.info("Done boolean operations on surfaces")
        rve_s_phy = geo.PhysicalGroup(rve_s, 2, "partition_plein")
        # cell_s_phy = geo.PhysicalGroup(cell_s, 2, "partition_plein")
        factory.synchronize()

        rve_s_phy.add_gmsh()
        # cell_s_phy.add_gmsh()

        factory.synchronize()
        data = model.getPhysicalGroups()
        logger.info(
            "All physical groups in the model "
            + repr(data)
            + " Names : "
            + repr([model.getPhysicalName(*dimtag) for dimtag in data])
        )
        logger.info("Done generating the gmsh geometrical model")
        gmsh.write(f"{name}.brep")
        # os.system(f"gmsh {name}.brep &")

        logger.info("Start defining a mesh refinement constraint")
        # constr_pts = [pt for ll in pattern_ll for pt in ll.vertices]

        for pt in fine_pts:
            pt.add_gmsh()
        # gmsh.fltk.run()
        factory.synchronize()
        # if alpha>0.41 and alpha<0.43 :
        # f = msh.set_mesh_refinement(
        # [r, a / 2],
        #    [r, a],
        #    [r / 10, a / 15],
        #    attractors={"points": fine_pts},
        #    sigmoid_interpol=True,
        #  )
        # else :
        #   f = msh.set_mesh_refinement(
        # [r, a / 2],
        #      [r, a],
        #     [r / 10, a / 10],
        #   attractors={"points": fine_pts},
        #    sigmoid_interpol=True,
        #  )
        f = msh.set_mesh_refinement(
            [r, a],
            [r / 10, a / 10],
            attractors={"points": fine_pts},
            sigmoid_interpol=True,
        )

        msh.set_background_mesh(f)

        logger.info("Done defining a mesh refinement constraint")
        macro_bndry = macro_ll.sides
        boundary = geo.AbstractSurface.get_surfs_boundary(rve_s)
        # rve_s.get_boundary(recursive=True)
        micro_bndry = [geo.macro_line_fragments(boundary, M_ln) for M_ln in macro_bndry]
        dirct = [
            (M_ln.def_pts[-1].coord - M_ln.def_pts[0].coord) for M_ln in macro_bndry
        ]
        logger.debug(
            "value and type of dirct items : " + repr([(i, type(i)) for i in dirct])
        )
        for i, crvs in enumerate(micro_bndry):
            msh.order_curves(crvs, dirct[i % 2], orientation=True)
        logger.debug("length of micro_bndry list : " + str(len(micro_bndry)))

        msh.set_periodicity_pairs(micro_bndry[0], micro_bndry[2])
        msh.set_periodicity_pairs(micro_bndry[1], micro_bndry[3])

        factory.remove([(1, l.tag) for l in macro_ll.sides])
        for l in macro_ll.sides:
            l.tag = None
        pre_val = gmsh.option.getNumber("Mesh.SaveAll")
        val = 0
        gmsh.option.setNumber("Mesh.SaveAll", val)
        logging.info(f"Option Mesh.SaveAll set to {val}. Initial value : {pre_val}.")
        geo.PhysicalGroup.set_group_mesh(1)
        gmsh.model.mesh.generate(2)
        # gmsh.write("%s.msh" % name)
        # os.system("gmsh %s.msh &" % name)

        # gmsh.fltk.run()

        # gmsh.fltk.run()
    elif a == 0.0:  # si on choisit la taille des triangles b
        a = 1.0  # taille du réseau de triangle
        x_a = (1 - alpha) * a / 2 - alpha * a
        y_a = (1 - alpha) * a * np.sqrt(3) / 2
        cote_triangle = np.sqrt(x_a ** 2 + y_a ** 2)
        theta = np.arcsin(
            np.sqrt(3) * alpha / (2 * cote_triangle)
        )  # compute theta for a unit cell

        a = 2 * b * np.cos(np.pi / 3 - theta)  # real cell

        gen_vect = np.array(((a, -a / 2), (0.0, np.sqrt(3) * a / 2)))

        O = np.zeros((3,))
        a0 = np.array((a, 0.0, 0.0))
        a1 = np.array((a / 2, np.sqrt(3) / 2 * a, 0.0))
        a2 = np.array((-a / 2, np.sqrt(3) / 2 * a, 0.0))
        A0 = geo.Point(a0)
        A1 = geo.Point(a1)
        O_pt = geo.Point(O)

        phi1 = geo.Point(alpha * a0)
        psi1 = geo.Point((1 - alpha) * a1)

        phi2 = geo.rotation(phi1, 2 * np.pi / 3, [0, 0, 1])
        psi2 = geo.rotation(psi1, 2 * np.pi / 3, [0, 0, 1])

        phi3 = geo.rotation(phi2, 2 * np.pi / 3, [0, 0, 1])
        psi3 = geo.rotation(psi2, 2 * np.pi / 3, [0, 0, 1])

        star_pts = [phi1, psi1, phi2, psi2, phi3, psi3]
        pattern_ll = [geo.LineLoop(star_pts, explicit=False)]
        trans = [a0, a1, a2]
        pattern_ll += [geo.translation(pattern_ll[0], t) for t in trans]

        pattern_ll = geo.remove_duplicates(pattern_ll)
        logger.info(
            f"Done removing of the line-loops duplicates. pattern_ll length : {len(pattern_ll)}"
        )
        # 13 no-redundant LineLoop to define the pantographe microstructure geometry in one cell.
        # [pt for pt in constr_pts if (pt.coord[0] % 1 < p[0]/2. or pt.coord[0] % 1 > 1. - p[0]/2.)]
        # fine_pts=[]
        for ll in pattern_ll:

            ll.round_corner_kagome(r, a, alpha)

            # fine_pts+=[f for f in fine_pts_ll]
        fine_pts = [pt for ll in pattern_ll for pt in ll.vertices]
        fine_pts = geo.remove_duplicates(fine_pts)
        logger.info("Done rounding all corners of pattern line-loops")

        macro_vtcs = [
            O,
            gen_vect[:, 0],
            gen_vect[:, 0] + gen_vect[:, 1],
            gen_vect[:, 1],
        ]
        macro_ll = geo.LineLoop([geo.Point(c) for c in macro_vtcs])
        macro_s = geo.PlaneSurface(macro_ll)

        logger.info("Start boolean operations on surfaces")
        pattern_s = [geo.PlaneSurface(ll) for ll in pattern_ll]
        # arc=geo.round_corner(phi1,psi1,phi2,1/20,False,False)
        # pattern_ll+=[arc_ll]
        for ll in pattern_ll:
            ll.add_gmsh()
        A0.add_gmsh()
        A1.add_gmsh()
        O_pt.add_gmsh()
        macro_s.add_gmsh()
        factory.synchronize()
        # gmsh.fltk.run()

        rve_s = geo.surface_bool_cut(macro_s, pattern_s)
        # cell_s = geo.surface_bool_cut(macro_s, rve_s)
        # rve_s = rve_s[0]
        # cell_s=cell_s[0]
        logger.info("Done boolean operations on surfaces")
        rve_s_phy = geo.PhysicalGroup(rve_s, 2, "partition_plein")
        # cell_s_phy = geo.PhysicalGroup(cell_s, 2, "partition_plein")
        factory.synchronize()

        rve_s_phy.add_gmsh()
        # cell_s_phy.add_gmsh()

        factory.synchronize()
        data = model.getPhysicalGroups()
        logger.info(
            "All physical groups in the model "
            + repr(data)
            + " Names : "
            + repr([model.getPhysicalName(*dimtag) for dimtag in data])
        )
        logger.info("Done generating the gmsh geometrical model")
        gmsh.write(f"{name}.brep")
        # os.system(f"gmsh {name}.brep &")

        logger.info("Start defining a mesh refinement constraint")
        # constr_pts = [pt for ll in pattern_ll for pt in ll.vertices]

        for pt in fine_pts:
            pt.add_gmsh()
        # gmsh.fltk.run()
        factory.synchronize()
        # if alpha>0.41 and alpha<0.43 :
        # f = msh.set_mesh_refinement(
        # [r, a / 2],
        #    [r, a],
        #    [r / 10, a / 15],
        #    attractors={"points": fine_pts},
        #    sigmoid_interpol=True,
        #  )
        # else :
        #   f = msh.set_mesh_refinement(
        # [r, a / 2],
        #      [r, a],
        #     [r / 10, a / 10],
        #   attractors={"points": fine_pts},
        #    sigmoid_interpol=True,
        #  )
        f = msh.set_mesh_refinement(
            [r, a],
            [r / 10, a / 10],
            attractors={"points": fine_pts},
            sigmoid_interpol=True,
        )

        msh.set_background_mesh(f)

        logger.info("Done defining a mesh refinement constraint")
        macro_bndry = macro_ll.sides
        boundary = geo.AbstractSurface.get_surfs_boundary(rve_s)
        # rve_s.get_boundary(recursive=True)
        micro_bndry = [geo.macro_line_fragments(boundary, M_ln) for M_ln in macro_bndry]
        dirct = [
            (M_ln.def_pts[-1].coord - M_ln.def_pts[0].coord) for M_ln in macro_bndry
        ]
        logger.debug(
            "value and type of dirct items : " + repr([(i, type(i)) for i in dirct])
        )
        for i, crvs in enumerate(micro_bndry):
            msh.order_curves(crvs, dirct[i % 2], orientation=True)
        logger.debug("length of micro_bndry list : " + str(len(micro_bndry)))

        msh.set_periodicity_pairs(micro_bndry[0], micro_bndry[2])
        msh.set_periodicity_pairs(micro_bndry[1], micro_bndry[3])

        factory.remove([(1, l.tag) for l in macro_ll.sides])
        for l in macro_ll.sides:
            l.tag = None
        pre_val = gmsh.option.getNumber("Mesh.SaveAll")
        val = 0
        gmsh.option.setNumber("Mesh.SaveAll", val)
        logging.info(f"Option Mesh.SaveAll set to {val}. Initial value : {pre_val}.")
        geo.PhysicalGroup.set_group_mesh(1)
        gmsh.model.mesh.generate(2)
        # gmsh.write("%s.msh" % name)
        # os.system("gmsh %s.msh &" % name)

        # gmsh.fltk.run()


def kagome_triangle_size_2_cell_size(alpha, b):
    """Passage de b (côté des triangles) à a (taille caractéristique cellule unitaire du kagome)."""
    _a = 1.0  # pour cellule de hauteur 1
    x = (1 - alpha) * _a / 2 - alpha * _a
    y = (1 - alpha) * _a * np.sqrt(3) / 2
    _b = np.sqrt(x ** 2 + y ** 2)
    theta = np.arcsin(np.sqrt(3) * alpha / (2 * _b))
    # compute theta for a unit cell

    a = 2 * b * np.cos(np.pi / 3 - theta)  # real cell
    return a


def kagome_cell_size_2_triangle_size(alpha, a):
    """Pour la microstructure "kagome", passage de a (taille caractéristique de la cellule unitaire) à  b (côté des triangles)

    Parameters
    ----------
    alpha : float
        paramètre d'ouverture de la microstructure
    a : float
        taille cellule unitaire parallélogramme.
    """
    t1 = (1 - alpha) * a / 2 - alpha * a
    t2 = (1 - alpha) * np.sqrt(3) * a / 2
    b = np.sqrt(t1 ** 2 + t2 ** 2)
    return b

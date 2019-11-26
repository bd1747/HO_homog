# coding: utf-8
"""
Created on 17/01/2019
@author: Baptiste Durand, baptiste.durand@enpc.fr

Tools and classes designed to create 2D gmsh model of a periodic metamaterial represented on a certain domain.

The Gmsh2DRVE class has been specially designed for the generation of meshes that represent RVE with a given microstructure.
"""

import copy
import logging
import math
from itertools import chain, product
from pathlib import Path, PurePath

import gmsh
import numpy as np
from more_itertools import flatten, one, padded

import ho_homog.geometry as geo
import ho_homog.mesh_tools as msh

# nice shortcuts
model = gmsh.model
factory = model.occ


logger = logging.getLogger(__name__)  # http://sametmax.com/ecrire-des-logs-en-python/


__all__ = [
    "pantograph",
    "duplicate_pattern",
    "offset_pattern",
    "Gmsh2DRVE",
    "Gmsh2DPart",
    "Gmsh2DPartFromRVE",
]


def duplicate_pattern(cell_ll, nb_cells, gen_vect):
    """
    Propager la cellule de base dans selon les directions de l'espace.

    Parameters
    ----------
    cell_ll : list
        The LineLoops that will be replicated. They define the pattern for a unit cell.
    np_cells : tuple, dimension 2 or 3
        Number of cells in each direction.
    gen_vect : array
        The generating vectors that are related to the given microstruture.

    Returns
    -------
    repeated_ll : list
        Repeated lineloops that define the pattern over the whole domain associated with the given numbers of cells.
    """
    repeated_ll = cell_ll
    if gen_vect.shape != (3, 3):
        gen_vect_3D = np.zeros((3, 3))
        gen_vect_3D[: gen_vect.shape[0], : gen_vect.shape[1]] = gen_vect
    else:
        gen_vect_3D = gen_vect
    for k in range(len(nb_cells)):
        if nb_cells[k] > 1:
            new_contours = list()
            for i in range(1, int(nb_cells[k])):
                new_contours += [
                    geo.translation(ll, i * gen_vect_3D[k]) for ll in repeated_ll
                ]
            repeated_ll += new_contours
    repeated_ll = geo.remove_duplicates(repeated_ll)
    return repeated_ll


def offset_pattern(cell_ll, offset, cell_vect):
    """
    Translation of the lineloops that define the microstructure geometry of a unit cell.

    Parameters
    ----------
    cell_ll : list of instances of LineLoop
    offset : 1D array
        relative coordinates with respect to the unit-cell generating vectors of the point that will be moved to the origin
    gen_vect : 2D array
        The generating vectors that are related to the given unit-cell.
    """
    if cell_vect.shape != (3, 3):
        cell_vect_3D = np.zeros((3, 3))
        cell_vect_3D[: cell_vect.shape[0], : cell_vect.shape[1]] = cell_vect
    else:
        cell_vect_3D = cell_vect
    offset_vect_relat = np.zeros(3)
    for i, val in enumerate(offset):
        offset_vect_relat[i] = val % 1.0
    offset_vect_abs = np.dot(cell_vect_3D, offset_vect_relat)
    t_vect = -1 * offset_vect_abs
    shifted_ll = [geo.translation(ll, t_vect) for ll in cell_ll]
    return shifted_ll


class Gmsh2DRVE(object):
    def __init__(
        self, pattern_ll, cell_vect, nb_cells, offset, attractors, soft_mat, name
    ):
        """
        Contrat : Créer un maillage pour des RVE 2D, plans, comportant au plus 2 matériaux constitutifs et pouvant contenir plusieurs cellules.
        #! La cellule est un parallélogramme.

        Parameters
        ----------
        pattern_ll : list
            Instances of LineLoop that define the contours of the microstructure.
        cell_vect : 2D array
            dimensions of the unit cell and directions of periodicity.
            (given in a 2D cartesian coordinate system)
        nb_cells : 1D array
            Numbers of cells in each direction of repetition/periodicity.
        offset : 1D array
            Relative position inside a cell of the point that will coincide with the origin of the global domain.
        attractors : list
            Instances of Point.
            Can also be = None or empty.
            It represent the points that will be used as attractors in the definition of the element characteristic length fields.
            Attractors are geometrical elements of the cell around which mesh refinement constraints will be set.
        name : string or Path
        """

        self.name = name.stem if isinstance(name, PurePath) else name
        model.add(self.name)
        model.setCurrent(self.name)

        if offset.any():
            nb_pattern = [
                math.ceil(val + 1) if offset[i] != 0 else math.ceil(val)
                for i, val in enumerate(nb_cells)
            ]
            nb_pattern = np.array(nb_pattern, dtype=np.int8)
            pattern_ll = offset_pattern(pattern_ll, offset, cell_vect)
        else:
            nb_pattern = np.int8(np.ceil(nb_cells))

        if not np.equal(nb_pattern, 1).all():
            duplicate_pattern(pattern_ll, nb_pattern, cell_vect)

        rve_vect = cell_vect * nb_cells[:, np.newaxis]
        O = np.zeros((3,))
        macro_vtx = [O, rve_vect[0], rve_vect[0] + rve_vect[1], rve_vect[1]]
        macro_ll = geo.LineLoop([geo.Point(c) for c in macro_vtx])
        macro_s = geo.PlaneSurface(macro_ll)

        if attractors:
            for entity in attractors:
                if not isinstance(entity, geo.Point):
                    raise TypeError(
                        """Use of curves as attractors for the refinement of the mesh
                    is not yet fully supported in our python library for gmsh."""
                    )
            if offset.any():
                attractors = offset_pattern(attractors, offset, cell_vect)
            if not np.equal(nb_pattern, 1).all():
                duplicate_pattern(attractors, nb_pattern, cell_vect)

        logger.info("Start boolean operations on surfaces")
        phy_surf = list()
        pattern_s = [geo.PlaneSurface(ll) for ll in pattern_ll]
        rve_s = geo.surface_bool_cut(macro_s, pattern_s)
        if len(rve_s) == 1:
            logger.info(
                "The main material domain of the RVE is connected (topological property)."
            )
        elif len(rve_s) == 0:
            logger.warning(
                "The boolean operation for creating the main material domain of the RVE return 0 surfaces."
            )
        else:
            logger.warning(
                "The main material domain of the RVE obtained by a boolean operation is disconnected (topological property)."
            )
        rve_s_phy = geo.PhysicalGroup(rve_s, 2, "microstruct_domain")
        phy_surf.append(rve_s_phy)
        if soft_mat:
            soft_s = geo.surface_bool_cut(macro_s, rve_s)
            soft_s_phy = geo.PhysicalGroup(soft_s, 2, "soft_domain")
            phy_surf.append(soft_s_phy)
        logger.info("Done boolean operations on surfaces")

        if attractors:
            need_sync = False
            for entity in attractors:
                if not entity.tag:
                    entity.add_gmsh()
                    need_sync = True
            if need_sync:
                factory.synchronize()  # ? Pourrait être enlevé ?

        for gp in phy_surf:
            gp.add_gmsh()
        factory.synchronize()

        data = model.getPhysicalGroups()
        details = [
            f"Physical group id : {dimtag[1]}, "
            + f"dimension : {dimtag[0]}, "
            + f"name : {model.getPhysicalName(*dimtag)}, "
            + f"nb of entitities {len(model.getEntitiesForPhysicalGroup(*dimtag))} \n"
            for dimtag in data
        ]
        logger.debug(f"All physical groups in the model : {data}")
        logger.debug(f"Physical groups details : \n {details}")
        logger.info("Done generating the gmsh geometrical model")
        if isinstance(name, PurePath):
            gmsh.write(str(name.with_suffix(".brep")))
        else:
            gmsh.write(f"{name}.brep")
        macro_bndry = macro_ll.sides
        if soft_mat:
            boundary = geo.AbstractSurface.get_surfs_boundary(rve_s + soft_s)
        else:
            try:
                s = one(rve_s)
                boundary = geo.AbstractSurface.get_surfs_boundary(s)
            except ValueError:
                boundary = geo.AbstractSurface.get_surfs_boundary(rve_s)
        factory.synchronize()
        micro_bndry = [geo.macro_line_fragments(boundary, M_ln) for M_ln in macro_bndry]
        macro_dir = [
            macro_bndry[i].def_pts[-1].coord - macro_bndry[i].def_pts[0].coord
            for i in range(len(macro_bndry) // 2)
        ]
        for i, crvs in enumerate(micro_bndry):
            msh.order_curves(crvs, macro_dir[i % 2], orientation=True)
        msh.set_periodicity_pairs(micro_bndry[0], micro_bndry[2])
        msh.set_periodicity_pairs(micro_bndry[1], micro_bndry[3])
        logger.info("Done defining a mesh periodicity constraint")
        tags = [
            "per_pair_1_slave",
            "per_pair_2_slave",
            "per_pair_1_mast",
            "per_pair_2_mast",
        ]
        per_pair_phy = list()
        for crvs, tag in zip(micro_bndry, tags):
            per_pair_phy.append(geo.PhysicalGroup(crvs, 1, tag))
        for gp in per_pair_phy:
            gp.add_gmsh()

        self.gen_vect = rve_vect
        self.nb_cells = nb_cells
        self.attractors = attractors if attractors else []
        self.phy_surf = phy_surf
        self.mesh_fields = []
        self.mesh_abs_path = ""

    def main_mesh_refinement(self, d_min_max, lc_min_max, sigmoid_interpol=False):
        model.setCurrent(self.name)
        attractors = {"points": self.attractors}
        logger.debug(
            f"When main_mesh_refinement(...) is called, physical groups in model : {model.getPhysicalGroups()}"
        )
        rve_s = self.phy_surf[0].entities
        for s in rve_s:
            if not s.boundary:
                s.get_boundary()
        rve_boundary = list(flatten([s.boundary for s in rve_s]))
        restrict_domain = {"surfaces": rve_s, "curves": rve_boundary}
        field = msh.set_mesh_refinement(
            d_min_max, lc_min_max, attractors, 10, sigmoid_interpol, restrict_domain
        )
        self.mesh_fields.append(field)

    def soft_mesh_refinement(self, d_min_max, lc_min_max, sigmoid_interpol=False):
        model.setCurrent(self.name)
        attractors = {"points": self.attractors}
        soft_s = self.phy_surf[1].entities
        for s in soft_s:
            if not s.boundary:
                s.get_boundary()
        soft_boundary = list(flatten([s.boundary for s in soft_s]))
        restrict_domain = {"surfaces": soft_s, "curves": soft_boundary}
        field = msh.set_mesh_refinement(
            d_min_max, lc_min_max, attractors, 1, sigmoid_interpol, restrict_domain
        )
        self.mesh_fields.append(field)

    def mesh_generate(self, mesh_field=None, directory: Path = None):
        """Generate a 2D mesh of the model which represent a RVE.

        Parameters
        ----------
        mesh_field : mesh_tools.Field, optional
            The characteristic length of the elements can be explicitly prescribe by means of this field.
            The default is None. In this case, the fields that have been created with the soft_mesh_refinement and main_mesh_refinement methods are used.
        directory : pathlib.Path, optional
            Indicate in which directory the .msh file must be created.
            If None (default), the .msh file is written in the current working directory.
            Ex: Path('/media/sf_VM_share/homog')
        """

        model.setCurrent(self.name)
        if not mesh_field:
            self.background_field = msh.set_background_mesh(self.mesh_fields)
        else:
            self.background_field = msh.set_background_mesh(mesh_field)
        data = model.getPhysicalGroups()
        logger.debug(f"Physical groups in model just before generating mesh : {data}")
        geo.PhysicalGroup.set_group_mesh(True)
        model.mesh.generate(1)
        gmsh.model.mesh.removeDuplicateNodes()
        model.mesh.generate(2)
        gmsh.model.mesh.removeDuplicateNodes()
        geo.PhysicalGroup.set_group_visibility(False)
        if directory:
            mesh_path = directory if not directory.suffix else directory.with_suffix("")
            if not mesh_path.exists():
                mesh_path.mkdir(mode=0o777, parents=True)
        else:
            mesh_path = Path.cwd()
        mesh_path = mesh_path.joinpath(f"{self.name}.msh")
        gmsh.write(str(mesh_path))
        self.mesh_abs_path = mesh_path


class Gmsh2DPart(object):
    def __init__(self, gen_vect, nb_cells, phy_surf, mesh_path: PurePath):
        self.gen_vect = gen_vect
        self.nb_cells = nb_cells
        self.phy_surf = phy_surf
        self.mesh_abs_path = mesh_path.resolve()


from .pantograph import pantograph_RVE, pantograph_offset_RVE, beam_pantograph_RVE

# from .other_2d_microstructures import auxetic_square_RVE


def Gmsh2DPartFromRVE(cell: Gmsh2DRVE, nb_cells, part_name=None):
    """[summary]

    Parameters
    ----------
    rve : Gmsh2DRVE
        [description]
    nb_cells : tuple, dimension 2 or 3
        Number of cells in each direction.
    part_name: str, optional
        Desired name for the mesh file
    Returns
    -------
    tuple
        Paths to the RVE mesh and the part mesh

    Remarques
    ----------
    Pour le moment, le RVE est composé d'une unique cellule.
    Pour le moment, le domaine macro est un parallélogramme aligné avec les axes
    de la cellule et contient un nombre entier de cellules.
    """

    name = cell.name
    cell_vect = cell.gen_vect
    # * 2D -> 3D
    if cell_vect.shape != (3, 3):
        cell_vect_3D = np.zeros((3, 3))
        cell_vect_3D[: cell_vect.shape[0], : cell_vect.shape[1]] = cell_vect
        cell_vect = cell_vect_3D
    if len(nb_cells) != 3:
        nb_cells = tuple(padded(nb_cells, 1, 3))

    # TODO : Activer le model gmsh correspondant au RVE
    model.setCurrent(name)
    # TODO : Créer un domaine macro
    part_vect = cell_vect * np.asarray(nb_cells)[:, np.newaxis]
    macro_vertices = [
        np.zeros((3,)),
        part_vect[0],
        part_vect[0] + part_vect[1],
        part_vect[1],
    ]
    macro_lloop = geo.LineLoop([geo.Point(c) for c in macro_vertices])
    macro_surf = geo.PlaneSurface(macro_lloop)

    translat_vectors = list()
    for translat_combination in product(*[range(i) for i in nb_cells]):
        if translat_combination == (0, 0, 0):
            continue  # Correspond à la cellule de base
        # * cell_vect : vectors in column
        t_vect = np.dot(cell_vect, np.array(translat_combination))
        translat_vectors.append(t_vect)
    # ? Exemple :
    # ? >>> nb_cells = (2,3)
    # ? >>> nb_cells = tuple(padded(nb_cells,1,3))
    # ? >>> nb_cells
    # * (2, 3, 1)
    # ? >>> translat_vectors = list()
    # ? >>> cell_vect = np.array(((4.,1.,0.),(3.,8.,0.),(0.,0.,0.)))
    # ? >>> for translat_combination in product(*[range(i) for i in nb_cells]):
    # ?         t_vect = np.dot(cell_vect, np.array(translat_combination))
    # ?         translat_vectors.append(t_vect)
    # ? >>> translat_vectors
    # * [array([0., 0., 0.]), array([1., 8., 0.]), array([ 2., 16.,  0.]), array([4., 3., 0.]), array([ 5., 11.,  0.]), array([ 6., 19.,  0.])] #noqa
    cell_surfaces_by_gp = [phy_surf.entities for phy_surf in cell.phy_surf]
    repeated_surfaces_by_gp = [list() for i in range(len(cell.phy_surf))]
    # * Structure de repeated_surfaces_by_gp :
    # * List avec :
    # *     pour chaque physical group, et pour chaque translation
    # *          la liste des surfaces (entitées) translatées

    for i, gp_surfaces in enumerate(cell_surfaces_by_gp):
        for t_vect in translat_vectors:
            dimTags = factory.copy([(2, s.tag) for s in gp_surfaces])
            factory.translate(dimTags, *(t_vect.tolist()))
            # ? Opération booléenne d'intersection ?
            # ? Pour détecter si surface entière : comparaison de boundingbox surface d'origine et bounding box resultat - vecteur translation
            this_translation_surfs = [geo.AbstractSurface(dt[1]) for dt in dimTags]
            repeated_surfaces_by_gp[i].append(this_translation_surfs)
    factory.synchronize()
    # TODO : Contraintes de périodicité
    for j, t_vect in enumerate(translat_vectors):
        master = list(chain.from_iterable(cell_surfaces_by_gp))
        all_surfs_this_transl = [surfs[j] for surfs in repeated_surfaces_by_gp]
        slaves = list(chain.from_iterable(all_surfs_this_transl))
        msh.set_periodicity_pairs(slaves, master, t_vect)
    # TODO : Extension des physical groups
    phy_surfaces = list()
    for i in range(len(cell.phy_surf)):
        all_surfaces = cell_surfaces_by_gp[i] + list(
            flatten(repeated_surfaces_by_gp[i])
        )
        tag = cell.phy_surf[i].tag + 1000
        name = cell.phy_surf[i].name
        phy_surfaces.append(geo.PhysicalGroup(all_surfaces, 2, name, tag))
    # gmsh.fltk.run()
    for gp in cell.phy_surf:
        gp.remove_gmsh()
    factory.synchronize()
    for gp in phy_surfaces:
        gp.add_gmsh()
    # gmsh.fltk.run()
    # ! Pour le moment, il semble impossible de réutiliser le tag d'un physical group
    # ! qui a été supprimé.
    # ! Voir : \Experimental\Test_traction_oct19\pb_complet\run_3\MWE_reuse_tag.py
    # ! Autre solution :
    # ! - Compléter les physical group existants ?
    # !      Impossible car groups déjà ajoutés au model
    # ! Utiliser un autre tag, avec une règle pour relier les 2.
    # !     Solution retenue. Règle choisie : le tag pour la part = 1000 + tag pour la cell

    # TODO : All mesh generation
    geo.PhysicalGroup.set_group_mesh(True)
    model.mesh.generate(1)
    model.mesh.generate(2)
    model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()
    geo.PhysicalGroup.set_group_visibility(False)

    rve_path = cell.mesh_abs_path
    conversion = {
        "RVE": "PART",
        "rve": "part",
        "Rve": "Part",
        "cell": "part",
        "CELL": "PART",
        "Cell": "Part",
    }
    if part_name:
        part_path = rve_path.with_name(part_name).with_suffix(".msh")
    elif any(x in rve_path.name for x in conversion.keys()):
        name = rve_path.name
        for old, new in conversion.items():
            name = name.replace(old, new)
        part_path = rve_path.with_name(name).with_suffix(".msh")
    else:
        part_path = rve_path.with_name(rve_path.stem + "_part.msh")
    gmsh.write(str(part_path.with_suffix(".brep")))
    gmsh.write(str(part_path))
    return Gmsh2DPart(part_vect, nb_cells, phy_surfaces, part_path)


from . import pantograph

__all__ = [
    "pantograph_RVE",
    "pantograph_offset_RVE",
    "beam_pantograph_RVE",
    "pantograph_E11only_RVE",
    "auxetic_square_RVE",
    "Gmsh2DRVE",
    "Gmsh2DPart",
    "Gmsh2DPartFromRVE",
]


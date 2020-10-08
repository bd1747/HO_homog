# coding: utf8
"""
Created on 09/10/2018
@author: baptiste

Definition of a class designed to represent gmsh physical entities.
"""

from . import logger, factory, model, set_gmsh_option


class PhysicalGroup(object):
    """
    Create and manage groups of geometrical entities in the gmsh model.
    Physical groups can be visible in the exported mesh.

    "Groups of elementary geometrical entities can also be defined
    and are called “physical” groups.
    These physical groups cannot be modified by geometry commands: their only
    purpose is to assemble elementary entities into larger groups so that they can be
    referred to by the mesh module as single entities." (gmsh reference manual 26/09/2018)

    """

    all_groups = dict()

    def __init__(self, entities, geo_dim, name="", tag=0):
        """
        Gather multiple instances of one of the geometrical Python classes
        (Point, Curve, PlaneSurface) to form a single object in the gmsh model.

        Parameters
        ----------
        entities : list
            The instances that will compose the physical group.
            They must have the same geometrical dimension
            0 for points,
            1 for line, arcs, instances of AbstractCurve,
            2 for surfaces,
            3 for volumes
        geo_dim : int
            Geometrical dimension of the entities that are gathered.
        name : string, optional
            name of the group.
        tag : int
            Impose a tag.

        """
        try:
            _ = (item for item in entities)
        except TypeError:
            logger.debug("Physical group single entity -> Input conversion to list")
            entities = [entities]
        self.entities = entities
        self.dim = geo_dim
        self.name = name
        self.tag = None
        self.input_tag = tag if tag else None
        try:
            PhysicalGroup.all_groups[self.dim].append(self)
        except KeyError:
            PhysicalGroup.all_groups[self.dim] = [self]

    def add_gmsh(self):
        factory.synchronize()
        if self.tag:
            return self.tag
        tags = list()
        for item in self.entities:
            if not item.tag:
                item.add_gmsh()
            tags.append(item.tag)
        factory.synchronize()
        if self.input_tag:
            try:
                self.tag = model.addPhysicalGroup(self.dim, tags, self.input_tag)
            except ValueError:
                logger.warning(f"Tag {self.input_tag} already used for Physical groups")
                self.tag = model.addPhysicalGroup(self.dim, tags)
        else:
            self.tag = model.addPhysicalGroup(self.dim, tags)

        # ! TEMPORAIRE, un appel à synchronize devrait pouvoir être enlevé.
        logger.info(f"Physical group {self.tag} of dim {self.dim} added to gmsh")
        phy_before = model.getPhysicalGroups()
        factory.synchronize()
        phy_after = model.getPhysicalGroups()
        dbg_msg = (
            "Call of factory.synchronize(). Physical groups in the model \n"
            f"before : \n {phy_before} \n "
            f"and after : \n {phy_after}"
        )
        logger.debug(dbg_msg)
        factory.synchronize()
        logger.debug(
            f"And after a 2nd call to synchronize() : {model.getPhysicalGroups()}"
        )
        if self.name:
            model.setPhysicalName(self.dim, self.tag, self.name)

    def remove_gmsh(self):
        """
        Remove this physical group of the current model
        """
        model.removePhysicalGroups([(self.dim, self.tag)])
        self.tag = None

    def add_to_group(self, entities):
        """
        Add geometrical entities to an existing physical group.

        entitites :
            A geometrical entity or a list of geometrical entities.
        The appended items must be of the same geometrical dimension.
        """
        if self.tag:
            raise AttributeError(
                "The physical group has been already defined in the gmsh model."
                "It is too late to add entities to this group."
            )
        if isinstance(entities, list):
            self.entities += entities
        else:
            self.entities.append(entities)

    def set_color(self, rgba_color, recursive=False):
        """
        Choisir la couleur du maillage des éléments géométriques de l'entité physique.

       Parameters
       ----------
        rgba_color : list of 4 integers between 0 and 255.
            RGBA code of the desired color.
        recursive : bool, optional
            Apply the color setting to the parent geometrical entities as well.
        """
        dimtags = [(self.dim, e.tag) for e in self.entities]
        model.setVisibility(dimtags, 1)
        model.setColor(dimtags, *rgba_color, recursive=recursive)

    def set_visibility(self, val: bool, recursive: bool = False):
        """Show or hide entities that belong to the physical group."""
        dim_tags = [(self.dim, entity.tag) for entity in self.entities]
        visibility = 1 if bool(val) else 0
        model.setVisibility(dim_tags, visibility, bool(recursive))

    @classmethod
    def set_group_visibility(cls, val):
        """
        Make only entities that belong to at least one physical group visible,
        or make all geometrical entities visibles.

        Only physical groups that are active in the gmsh model are taken into account.

        Parameters
        ----------
        val : bool
            If True, only entities that to at least one physical group visible.
            If False, all geometrical entities will be visible.
        """
        if val:
            model.setVisibility(model.getEntities(), 0)
            dimtags = list()
            for gps in cls.all_groups.values():
                for gp in gps:
                    if gp.tag:
                        dimtags += [(gp.dim, ent.tag) for ent in gp.entities]
            model.setVisibility(dimtags, 1, recursive=True)
        else:
            model.setVisibility(model.getEntities(), 1)
        return None

    @classmethod
    def set_group_mesh(cls, val):
        """
        Mesh only the entities that belong to at least one physical group.

        Based on the EXPERIMENTAL MeshOnlyVisible option.

        Parameters
        ----------
        val : bool
            If True, only entities that to at least one physical group will be mesh.
            If False, a mesh will be generate for all geometrical entities.

        """
        if val:
            cls.set_group_visibility(1)
            set_gmsh_option("Mesh.MeshOnlyVisible", 1)
        else:
            cls.set_group_visibility(0)
            set_gmsh_option("Mesh.MeshOnlyVisible", 0)

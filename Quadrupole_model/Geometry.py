import gmsh
import os
from Data import Data

#Definition of classes to describe the geometry
class Cylinder:
    """
    Represents a cylindrical electrode in the GMSH geometry.
    """
    def __init__(self, length: int, radius: int, coord_x: int, coord_y: int, coord_z: int) -> None:
        """
        Initialize cylinder parameters.

        Args:
            length: Length of the cylinder along the Z-axis.
            radius: Radius of the cylinder.
            coord_x, coord_y, coord_z: Starting position of the cylinder.
        """
        self.length = length
        self.radius = radius
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z

        self.cyl_tag = None
        self.cyl_surf = None


    def add(self) -> None:
        """Add the cylinder to the OpenCASCADE model and extract its surface loop."""
        self.cyl_tag = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.length, self.radius)
        self.cyl_surf = gmsh.model.occ.get_surface_loops(self.cyl_tag)[1][0]


class Aperture: 
    """
    Represents an  aperture  in the GMSH geometry.
    """
    def __init__(self, radius_ext: int, radius_in: int, thickness: int, coord_x: int, coord_y: int, coord_z: int) -> None:
        """
        Initialize aperture parameters.

        Args:
            radius_ext: Outer radius of the aperture disc.
            radius_in: Inner radius (hole) of the aperture.
            thickness: Thickness of the disc.
        """
        self.radius_ext = radius_ext
        self.radius_in = radius_in
        self.thickness = thickness
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z

        self.apert_tag = None
        self.apert_surf = None


    def add (self) -> None:
        """Create the aperture """
        aperture_out = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.thickness, self.radius_ext)
        aperture_in = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.thickness, self.radius_in)
        apert_vol , _=  gmsh.model.occ.cut([(3,aperture_out)],[(3,aperture_in)])
        self.apert_tag=apert_vol[0][1]
        self.apert_surf = gmsh.model.occ.get_surface_loops(self.apert_tag)[1][0]

class Shield:
    """
    Represents the shield to fixe the potential ot 0
    """
    def __init__(self, length: int, radius_ext: int, radius_in: int, radius_hole: int, thickness: int, coord_x: int, coord_y: int, coord_z: int) -> None:
        self.length = length
        self.radius_ext = radius_ext
        self.radius_in = radius_in
        self.radius_hole = 3
        self.thickness = thickness
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z

        self.shield_tag = None
        self.shield_surf = None


    def add(self) -> None:
        shield_in = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z + self.thickness, 0, 0, self.length -2*self.thickness, self.radius_in)
        shield_out = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.length, self.radius_ext)
        shield , _=  gmsh.model.occ.cut([(3,shield_out)],[(3,shield_in)])
        shield_hole1 = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.length - self.thickness, 0, 0, self.thickness, self.radius_hole)
        shield_hole2 = gmsh.model.occ.addCylinder(self.coord_x, self.coord_y, self.coord_z, 0, 0, self.thickness, self.radius_hole)
        shield_vol_1, _=  gmsh.model.occ.cut([(3,shield[0][1])],[(3,shield_hole1)])
        shield_vol_2, _=  gmsh.model.occ.cut([(3,shield_vol_1[0][1])],[(3,shield_hole2)])
        self.shield_tag=shield_vol_2[0][1]
        self.shield_tag2=shield[0][1]

        self.shield_surf = gmsh.model.occ.get_surface_loops(self.shield_tag)[1][0]


class Mesh_Generation: 
    """
    Manages the full GMSH workflow: initialization, geometry construction, 
    physical grouping, and mesh generation.
    """
    def __init__(self, data: Data, visual: bool) -> None:
        """
        data (Data): Object containing all geometric dimensions and mesh settings.
            visual (bool): If True, launches the GMSH GUI after generation.
        """
        self.data = data
        self.visual = visual

        self.objects = None

    def initialisation(self) -> None:
        """Initialize GMSH and create a new model."""

        gmsh.initialize()
        gmsh.clear()
        gmsh.model.add("Quadrupole lens")


    def geometry(self) -> None:
        """Instantiate and add all geometric components (electrodes, apertures, shield)."""

        cylinder1 = Cylinder(self.data.length_cylinder, self.data.radius_axis, self.data.coord_cylinder_x_or_y, 0, self.data.coord_cylinder_z)
        Cylinder.add(cylinder1)

        cylinder2 = Cylinder(self.data.length_cylinder, self.data.radius_axis, -self.data.coord_cylinder_x_or_y, 0, self.data.coord_cylinder_z)
        Cylinder.add(cylinder2)

        cylinder3 = Cylinder(self.data.length_cylinder, self.data.radius_axis, 0, self.data.coord_cylinder_x_or_y, self.data.coord_cylinder_z)
        Cylinder.add(cylinder3)

        cylinder4 = Cylinder(self.data.length_cylinder, self.data.radius_axis, 0, -self.data.coord_cylinder_x_or_y, self.data.coord_cylinder_z)
        Cylinder.add(cylinder4)


        aperture1 = Aperture(self.data.radius_apert, self.data.radius_axis, self.data.thickness_apert, 0, 0, self.data.coord_apert_z1)
        Aperture.add(aperture1)

        aperture2 = Aperture(self.data.radius_apert, self.data.radius_axis, self.data.thickness_apert, 0, 0, self.data.coord_apert_z2)
        Aperture.add(aperture2)


        shield = Shield(self.data.total_length, self.data.radius_ext_shield, self.data.radius_in_shield, self.data.radius_axis, self.data.thickness_shield, 0, 0, 0)
        Shield.add(shield)

        self.objects = aperture1, aperture2, cylinder1, cylinder2, cylinder3, cylinder4, shield


    def creation_mesh(self) -> None:
        """Synchronize the OpenCASCADE internal CAD representation with the GMSH model."""
        gmsh.model.occ.synchronize()
    
    def surfaces(self) -> None:
        """
        Define physical groups for the surfaces. 
        """
        aperture1, aperture2, cylinder1, cylinder2, cylinder3, cylinder4, shield = self.objects

    #Outwards orientation of the surfaces' normals
        gmsh.model.mesh.setOutwardOrientation(aperture1.apert_tag)
        gmsh.model.mesh.setOutwardOrientation(aperture2.apert_tag)
        gmsh.model.mesh.setOutwardOrientation(cylinder1.cyl_tag)
        gmsh.model.mesh.setOutwardOrientation(cylinder2.cyl_tag)
        gmsh.model.mesh.setOutwardOrientation(cylinder3.cyl_tag)
        gmsh.model.mesh.setOutwardOrientation(cylinder4.cyl_tag)
        gmsh.model.mesh.setOutwardOrientation(shield.shield_tag)

        #Adds surfaces on top of the volumes created 
        group_id_apert1 = gmsh.model.addPhysicalGroup(2, aperture1.apert_surf)
        group_id_apert2 = gmsh.model.addPhysicalGroup(2, aperture2.apert_surf)
        group_id_cyl1 = gmsh.model.addPhysicalGroup(2, cylinder1.cyl_surf)
        group_id_cyl2 = gmsh.model.addPhysicalGroup(2, cylinder2.cyl_surf)
        group_id_cyl3 = gmsh.model.addPhysicalGroup(2, cylinder3.cyl_surf)
        group_id_cyl4 = gmsh.model.addPhysicalGroup(2, cylinder4.cyl_surf)
        group_id_shield = gmsh.model.addPhysicalGroup(2, shield.shield_surf)

        self.data.group_id = group_id_apert1, group_id_apert2, group_id_cyl1, group_id_cyl2, group_id_cyl3, group_id_cyl4, group_id_shield


    def mesh(self) -> None:
        """Configure mesh size  and generate the  surface mesh."""
    #Size of the mesh
    #ATTENTION, MeshSizeMin and MeshSizeMax need to be equal
        gmsh.option.set_number('Mesh.MeshSizeMin', self.data.MeshSizeMin)
        gmsh.option.set_number('Mesh.MeshSizeMax', self.data.MeshSizeMax)
        gmsh.option.set_number('Mesh.MeshSizeFromCurvature', self.data.MeshSizeFromCurvature)

        gmsh.model.mesh.generate(2)

    def finalize(self) -> None:
    #Creates a file .msh
        mesh_path = os.path.join(self.data.output_dir, "mesh_quadrupole.msh")

        gmsh.write(mesh_path)
        print("Mesh saved to:", mesh_path)

        if self.visual == True:
            #Opens a terminal to see the geometry 
            gmsh.fltk.run()

        gmsh.finalize()

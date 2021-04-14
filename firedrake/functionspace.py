"""
This module implements the user-visible API for constructing
:class:`.FunctionSpace` and :class:`.MixedFunctionSpace` objects.  The
API is functional, rather than object-based, to allow for simple
backwards-compatibility, argument checking, and dispatch.
"""
import ufl

from pyop2.utils import flatten
from pyop2.profiling import timed_function

from firedrake import functionspaceimpl as impl


__all__ = ("CreateMixedFunctionSpace", "CreateFunctionSpace",
           "CreateVectorFunctionSpace", "CreateTensorFunctionSpace")

class CreateFunctionSpace:
    def __init__(self, mesh, family=None, degree=None, name=None, vfamily=None, vdegree=None):
        self.mesh = mesh
        self.family = family
        self.degree = degree
        self.name = name
        self.vfamily = vfamily
        self.vdegree = vdegree
        self.get_element()
        self.create_function_space()

        return self.function_space

    def create_element(self):
        self.mesh.init()
        self.topology = self.mesh.toplogoy
        self.cell = self.topology.ufl_cell()
        self.element = ufl.FiniteElement(self.family, cell=self.cell, degree=self.degree)

    def check_element(self, top=True):
        if type(self.element) in (ufl.BrokenElement, ufl.FacetElement,
                             ufl.InteriorElement, ufl.RestrictedElement,
                             ufl.HDivElement, ufl.HCurlElement):
            inner = (self.element._element, )
        elif type(self.element) is ufl.EnrichedElement:
            inner = self.element._elements
        elif type(self.element) is ufl.TensorProductElement:
            inner = self.element.sub_elements()
        elif isinstance(self.element, ufl.MixedElement):
            if not top:
                raise ValueError("%s modifier must be outermost" % type(self.element))
            else:
                inner = self.element.sub_elements()
        else:
            return
        for e in inner:
            check_element(e, top=False)

    def build_function_space(self):
        if self.element.family() == "Real":
            self.function_space = impl.RealFunctionSpace(self.topology, self.element, name=self.name)
        else:
            self.function_space = impl.FunctionSpace(self.topology, self.element, name=self.name)
        if mesh is not self.topology:
            self.function_space = impl.WithGeometry(self.function_space, self.mesh)

    def get_element(self):
        self.element = self.create_element(self.mesh, self.family, self.degree, self.vfamily, self.vdegree)



class CreateVectorFunctionSpace(CreateFunctionSpace):

    def __init__(self, mesh, family, degree=None, dim=None, name=None,
                 vfamily=None, vdegree=None):
        super().__init__(mesh, family, degree, name, vfamily, vdegree)
        self.dim = self.dim or self.mesh.ufl_cell().geometric_dimension()

    def get_element(self):
        sub_element = self.create_element()
        self.element = ufl.VectorElement(sub_element, dim=dim)
        # Check that any Vector/Tensor/Mixed modifiers are outermost.
        self.check_element()
        self.element.reconstruct(cell=self.cell)

class CreateTensorFunctionSpace(CreateFunctionSpace):

    def __init__(self, mesh, family, degree=None, shape=None, symmetry=None,
                 name=None, vfamily=None, vdegree=None):
        super().__init__(mesh, family, degree, name, vfamily, vdegree)
        self.shape = shape
        self.symmetry = symmetry

    def create_element(self):
        self.mesh.init()
        self.topology = mesh.topology
        self.cell = self.topology.ufl_cell()
        if isinstance(self.cell, ufl.TensorProductCell) \
           and self.vfamily is not None and self.vdegree is not None:
            la = ufl.FiniteElement(self.family, cell=cell.sub_cells()[0], degree=self.degree)
            # If second element was passed in, use it
            lb = ufl.FiniteElement(self.vfamily, cell=ufl.interval, degree=self.vdegree)
            # Now make the TensorProductElement
            self.element = ufl.TensorProductElement(la, lb)
        else:
            self.element = ufl.FiniteElement(self.family, cell=cell, degree=self.degree)


    def get_element(self):
        sub_element = self.create_element()
        self.shape = self.shape or (mesh.ufl_cell().geometric_dimension(),) * 2
        self.element = ufl.TensorElement(sub_element, shape=self.shape, symmetry=self.symmetry)
        # Check that any Vector/Tensor/Mixed modifiers are outermost.
        self.check_element()
        self.element.reconstruct(cell=self.cell)

class CreateMixedFunctionSpace(CreateFunctionSpace):

    def __init__(self, spaces, name=None, mesh=None):
        super().__init__(mesh, name=name)
        self.spaces = spaces


    def build_spaces(self, sub_elements):
        self.mesh.init()
        self.topology = self.mesh.topology
        cell = topology.ufl_cell()
        element = family.reconstruct(cell=self.cell)
        return MixedFunctionSpace(element, mesh=self.mesh, name=self.name)

    def get_element(self):
        if isinstance(self.spaces, ufl.FiniteElementBase):
            # Build the spaces if we got a mixed element
            assert type(self.spaces) is ufl.MixedElement and self.mesh is not None
            sub_elements = []

            def rec(eles):
                for ele in eles:
                    # Only want to recurse into MixedElements
                    if type(ele) is ufl.MixedElement:
                        rec(ele.sub_elements())
                    else:
                        sub_elements.append(ele)
            rec(self.spaces.sub_elements())
            self.spaces = [self.build_spaces(element) for element in sub_elements]

        # Check that function spaces are on the same mesh
        meshes = [space.mesh() for space in self.spaces]
        for i in range(1, len(meshes)):
            if meshes[i] is not meshes[0]:
                raise ValueError("All function spaces must be defined on the same mesh!")

        # Select mesh
        self.mesh = meshes[0]
        # Get topological spaces
        self.spaces = tuple(s.topological for s in flatten(self.spaces))
        # Error checking
        for space in self.spaces:
            if type(space) in (impl.FunctionSpace, impl.RealFunctionSpace):
                continue
            elif type(space) is impl.ProxyFunctionSpace:
                if space.component is not None:
                    raise ValueError("Can't make mixed space with %s" % space)
                continue
            else:
                raise ValueError("Can't make mixed space with %s" % type(space))

    def build_function_space(self):
        self.functionspace = impl.MixedFunctionSpace(self.spaces, name=name)
        if self.mesh is not self.topology:
            self.functionspace = impl.WithGeometry(self.functionspace, self.mesh)

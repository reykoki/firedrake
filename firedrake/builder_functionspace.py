"""
TEST
This module implements the user-visible API for constructing
:class:`.FunctionSpace` and :class:`.MixedFunctionSpace` objects.  The
API is functional, rather than object-based, to allow for simple
backwards-compatibility, argument checking, and dispatch.
"""
import ufl

from pyop2.utils import flatten
from pyop2.profiling import timed_function

from firedrake import functionspaceimpl as impl

print('\nbuilder\n')

__all__ = ("MixedFunctionSpace", "FunctionSpace",
           "VectorFunctionSpace", "TensorFunctionSpace")

def FunctionSpace(mesh, family, degree=None, name=None, vfamily=None,
                  vdegree=None):

    function_space_obj = FunctionSpaceBuilder(mesh, family=family, degree=degree,
                                              name=name, vfamily=vfamily,
                                              vdegree=vdegree).get_element().build()
    return function_space_obj.function_space

def VectorFunctionSpace(mesh, family, degree=None, dim=None,
                        name=None, vfamily=None, vdegree=None):

    function_space_obj = FunctionSpaceBuilder(mesh, family=family, degree=degree,
                                              dim=dim, name=name, vfamily=vfamily,
                                              vdegree=vdegree).get_vector_element().build()
    return function_space_obj.function_space

def TensorFunctionSpace(mesh, family, degree=None, name=None, vfamily=None, vdegree=None):
    pass
def MixedFunctionSpace(mesh, family, degree=None, name=None, vfamily=None, vdegree=None):
    pass


class functionSpace:

    def __init__(self, builder):
        self.build_function_space(builder)

    def build_function_space(self, builder):
        if builder.element.family() == "Real":
            self.function_space = impl.RealFunctionSpace(builder.topology, builder.element, name=builder.name)
        else:
            self.function_space = impl.FunctionSpace(builder.topology, builder.element, name=builder.name)
        if builder.mesh is not builder.topology:
            self.function_space = impl.WithGeometry(self.function_space, builder.mesh)

class FunctionSpaceBuilder:

    def __init__(self, mesh, family=None, dim=None, shape=None, degree=None, symmetry=None, name=None, vfamily=None, vdegree=None):
        self.mesh = mesh
        self.family = family
        self.dim = dim
        self.shape = shape
        self.degree = degree
        self.name = name
        self.vfamily = vfamily
        self.vdegree = vdegree

    def build(self):
        return functionSpace(self)

    def create_element(self):
        self.mesh.init()
        self.topology = self.mesh.topology
        self.cell = self.topology.ufl_cell()
        return ufl.FiniteElement(self.family, cell=self.cell, degree=self.degree)

    def check_element(self, el, top=True):
        if type(el) in (ufl.BrokenElement, ufl.FacetElement,
                             ufl.InteriorElement, ufl.RestrictedElement,
                             ufl.HDivElement, ufl.HCurlElement):
            inner = (el._element, )
        elif type(el) is ufl.EnrichedElement:
            inner = el._elements
        elif type(el) is ufl.TensorProductElement:
            inner = el.sub_elements()
        elif isinstance(el, ufl.MixedElement):
            if not top:
                raise ValueError("%s modifier must be outermost" % type(el))
            else:
                inner = el.sub_elements()
        else:
            return
        for e in inner:
            self.check_element(e, top=False)


    def get_element(self):
        self.element = self.create_element()
        return self

    def get_vector_element(self):
        sub_element = self.create_element()
        self.element = ufl.VectorElement(sub_element, dim=self.dim)
        # Check that any Vector/Tensor/Mixed modifiers are outermost.
        self.check_element(self.element)
        self.element.reconstruct(cell=self.cell)
        return self

    def create_tensor_element(self):
        self.mesh.init()
        self.topology = self.mesh.topology
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

    def get_tensor_element(self):
        sub_element = self.create_tensor_element()
        self.shape = self.shape or (self.mesh.ufl_cell().geometric_dimension(),) * 2
        self.element = ufl.TensorElement(sub_element, shape=self.shape, symmetry=self.symmetry)
        # Check that any Vector/Tensor/Mixed modifiers are outermost.
        self.check_element(self.element)
        self.element.reconstruct(cell=self.cell)
        return self


class CreateMixedFunctionSpace():

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



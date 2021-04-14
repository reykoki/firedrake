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


__all__ = ("MixedFunctionSpace", "FunctionSpace",
           "VectorFunctionSpace", "TensorFunctionSpace")

class CreateFunctionSpaceBase:
    def __init__(self, mesh, family, degree=None, name=None, vfamily=None, vdegree=None):
        self.mesh = mesh
        self.family = family
        self.degree = degree
        self.name = name
        self.vfamily = vfamily
        self.vdegree = vdegree
        self.get_element()
        self.create_function_space()

        return self.function_space

    def make_scalar_element(self):
        self.mesh.init()
        topology = self.mesh.toplogoy
        cell = topology.ufl_cell()
        self.element = ufl.FiniteElement(family, cell=cell, degree=degree)

    def check_element(self, element, top=True):
        pass

    def build_function_space(self):
        topology = self.mesh.topology
        if element.family() == "Real":
            self.function_space = impl.RealFunctionSpace(topology, element, name=name)
        else:
            self.function_space = impl.FunctionSpace(topology, element, name=name)
        if mesh is not topology:
            self.function_space = impl.WithGeometry(self.function_space, self.mesh)

    def get_element(self):
        self.element = self.make_scalar_element(self.mesh, self.family, self.degree, self.vfamily, self.vdegree)
        # Check that any Vector/Tensor/Mixed modifiers are outermost.

class CreateScalarFunctionSpace(CreateFunctionSpaceBase):

    def __init__(self, mesh, family, degree=None, name=None, vfamily=None, vdegree=None):
        super().__init__(mesh, family, degree, name, vfamily, vdegree)


def CreateVectorFunctionSpace(CreateFunctionSpaceBase):
    def __init__(self, mesh, family, degree=None, dim=None, name=None, vfamily=None, vdegree=None):
        super().__init__(mesh, family, degree, name, vfamily, vdegree)
        self.dim = dim

    def check_element(element, top=True):
        if type(element) in (ufl.BrokenElement, ufl.FacetElement,
                             ufl.InteriorElement, ufl.RestrictedElement,
                             ufl.HDivElement, ufl.HCurlElement):
            inner = (element._element, )
        elif type(element) is ufl.EnrichedElement:
            inner = element._elements
        elif type(element) is ufl.TensorProductElement:
            inner = element.sub_elements()
        elif isinstance(element, ufl.MixedElement):
            if not top:
                raise ValueError("%s modifier must be outermost" % type(element))
            else:
                inner = element.sub_elements()
        else:
            return
        for e in inner:
            check_element(e, top=False)


    def get_element(self):
        sub_element = self.make_scalar_element()
        dim = self.dim or self.mesh.ufl_cell().geometric_dimension()
        self.element = ufl.VectorElement(sub_element, dim=dim)
        self.check_element()



def TensorFunctionSpace(mesh, family, degree=None, shape=None,
                        symmetry=None, name=None, vfamily=None,
                        vdegree=None):
    """Create a rank-2 :class:`.FunctionSpace`.

    :arg mesh: The mesh to determine the cell from.
    :arg family: The finite element family.
    :arg degree: The degree of the finite element.
    :arg shape: An optional shape for the tensor-valued degrees of
       freedom at each function space node (defaults to a square
       tensor using the geometric dimension of the mesh).
    :arg symmetry: Optional symmetries in the tensor value.
    :arg name: An optional name for the function space.
    :arg vfamily: The finite element in the vertical dimension
        (extruded meshes only).
    :arg vdegree: The degree of the element in the vertical dimension
        (extruded meshes only).

    The ``family`` argument may be an existing
    :class:`~ufl.classes.FiniteElementBase`, in which case all other arguments
    are ignored and the appropriate :class:`.FunctionSpace` is
    returned.  In this case, the provided element must have an empty
    :meth:`~ufl.classes.FiniteElementBase.value_shape`.

    .. note::

       The element that you provide must be a scalar element (with
       empty ``value_shape``).  If you already have an existing
       :class:`~ufl.classes.TensorElement`, you should pass it to
       :func:`FunctionSpace` directly instead.
    """
    sub_element = make_scalar_element(mesh, family, degree, vfamily, vdegree)
    shape = shape or (mesh.ufl_cell().geometric_dimension(),) * 2
    element = ufl.TensorElement(sub_element, shape=shape, symmetry=symmetry)
    return FunctionSpace(mesh, element, name=name)


def MixedFunctionSpace(spaces, name=None, mesh=None):
    """Create a :class:`.MixedFunctionSpace`.

    :arg spaces: An iterable of constituent spaces, or a
        :class:`~ufl.classes.MixedElement`.
    :arg name: An optional name for the mixed function space.
    :arg mesh: An optional mesh.  Must be provided if spaces is a
        :class:`~ufl.classes.MixedElement`, ignored otherwise.
    """
    if isinstance(spaces, ufl.FiniteElementBase):
        # Build the spaces if we got a mixed element
        assert type(spaces) is ufl.MixedElement and mesh is not None
        sub_elements = []

        def rec(eles):
            for ele in eles:
                # Only want to recurse into MixedElements
                if type(ele) is ufl.MixedElement:
                    rec(ele.sub_elements())
                else:
                    sub_elements.append(ele)
        rec(spaces.sub_elements())
        spaces = [FunctionSpace(mesh, element) for element in sub_elements]

    # Check that function spaces are on the same mesh
    meshes = [space.mesh() for space in spaces]
    for i in range(1, len(meshes)):
        if meshes[i] is not meshes[0]:
            raise ValueError("All function spaces must be defined on the same mesh!")

    # Select mesh
    mesh = meshes[0]
    # Get topological spaces
    spaces = tuple(s.topological for s in flatten(spaces))
    # Error checking
    for space in spaces:
        if type(space) in (impl.FunctionSpace, impl.RealFunctionSpace):
            continue
        elif type(space) is impl.ProxyFunctionSpace:
            if space.component is not None:
                raise ValueError("Can't make mixed space with %s" % space)
            continue
        else:
            raise ValueError("Can't make mixed space with %s" % type(space))

    new = impl.MixedFunctionSpace(spaces, name=name)
    if mesh is not mesh.topology:
        return impl.WithGeometry(new, mesh)
    return new

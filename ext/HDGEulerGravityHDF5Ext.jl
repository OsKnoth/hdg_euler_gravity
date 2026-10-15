"""
VTKHDF output (UnstructuredGrid, transient) for Euler2D, written with HDF5.jl.

Layout (VTKHDF ≥ 2.0, ParaView ≥ 5.13): the mesh is stored once; every call of
`write_vtkhdf!` appends the point data of one time step and an entry in /VTKHDF/Steps whose
geometry offsets all point to the single stored mesh. Each element is one
VTK_LAGRANGE_QUADRILATERAL of order p with the DG polynomial resampled to equispaced nodes
(exact), so ParaView sees the high-order solution and the curved terrain-following elements.
Points are (x, z, 0).
"""
module HDGEulerGravityHDF5Ext

using HDF5
using HDGEulerGravity.Euler2D: Euler2DModel, output_fields
using HDGEulerGravity.HDGRelaxTheta: lagrange_basis
import HDGEulerGravity.Euler2D: open_vtkhdf, write_vtkhdf!

const VTK_LAGRANGE_QUADRILATERAL = UInt8(70)

mutable struct VTKHDFFile
    file::HDF5.File
    npts::Int
    nsteps::Int
    V::Matrix{Float64}          # GLL → equispaced resampling (np × np)
end

"""VTK point index (0-based) of node (i, j) ∈ 0:n × 0:n of a Lagrange quadrilateral of order n
(vtkHigherOrderQuadrilateral::PointIndexFromIJK)."""
function vtk_lagrange_index(i, j, n)
    ib = i == 0 || i == n; jb = j == 0 || j == n
    ib && jb && return i == 0 ? (j == 0 ? 0 : 3) : (j == 0 ? 1 : 2)
    off = 4
    if jb                               # edge along i
        return off + (i - 1) + (j == 0 ? 0 : 2 * (n - 1))
    elseif ib                           # edge along j
        return off + (j - 1) + (i == 0 ? 3 * (n - 1) : n - 1)
    end
    return off + 4 * (n - 1) + (i - 1) + (n - 1) * (j - 1)
end

"""Fixed-length ASCII string attribute (VTK does not read HDF5's default UTF-8 strings)."""
function ascii_attribute(obj, name, s::String)
    dt = HDF5.Datatype(HDF5.API.h5t_copy(HDF5.API.H5T_C_S1))
    HDF5.API.h5t_set_size(dt, ncodeunits(s))
    HDF5.API.h5t_set_cset(dt, HDF5.API.H5T_CSET_ASCII)
    HDF5.API.h5t_set_strpad(dt, HDF5.API.H5T_STR_NULLPAD)
    attr = create_attribute(obj, name, dt, dataspace(()))
    HDF5.API.h5a_write(attr, dt, Vector{UInt8}(codeunits(s)))
    close(attr); close(dt)
end

extendable(g, name, T, chunk) = create_dataset(g, name, T, ((0,), (-1,)); chunk = (chunk,))

function extend!(d::HDF5.Dataset, v::AbstractVector)
    n = length(d)
    HDF5.set_extent_dims(d, (n + length(v),))
    d[n+1:n+length(v)] = v
end

"""Resample a nodal element field (np × np) to equispaced nodes, flattened in VTK order."""
function element_values(V, F, perm)
    G = V * F * V'
    out = similar(G, length(G))
    for j in axes(G, 2), i in axes(G, 1)
        out[perm[i, j]] = G[i, j]
    end
    return out
end

function open_vtkhdf(filename::AbstractString, m::Euler2DModel)
    M = m.mesh; np, n, Ne = M.np, M.p, M.Ne
    V, _ = lagrange_basis(M.xg, collect(range(-1, 1; length = np)))
    perm = [vtk_lagrange_index(i - 1, j - 1, n) + 1 for i in 1:np, j in 1:np]
    npe = np^2; npts = npe * Ne
    pts = zeros(3, npts)
    for e in 1:Ne
        r = (e-1)*npe+1:e*npe
        pts[1, r] = element_values(V, M.X[:, :, e], perm)
        pts[2, r] = element_values(V, M.Z[:, :, e], perm)
    end
    f = h5open(filename, "w")
    g = create_group(f, "VTKHDF")
    attributes(g)["Version"] = Int64[2, 0]
    ascii_attribute(g, "Type", "UnstructuredGrid")
    g["NumberOfPoints"] = Int64[npts]
    g["NumberOfCells"] = Int64[Ne]
    g["NumberOfConnectivityIds"] = Int64[npts]
    g["Points"] = pts
    g["Offsets"] = Int64.(collect(0:npe:npts))
    g["Connectivity"] = Int64.(collect(0:npts-1))
    g["Types"] = fill(VTK_LAGRANGE_QUADRILATERAL, Ne)
    pd = create_group(g, "PointData")
    st = create_group(g, "Steps")
    attributes(st)["NSteps"] = 0
    extendable(st, "Values", Float64, 64)
    for name in ("PartOffsets", "NumberOfParts", "PointOffsets", "CellOffsets", "ConnectivityIdOffsets")
        extendable(st, name, Int64, 64)
    end
    pdo = create_group(st, "PointDataOffsets")
    for (name, _) in output_fields(m, zeros(m.nq) .+ 1)
        extendable(pd, name, Float64, npts)
        extendable(pdo, name, Int64, 64)
    end
    flush(f)
    return VTKHDFFile(f, npts, 0, V)
end

function write_vtkhdf!(vtk::VTKHDFFile, m::Euler2DModel, q, t::Real)
    M = m.mesh; np, npe = M.np, M.np^2
    perm = [vtk_lagrange_index(i - 1, j - 1, M.p) + 1 for i in 1:np, j in 1:np]
    g = vtk.file["VTKHDF"]
    pd = g["PointData"]; st = g["Steps"]; pdo = st["PointDataOffsets"]
    for (name, F) in output_fields(m, q)
        vals = Vector{Float64}(undef, vtk.npts)
        for e in 1:M.Ne
            vals[(e-1)*npe+1:e*npe] = element_values(vtk.V, F[:, :, e], perm)
        end
        extend!(pdo[name], Int64[vtk.nsteps * vtk.npts])
        extend!(pd[name], vals)
    end
    extend!(st["Values"], [Float64(t)])
    extend!(st["PartOffsets"], Int64[0])
    extend!(st["NumberOfParts"], Int64[1])
    for name in ("PointOffsets", "CellOffsets", "ConnectivityIdOffsets")
        extend!(st[name], Int64[0])
    end
    vtk.nsteps += 1
    delete_attribute(st, "NSteps")
    attributes(st)["NSteps"] = vtk.nsteps
    flush(vtk.file)
    return vtk
end

Base.close(vtk::VTKHDFFile) = close(vtk.file)

end # module

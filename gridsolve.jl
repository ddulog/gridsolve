module GridSolve

export span_evol_grid
export g_evol

using DifferentialEquations: ODEProblem, solve
using LinearAlgebra: I

struct Grid
    t_array::Vector{Float64}
    u_array::Array{ComplexF64, 3}
    dim::Int64
end

"""
    span_evol_grid(period, ode_input, ode_func, grid_points, dim)

Create a grid of evaluation points along the interval [0, `period`]. Returns a time-array
and an array of evolution operator matrices.

# Arguments
- `period::Float64`: Period of the Hamiltonian.
- `ode_input`: Parameters for the ODE solver.
- `ode_func::Function`: Function that resembles the time-dependent Liouvillian
- `grid_points::Int64`: Number of evaluation points along the inteval.
- `dim::Int64`: Dimension of the Hilbert space.
"""
function span_evol_grid(period::Float64, ode_input, ode_func::Function, 
                        grid_points::Int64, dim::Int64)
    println("Selected number of grid points: $(grid_points)")

    t_eval = Vector{Float64}(range(start=0.0, stop=period, length=grid_points))
    t_grid, u_grid = integrate(0.0, period, t_eval, ode_input, ode_func, dim)

    # From Vector{Vector{Complex64}} to Matrix{ComplexF64}
    u_grid = reduce(hcat, u_grid)

    # Reshape to sequence of propper matrices
    u_grid = reshape(u_grid, (dim, dim, :))
    printstyled("GRID CREATION FINISHED (with $(grid_points) points).\n", 
                color=:default, underline=false, bold=true)

    return Grid(t_grid, u_grid, dim)
end

"""
    g_evol(t_start, t_end, grid)

Retrun the evolution operator from `t_start` to `t_end` by using the grid.
"""
function g_evol(t_start, t_end, grid)
    # Period of grid
    t_p = maximum(grid.t_array)
    # Find evol time to origin of grid
    n_1 = Int64(fld(t_start, t_p))  # floored division
    t_s2o = mod(t_start, t_p)
    # Find evol from origin to end
    n_2 = Int64(fld(t_end, t_p))
    t_o2e = mod(t_end, t_p)

    u_s2o = find_u(t_s2o, grid)
    u_o2e = find_u(t_o2e, grid)
    u_period = grid.u_array[:, :, lastindex(grid.u_array, 3)]

    return u_o2e * u_period^(n_2 - n_1) * adjoint(u_s2o)
end

"""
    find_u(t, grid)

Find the evolution operator on `grid` that corresponds to evolution time `t`.
"""
function find_u(t, grid)
    return grid.u_array[:, :, find_closest(t, grid.t_array)]
end

"""
    find_closest(element, array)

Return the index of that entry of `array` which is closest to `element`.
"""
function find_closest(element, array)
    i = searchsortedlast(array, element)

    # Edge cases
    i == 0 && return 1
    i == lastindex(array) && return lastindex(array)

    # Find shortest distance
    a = abs(array[i] - element)
    b = abs(array[i + 1] - element)
    return a < b ? i : i + 1
end

"""
    integrate(t_start, t_end, t_eval, ode_input, func, dim)

Solves the ODE with Liouvillian `func` on Hilbert space with dimension `dim` for the 
interval (t_start, t_end).
"""
function integrate(t_start::Float64, t_end::Float64, t_eval::Vector{Float64}, 
                   ode_input, func::Function, dim::Int64;
                   abstol=1e-8, reltol=1e-8)
    prob = ODEProblem(func, Matrix{ComplexF64}(I(dim)), (t_start, t_end), ode_input)
    sol = solve(prob, saveat=t_eval, abstol=abstol, reltol=reltol)

    return (sol.t, sol.u)
end

end # module
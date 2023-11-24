using Images
using ImageFiltering
using Random
using Plots
using LinearAlgebra
using Colors
using Dates
using DelimitedFiles
using Base.Threads: @threads, @spawn, @sync
# Assuming load and other required functions are from these packages
using FileIO
using CSV
using DataStructures

function average_mass_placements(rows::Int64, cols::Int64, num_routers::Int64)::Matrix{Int32}
    if num_routers == 1
        c  = Int32(round(cols/2))
        r  = Int32(round(rows/2))
        return Matrix{Int32}([r c])
    end
    placements = Matrix{Int32}(undef, num_routers, 2)
    elipse_matrix = [cols/4 0; 0 rows/4]
    centering_matrix = [cols/2; rows/2]
    angle_between_routers = (2*pi)/num_routers
    for i in 1:num_routers
        x = cos(i*angle_between_routers)
        y = sin(i*angle_between_routers)
        coords = elipse_matrix*[x; y] + centering_matrix
        new_x = Int32(round(coords[1]))
        new_y = Int32(round(coords[2]))
        placements[i, 1] = new_y
        placements[i, 2] = new_x
    end
    return placements
end

#Another benchmark can just be to 'eye-ball' it and see what a human's best guess would be


function peak_placements(matrix::Matrix{Float64}, num_routers)
    r, c = size(matrix)
    heap = BinaryMinHeap{Vector{Float64}}() #insert a min heap of the form [value, row, col]
    push!(heap, [-Inf, -1, -1])
    for i in 1:r
        for j in 1:c
            min_element_heap = first(heap)#get min element from heap
            pixel_value = matrix[i,j]
            if length(heap) < num_routers
                push!(heap, [pixel_value, i, j])

            elseif pixel_value > min_element_heap[1]
                pop!(heap)#pop min element from heap
                push!(heap, [pixel_value, i, j])#insert [pixel_value, i,j] into heap
            end
        end
    end
    placements = Matrix{Int32}(undef, num_routers, 2)
    for i in 1:length(heap)
        heap_element = pop!(heap)
        placements[i, 1] = heap_element[2]
        placements[i, 2] = heap_element[3]
    end
    return placements
end



function plot_average_mass(xdim, ydim, num)
    l = average_mass_placements(ydim, xdim, num)
    k, t = size(l)
    x = []
    y = []
    for i in 1:k
        to_x = l[i, 2]
        to_y = l[i, 1]
        push!(x, to_x)
        push!(y , to_y)
    end
    Plots.plot(x, y, seriestype=:scatter, xlims = (0, xdim), ylims = (0,ydim), label = "")
end



# plot_average_mass(2000,1000, 12)
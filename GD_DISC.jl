using Images
using ImageFiltering
using Random
using Plots
using LinearAlgebra
using Colors
using Dates
using Makie
using GLMakie
GLMakie.activate!()
using DelimitedFiles


function image_to_grayscale(image_path::String; blur_size::Int=5, blur_sigma::Float64=1.5)
    # Load the image in grayscale mode
    img = Gray.(load(image_path))
    
    # Invert the grayscale image colors
    img = 1.0 .- img
    
    # Apply Gaussian blur to smooth the image
    img_smooth = imfilter(img, Kernel.gaussian((blur_sigma, blur_sigma)))
    
    return img_smooth
end

function loss_function_squared_distance(matrix, placements)
    m, n = size(matrix)
    k, _ = size(placements)

    # Create coordinate grid for matrix
    x, y = meshgrid(1:m, 1:n)
    coords = hcat(x[:], y[:])

    # Compute squared distances between each point in the matrix and each placement
    squared_distances = [sum((c .- p).^2) for c in eachrow(coords), p in eachrow(placements)]

    # Avoid division by zero by adding a small value
    epsilon = 1e-10
    distances = sqrt.(1.0 ./ (squared_distances .+ epsilon))

    # Handle the case where (u, v) == (r, c)
    distances[squared_distances .< epsilon] .= 2.0

    # Compute the loss
    f = sum(matrix .* distances)

    return f
end

function loss_function(matrix, placements)
    m, n = size(matrix)
    k = size(placements, 1)
    
    # Preallocate arrays
    distances = Array{Float64, 3}(undef, m, n, k)
    weights = Array{Float64, 2}(undef, m, n)
    
    # Reshape the placements vectors to be broadcast-compatible
    placements_rows = reshape(placements[:, 1], 1, 1, k)
    placements_cols = reshape(placements[:, 2], 1, 1, k)
    
    epsilon = 1

    # Parallelize the computation of distances using threads
    @Threads.threads for i in 1:m
        for j in 1:n
            for l in 1:k
                dist = sqrt((placements_rows[1, 1, l] - i)^2 + (placements_cols[1, 1, l] - j)^2)
                distances[i, j, l] = dist < epsilon ? dist + epsilon : dist
            end
            # Calculate the weight based on the nearest placement
            weights[i, j] = 1.0 / minimum(view(distances, i, j, :))
        end
    end
    
    return sum(matrix .* weights)
end

function new_placement(loss_function, matrix, placements; multiplier::Int=1)
    m, n = size(matrix)
    k, _ = size(placements)
    changes = zeros(Int, k, 2)
    
    current_val = loss_function(matrix, placements)
    
    Threads.@threads for i in 1:k
        for j in 1:2
            cur_max = current_val
            
            # Try decrementing the placement
            placements[i, j] -= 1
            if 1 <= placements[i, j] && placements[i, j] <= (j == 1 ? m : n)
                loss = loss_function(matrix, placements)
                if loss > cur_max
                    changes[i, j] = -1
                    cur_max = loss
                end
            end
            
            # Revert the decrement and try incrementing
            placements[i, j] += 2
            if 1 <= placements[i, j] && placements[i, j] <= (j == 1 ? m : n)
                loss = loss_function(matrix, placements)
                if loss > cur_max
                    changes[i, j] = 1
                end
            end
            
            # Revert the increment to restore the original placement
            placements[i, j] -= 1
        end
    end
    
    return placements .+ changes .* multiplier
end

function generate_random_array(k, m, n)
    return hcat(rand(1:m, k), rand(1:n, k))
end

function gradient_decent(loss_function, matrix; num_routers::Int=2, iters::Int=500, attempts::Int=20, multiplier::Int=10)
    m, n = size(matrix)
    max_val_achieved = 0.0
    best_placement = nothing
    history = []  # List to store the progression of placements

    for _ in 1:attempts
        placement = generate_random_array(num_routers, m, n)
        placement_history = [copy(placement)]  # Store the initial placement for this attempt

        for i in 1:iters
            mult = floor(Int, iters / (i+1)) * multiplier
            placement = new_placement(loss_function, matrix, placement, multiplier=mult)
            push!(placement_history, copy(placement))  # Append the new placement to the history
        end
        if loss_function(matrix, placement) > max_val_achieved
            max_val_achieved = loss_function(matrix, placement)
            best_placement = placement
            history = placement_history  # Update the main history with the current attempt's history
        end
    end

    return best_placement, max_val_achieved, history
end



function history_to_matrix(history)
    num_timesteps = length(history)
    num_placements, _ = size(history[1])
    
    # Preallocate the history_matrix
    history_matrix = Array{Int64, 2}(undef, num_timesteps, 2*num_placements)
    
    for (i, placement) in enumerate(history)
        history_matrix[i, :] = reshape(placement', 1, 2*num_placements)
    end
    
    return history_matrix
end

matrix = image_to_grayscale("maincampusgray.jpg")
start_time = Dates.now()
bp, mva, history = gradient_decent(loss_function, matrix, num_routers=20, iters=30, attempts=10, multiplier=1)
println(mva)
end_time = Dates.now()
elapsed_time = Dates.value(end_time - start_time) / 1000  # Convert milliseconds to seconds
println("Elapsed time: ", elapsed_time, " seconds")


writedlm("matrix.csv", Float64.(matrix), ',')
writedlm("bp.csv", Float64.(bp), ',')
open("mva.txt", "w") do f
    write(f, string(mva))
end
history_matrix = history_to_matrix(history)
writedlm("history.csv", history_matrix, ',')

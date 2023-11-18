using Images
using ImageFiltering
using Random
using Plots
using LinearAlgebra
using Colors
using Dates
using Makie
using GLMakie
using DelimitedFiles
using Base.Threads: @threads, @spawn, @sync
# Assuming load and other required functions are from these packages
using FileIO
using CSV



function image_to_matrix(filepath::String, blue_size::Int = 5, blur_sigma::Float64 = 3.5)::Matrix{Float64}
    img = load(filepath)
    grey_scale = Gray.(img)
    grey_matrix = convert(Matrix{Float64}, grey_scale)
    transformed_matrix = (255 .- grey_matrix .* 255) .* 1000

    # Replace all values less than 10 with 10
    transformed_matrix[transformed_matrix .< 100] .= 100

    return transformed_matrix
end

function calculate_loss(matrix::Matrix{Float64}, placements::Matrix{Int32})::Float64
    m, n = size(matrix)
    k = size(placements, 1)
    epsilon::Float64 = .001

    # Shared minimum value across threads; each thread will have its own local copy
    min_seen = Threads.Atomic{Float64}(Inf)

    # Pre-compute the squared coordinates of placements to avoid this computation in the loop
    placements_sq = [placements[p, 1]^2 + placements[p, 2]^2 for p in 1:k]

    # Parallelize the outer loops
    @threads for i in 1:m
        for j in 1:n
            pixel_value = matrix[i, j]
            local_min_seen = Inf  # Local minimum for the current thread

            # Calculate the distance to all points and find the minimum
            for p in 1:k
                # Compute the difference in squares to avoid calculating the square root
                dist_sq = placements_sq[p] - 2 * (placements[p, 1] * i + placements[p, 2] * j) + i^2 + j^2
                dist_sq = max(dist_sq, epsilon^2)  # Apply epsilon to avoid division by zero

                # Calculate the pixel function and update the local minimum
                pixel_func = pixel_value / sqrt(dist_sq)
            
                local_min_seen = min(local_min_seen, pixel_func)
            end

            # Safely update the global minimum across all threads
            Threads.atomic_min!(min_seen, local_min_seen)
        end
    end
    return min_seen[]
end


function update_placements(matrix::Matrix{Float64}, placements::Matrix{Int32}, multiplier::Int = 2)::Matrix{Int32}
    base_error = calculate_loss(matrix, placements)
    k = size(placements, 1)
    delta_matrix = zeros(Int8, k, 2)  # Use the same delta_matrix if calling in a loop

    # Array to store tasks for parallel execution
    tasks = Vector{Task}(undef, k)

    @sync for p in 1:k
        tasks[p] = @spawn begin
            local_delta = [0, 0]
            for i in 1:2
                original_value = placements[p, i]
                # Check negative adjustment
                placements[p, i] = original_value - 1
                neg_error = calculate_loss(matrix, placements)
                # Check positive adjustment
                placements[p, i] = original_value + 1
                pos_error = calculate_loss(matrix, placements)
                # Reset to original
                placements[p, i] = original_value

                # Determine the best move
                if neg_error > base_error && neg_error >= pos_error
                    local_delta[i] = -1
                elseif pos_error > base_error
                    local_delta[i] = 1
                end
            end
            delta_matrix[p, :] = local_delta
        end
    end

    # Wait for all tasks to complete
    for task in tasks
        wait(task)
    end

    return placements + multiplier * delta_matrix
end
function initialize_placements(matrix::Matrix{Float64}, k::Int)::Matrix{Int32}
    m, n = size(matrix)

    rand_row_matrix = rand(1:m, k, 1)
    rand_col_matrix = rand(1:n, k, 1)

    return hcat(rand_row_matrix, rand_col_matrix)
end

function gradient_descent(loss_function::Function, matrix::Matrix{Float64}, k::Int = 2, iters::Int = 30, attempts::Int= 1, multiplier::Int = 10)
    m, n = size(matrix)
    max_val_achieved = 0.0
    best_placement = nothing
    history = []

    for _ in 1:attempts
        placement = initialize_placements(matrix, k)
        placement_history = [copy(placement)]  # Store the initial placement for this attempt

        for i in 1:iters
            mult = floor(Int, iters / (i+1)) * multiplier
            placement = update_placements(matrix, placement, mult)
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

m = image_to_matrix("ThreeHumps.jpg")
p = Matrix{Int32}([2 1; 23 23; 234 23; 23 2; 234 23; 234 35; 53 53])
@time calculate_loss2(m, p)
# @time best_placements, max_val, history = gradient_descent(calculate_loss, m, 3, 30, 50, 20)



# writedlm("matrix.csv", Float64.(m), ',')
# writedlm("bp.csv", Float64.(best_placements), ',')
# writedlm("history.csv", history, ',')
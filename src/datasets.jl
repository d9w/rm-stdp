using RDatasets

function col_normalize(X::Array{Float64})
    xmin = minimum(X, 1)
    xmax = maximum(X, 1)
    for i in 1:size(X, 2)
        X[:, i] = (X[:, i] .- xmin[i]) ./ (xmax[i] - xmin[i])
    end
    X
end

function get_iris()
    iris = dataset("datasets", "iris")

    iris_data = Array{Float64}(hcat(iris[:SepalLength], iris[:SepalWidth],
                                    iris[:PetalWidth], iris[:PetalLength]));
    X = col_normalize(iris_data)'

    sps = unique(iris[:Species])
    iris[:label] = indexin(iris[:Species], sps)
    Y = Array{Int64}(iris[:label])
    X, Y
end

function get_spirals()
    spirals = readtable("data/spiral.txt", header=false, separator=' ',
                        names=[:x, :y, :label])
    X = col_normalize(Array{Float64}(hcat(spirals[:x], spirals[:y])))'
    Y = Array{Int64}(spirals[:label])
    X, Y
end

function get_yeast()
    yeast = readtable("data/yeast.data", header=false, separator=' ',
                      names=[:seq, :mcg, :gvh, :alm, :mit, :erl, :pox, :vac,
                             :nuc, :local])
    yeast_data = Array{Float64}(hcat(
        map(i->yeast[i],[:mcg, :gvh, :alm, :mit, :erl, :pox, :vac, :nuc])...))
    X = col_normalize(yeast_data)'
    locs = unique(yeast[:local])
    yeast[:label] = indexin(yeast[:local], locs)
    Y = Array{Int64}(yeast[:label])
    X, Y
end

function get_data(problem::String)
    if problem == "iris"
        return get_iris()
    elseif problem == "spirals"
        return get_spirals()
    elseif problem == "yeast"
        return get_yeast()
    end
    error("Problem not found")
end

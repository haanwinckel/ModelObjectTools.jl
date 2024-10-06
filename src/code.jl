
using Base: isexpr

const TOL_PARAM_CHECKS = 1e-15
const MAX_MAGNITUDE_ENCODED_VALS = log(prevfloat(Inf) / 1e10) 
#Maximum value x such that exp(x) or exp(-x) makes sense numerically.
const MIN_DIFF_ORDERED_VARS = 1e-10


"""
modelObjectFromAnother(base, pairs::Pair{Symbol,<:Any}...)

Use this function to create a new struct starting from an 
existing one, except for a few specified changes.
# Examples
```julia-repl

julia> struct ExStr
       a::Float64
       vec::Tuple{Float64, Float64, Float64}
       end

julia> s = ExStr(3.0, (1.0,2.0,10.0))
ExStr(3.0, (1.0, 2.0, 10.0))

julia> s2 = modelObjectFromAnother(s, :vec => (2.0, 1.0, 5.0))
ExStr(3.0, (2.0, 1.0, 5.0))

julia> s3 = modelObjectFromAnother(s, :vec => (2.0, 1.0, 5.0), :a => 4.5)
ExStr(4.5, (2.0, 1.0, 5.0))
```
"""
function modelObjectFromAnother(base, changes::Pair{Symbol,<:Any}...)
    return modelObjectFromAnother(base, Dict(changes))
end

"""
modelObjectFromAnother(base, newParams::Dict{Symbol,<:Any})

Create a ModelParameters or EquilibriumVariables structure with elements
given by an existing object (the `base' argument),
except as described in the changes dictionary. 
"""
function modelObjectFromAnother(base, changes::Dict{Symbol,<:Any})
    allParams = Vector{Any}()
    numChanges = 0
    for p in fieldnames(typeof(base))
        if haskey(changes, p)
            param = changes[p]
            numChanges = numChanges + 1
        else
            param = getproperty(base, p)
        end
        push!(allParams, param)
    end
    for (paramName, value) in changes
        if !(paramName in fieldnames(typeof(base)))
            throw(DomainError((paramName, value), "Input included symbols that are "
                                                  *
                                                  "not settable fields of type $(typeof(base))."))
        end
    end
    #= if numChanges < length(changes) #Unused elements in the changes Dict
         throw(DomainError(changes, "Input included symbols that are "
             * "not settable fields of type $(typeof(base))."))
     end=#
    baseType = typeof(base)
    return baseType(allParams...)
end

const varInfo = Dict{Symbol,NamedTuple{(:lb, :ub, :normalization),Tuple{Float64,Float64,Symbol}}}()


"""
addVarInfo(varName::Symbol; lb = -Inf, ub = Inf, normalization::Symbol = :none)

Use this function to register the variable names that will correspond 
to fields in the structures defining model objects (parameter sets, sets of 
endogenous variables, etc.).

For each variable, the allowed range of values is (lb, ub).
To allow the extreme points in the range, you may use a slightly smaller/larger 
value. For example, setting lb = prevfloat(0.0) will allow the parameter 
to be exactly 0.0 or more.

If bounds are not needed, use the default options: lb = -Inf, ub = Inf.

The allowed normalization values are:
- :sumToOne     (for unidimensional arrays)
- :firstIsZero  (for unidimensional arrays)
- :firstIsOne   (for unidimensional arrays)
- :ordered      (for unidimensional arrays or matrices. With matrices, it means
                the values must be increasing along columns)
- :none         (the default option)
"""
function addVarInfo(varName::Symbol; lb=-Inf, ub=Inf, normalization::Symbol=:none)
    if !(normalization in [:none, :sumToOne, :firstIsZero, :firstIsOne, :ordered])
        throw(error("Invalid normalization input: $normalization."))
    end
    val = (; lb=Float64(lb), ub=Float64(ub), normalization=normalization)
    if haskey(varInfo, varName) && varInfo[varName] != val
        throw(error("Tried to insert different varInfo for variable $varName:"
                    * "\n Pre-existing value was $(varInfo[varName])"
                    * "\n Attempted to redefine as $val"))
    else
        varInfo[varName] = val
    end
end


"""
checkValidity(m)

Checks that either parameters or endogenous variables lie within 
bounds and satisfy other normalizations/constraints embedded in the
varInfo dictionary. Every component of the structure m must be a 
variable registered in the varInfo dictionary, using the addVarInfo()
function.

Returns a tuple where the first element is a boolean indicating
validity, the second element is either nothing or a symbol indicating
the invalid parameter, and the third element is either nothing or the
invalid value.
"""
function checkValidity(m)
    for f in fieldnames(typeof(m))
        val = getproperty(m, f)
        outOfBounds = any(val .< varInfo[f].lb - TOL_PARAM_CHECKS) ||
                      any(val .> varInfo[f].ub + TOL_PARAM_CHECKS)
        notOrdered = val isa AbstractArray &&
                     (varInfo[f].normalization == :ordered) &&
                     !all(diff(val, dims=1) .>= -TOL_PARAM_CHECKS)
        notSumToOne = val isa AbstractArray &&
                      (varInfo[f].normalization == :sumToOne) &&
                      !isapprox(sum(val), 1.0, atol=TOL_PARAM_CHECKS)
        firstNotZero = val isa AbstractArray &&
                       (varInfo[f].normalization == :firstIsZero) &&
                       !isapprox(val[1], 0.0, atol=TOL_PARAM_CHECKS)
        firstNotOne = val isa AbstractArray &&
                      (varInfo[f].normalization == :firstIsOne) &&
                      !isapprox(val[1], 1.0, atol=TOL_PARAM_CHECKS)
        if outOfBounds || notOrdered || notSumToOne || firstNotZero || firstNotOne
            throw(error("$val is not a valid input for $f."))
        end
    end
end


"""
vectorRepresentation(input;
        fields = fieldnames(typeof(input)))

Creates a vector of real values that represents either
model parameters or equilibrium variables. This vector can be used for
optimization without the need for constraints; all elements can vary
in (-Inf, Inf) while keeping the values within bounds defined 
by varInfo.

By default, it will encode all fields of the input structure. If
only a subset of fields is desired, then set the fields keyword argument.
"""
function vectorRepresentation(input;
    fields=fieldnames(typeof(input)))
    #Start with empty vector
    v = Vector{Float64}()

    #Loop over each of the fields to be included in the representation.
    for f in fields
        #Start with original value for parameters/endog. variables
        val = getproperty(input, f)

        #If first element is normalized, ignore it:
        if val isa AbstractVector && length(val) > 1 &&
           (varInfo[f].normalization == :firstIsZero || varInfo[f].normalization == :firstIsOne)
            val = val[2:end]
        end

        #Transforming to a variable with range (-Inf, Inf)
        function transformToUnbounded(inp, a, b)
            if !isinf(a) && !isinf(b) #Original range: (a,b)
                y = log.((inp .- a) ./ (b .- inp))
            elseif !isinf(a) #Original range: (a,∞)
                y = log.(inp .- a)
            elseif !isinf(b) #Original range: (-∞, b)
                y = -log.(b .- inp)
            else #Otherwise, no action required!
                y = inp
            end
            return y
        end

        #If it's an ordered vector (matrix), represent it as first element (row)
        # and then differences between consecutive elements (along rows)
        if (varInfo[f].normalization == :ordered) && size(val, 1) > 1
            transfVal = zeros(size(val))
            for col = 1:size(val, 2)
                transfVal[1, col] = transformToUnbounded(val[1, col], varInfo[f].lb, varInfo[f].ub)
                for row = 2:size(val, 1)
                    transfVal[row, col] = transformToUnbounded(val[row, col],
                        val[row-1, col], varInfo[f].ub)
                end
            end
        elseif (varInfo[f].normalization == :sumToOne)
            transfVal = zeros(length(val) - 1)
            remainingProb = 1.0
            for row in eachindex(transfVal)
                ratio = val[row] / remainingProb
                transfVal[row] = log(ratio / (1.0 - ratio))
                remainingProb = remainingProb - val[row]
            end
        else
            transfVal = transformToUnbounded(val, varInfo[f].lb, varInfo[f].ub)
        end

        #If it's a matrix, represent as a vector
        if length(transfVal) > size(transfVal, 1)
            transfVal = vec(transfVal)
        end

        #Concatenate all values in output vector
        v = vcat(v, transfVal)
    end
    return v
end


"""
modelObjectFromVector(base, vectorInput::AbstractArray{<:Real},
        fields = fieldnames(typeof(base)))

Constructs a model object based a vector of transformed 
parameter values created by the vectorRepresentation() function.
The specific parameters being modified are given by the fields input.
Other parameters come from the base input.
"""
function modelObjectFromVector(base, vectorInput::AbstractVector{<:Real};
    fields=fieldnames(typeof(base)))

    #Auxiliary function
    #De-transforming from a variable with range (-Inf, Inf)
    function detransformFromUnbounded(inp, a, b)
        y = fill(NaN, size(inp))
        if !isinf(a) && !isinf(b) #Original range: (a,b)
            for i in eachindex(inp)
                if inp[i] > MAX_MAGNITUDE_ENCODED_VALS
                    y[i] = b
                elseif inp[i] < -MAX_MAGNITUDE_ENCODED_VALS #Negative infinity
                    y[i] = a
                else
                    tmp = exp(inp[i])
                    y[i] = (b * tmp + a) / (1 + tmp)
                end
            end 
        elseif !isinf(a) #Original range: (a,∞)
            for i in eachindex(inp)
                if inp[i] < -MAX_MAGNITUDE_ENCODED_VALS
                    y[i] = a
                else
                    y[i] = exp(inp[i]) + a
                end
            end
        elseif !isinf(b) #Original range: (-∞, b)
            for i in eachindex(inp)
                if inp[i]> MAX_MAGNITUDE_ENCODED_VALS
                    y[i] = b
                else
                    y[i] = b - exp(-inp[i])
                end
            end
        else #Otherwise, no action required!
            y .= inp
        end
        if inp isa Number 
            return y[1]
        else
            return y
        end
    end

    if any(isnan.(vectorInput))
        throw(DomainError(vectorInput, "Cannot have NaN's in vector input."))
    end

    #Start with empty dictionary which will map from parameter names to values.
    paramChanges = Dict{Symbol,Any}()

    #Loop over each of the fields to be changed to populate the Dict.
    for f in fields
        #Get corresponding entries in the input vector
        numElements = length(getproperty(base, f))
        if varInfo[f].normalization == :sumToOne ||
           varInfo[f].normalization == :firstIsZero || varInfo[f].normalization == :firstIsOne
            numElements = numElements - 1
        end
        if length(vectorInput) < numElements
            throw(DomainError(vectorInput,
                "Argument vectorInput does not have enough elements."))
        end
        val = vectorInput[1:numElements]

        #Erase from input vector (so that previous block works in the next iteration)
        vectorInput = vectorInput[(numElements+1):end]

        #Reshape to be of the original size, if needed
        if size(getproperty(base, f), 1) < numElements #multidimensional
            val = reshape(val, size(getproperty(base, f)))
        end

        #If ordered input, accumulate from (first, changes) format
        if (varInfo[f].normalization == :ordered) && size(val, 1) > 1
            val[1, :] .= detransformFromUnbounded(val[1, :], varInfo[f].lb, varInfo[f].ub)
            for row = 2:size(val, 1), col = 1:size(val, 2)
                val[row, col] = detransformFromUnbounded(val[row, col],
                    val[row-1, col], varInfo[f].ub)
                if row < size(val, 1) && isinf(val[row, col]) && val[row, col] > 0.0
                    val[row, col] = exp(MAX_MAGNITUDE_ENCODED_VALS + row)
                end
            end
        elseif varInfo[f].normalization == :sumToOne
            remainingProb = 1.0
            for row in eachindex(val)
                val[row] = remainingProb / (1 + exp(-val[row]))
                remainingProb = remainingProb - val[row]
            end
            val = vcat(val, remainingProb)
        else
            val = detransformFromUnbounded(val, varInfo[f].lb, varInfo[f].ub)
        end

        #If we get an inf, replace with a large number
        for row = 1:size(val, 1), col = 1:size(val, 2)
            if isinf(val[row, col])
                if val[row, col] > 0
                    val[row, col] = prevfloat(Inf)
                else
                    val[row, col] = nextfloat(-Inf)
                end
            end
        end

        #Change type from vector to number if necessary
        if getproperty(base, f) isa Number
            val = val[1]
        end

        #Add normalization
        if varInfo[f].normalization == :firstIsZero
            val = vcat(0.0, val)
        elseif varInfo[f].normalization == :firstIsOne
            val = vcat(1.0, val)
        end

        #Add to Dictionary
        paramChanges[f] = val
    end

    if length(vectorInput) > 0
        throw(DomainError(vectorInput,
            "Argument vectorInput has too many elements."))
    end

    #Create and return the new object
    return modelObjectFromAnother(base, paramChanges)
end


"""
modelObjectFromVector(baseType::Type, vectorInput::AbstractArray{<:Real},
        fields = fieldnames(baseType))

Constructs a model object based a vector of transformed 
parameter values created by the vectorRepresentation() function.
The specific parameters being modified are given by the fields input.
Other parameters come from the the default values for type baseType.
"""
function modelObjectFromVector(baseType::Type, vectorInput::AbstractVector{<:Real};
    fields=fieldnames(baseType))
    return modelObjectFromVector(baseType(), vectorInput; fields)
end



"""
showModelObject(mo)

Displays a representation of the model object mo that 
is simple to understand and that can be used as a command
to construct an identical object (if a default constructor
is available).
# Examples
```julia-repl

julia> struct ExStr
       a::Float64
       vec::Tuple{Float64, Float64, Float64}
       end

julia> s = ExStr(3.0, (1.0,2.0,10.0))
ExStr(3.0, (1.0, 2.0, 10.0))

julia> showModelObject(s)
modelObjectFromAnother(ExStr(), 
                           :a => 3.0, 
                         :vec => (1.0, 2.0, 10.0))
```
"""
function showModelObject(mo)
    println(modelObjectString(mo))
end


"""
modelObjectString(mo)

Returns a string that represents the model object mo.
```
"""
function modelObjectString(mo)
    str = "modelObjectFromAnother($(typeof(mo))(), \n"
    offset = length(str) - 4
    fields = fieldnames(typeof(mo))
    for i in eachindex(fields)
        f = fields[i]
        if f != :H && f != :G && f != :EH
            f_string = lpad(":$f", offset, " ")
            str *= f_string * " => $(getproperty(mo, f))"
            if i < length(fields)
                str *= ", \n"
            end
        end
    end
    str *= ")"
    return str
end


"""
compareModelObjects(mo1, mo2; tol=$TOL_PARAM_CHECKS, verbose=false)

Compares two model objects field by field. Output has the format
(differences, max_abs_diff), where differences is a vector stipulating
the observed differences and max_abs_diff is the maximum absolute
value of differences across all fields.
"""
function compareModelObjects(mo1, mo2; tol=TOL_PARAM_CHECKS*10, verbose=false)
    if typeof(mo1) != typeof(mo2)
        throw(error("Can only compare model objects of the same type."))
    end
    differences = Vector{Tuple{Symbol,Any}}()
    max_abs_diff = 0.0
    str = ""
    fields = fieldnames(typeof(mo1))
    for i in eachindex(fields)
        f = fields[i]
        val1 = getproperty(mo1, f)
        val2 = getproperty(mo2, f)
        if size(val1) != size(val2)
            push!(differences, (f, Inf64))
            str *= "$f: different dimensionality.\n"
        end
        diff = maximum(abs.(val2 - val1))
        if diff > tol
            push!(differences, (f, val2 - val1))
            if diff > max_abs_diff
                max_abs_diff = diff
            end
            str *= "$f: $(val2 - val1).\n"
        end
    end
    if verbose
        if max_abs_diff == 0.0
            println("No differences.")
        else
            println(str)
        end
    end
    return (differences, max_abs_diff)
end


"""
    @modef typedef

This is slight modification of the @kwdef helper macro defined in Base.
It ensures that the struct to which it applies can be effectively used  
with the functions defined in this package. Specifically:
(1) Every field must have a default value, or the macro throws an error;
(2) It automatically creates a constructor with optional keyword
    arguments in the same style as Base.@kwdef, but in addition, that
    constructor checks whether all parameters are valid as defined by 
    the registering of variable names via addVarInfo().

# Examples
```julia-repl
julia> @modef struct ModelParams
                  myScalarParam::Float64 = 4.0
                  myVecParam::Vector{Float64} = [1.0, 2.0]
              end
ModelParams

julia> addVarInfo(:myScalarParam; lb = 0.0); addVarInfo(:myVecParam; normalization = :ordered);

julia> ModelParams()
ModelParams(4.0, [1.0, 2.0])

julia> ModelParams(myVecParam = [2.0, 1.0])
ERROR: [2.0, 1.0] is not a valid input for myVecParam.
Stacktrace:
[...]
```
"""
macro modef(expr)

    isexpr(expr, :struct) || error("Invalid usage of @modef")
    _, T, fieldsblock = expr.args
    if T isa Expr && T.head === :<:
        T = T.args[1]
    end

    fnames = Any[]
    defvals = Any[]
    extract_names_and_defvals_from_kwdef_fieldblock!(fieldsblock, fnames, defvals)
    parameters = map(fnames, defvals) do fieldname, defval
        if isnothing(defval)
            throw(error("When using @modef, all fields must have default values."))
        else
            return Expr(:kw, fieldname, esc(defval))
        end
    end

    #Add explicit default creator that checks validity of inputs 
    body = Expr(:block, #__source__, 
        Expr(:(=), :mo, Expr(:call, :new, fnames...)),
        Expr(:call, :checkValidity, :mo),
        Expr(:return, :mo))
    functionExpr = Expr(:function, Expr(:call, T, fnames...), body)
    push!(expr.args[3].args, functionExpr)

    # Only define a constructor if the type has fields, otherwise we'll get a stack
    # overflow on construction
    if !isempty(parameters)
        T_no_esc = unescape(T)
        if T_no_esc isa Symbol
            sig = Expr(:call, esc(T), Expr(:parameters, parameters...))
            body = Expr(:block, __source__,
                Expr(:(=), :mo, Expr(:call, esc(T), fnames...)),
                Expr(:return, :mo))
            kwdefs = Expr(:function, sig, body)
        elseif isexpr(T_no_esc, :curly)
            throw(error("@modef not currently defined for parametric structs."))
            #=
            # if T == S{A<:AA,B<:BB}, define two methods
            #   S(...) = ...
            #   S{A,B}(...) where {A<:AA,B<:BB} = ...
            S = T.args[1]
            P = T.args[2:end]
            Q = Any[isexpr(U, :<:) ? U.args[1] : U for U in P]
            SQ = :($S{$(Q...)})
            body1 = Expr(:block, __source__, Expr(:call, esc(S), fnames...))
            sig1 = Expr(:call, esc(S), Expr(:parameters, parameters...))
            def1 = Expr(:function, sig1, body1)
            body2 = Expr(:block, __source__, Expr(:call, esc(SQ), fnames...))
            sig2 = :($(Expr(:call, esc(SQ), Expr(:parameters, parameters...))) where {$(esc.(P)...)})
            def2 = Expr(:function, sig2, body2)
            kwdefs = Expr(:block, def1, def2)
            =#
        else
            error("Invalid usage of @modef")
        end
    else
        kwdefs = nothing
    end
    return quote
        $(esc(:($Base.@__doc__ $expr)))
        $kwdefs
    end
end



# @kwdef helper function
# mutates arguments inplace
function extract_names_and_defvals_from_kwdef_fieldblock!(block, names, defvals)
    for (i, item) in pairs(block.args)
        if isexpr(item, :block)
            extract_names_and_defvals_from_kwdef_fieldblock!(item, names, defvals)
        elseif item isa Expr && item.head in (:escape, :var"hygienic-scope")
            n = length(names)
            extract_names_and_defvals_from_kwdef_fieldblock!(item, names, defvals)
            for j in n+1:length(defvals)
                if !isnothing(defvals[j])
                    defvals[j] = Expr(item.head, defvals[j])
                end
            end
        else
            def, name, defval = @something(def_name_defval_from_kwdef_fielddef(item), continue)
            block.args[i] = def
            push!(names, name)
            push!(defvals, defval)
        end
    end
end

function def_name_defval_from_kwdef_fielddef(kwdef)
    if kwdef isa Symbol
        return kwdef, kwdef, nothing
    elseif isexpr(kwdef, :(::))
        name, _ = kwdef.args
        return kwdef, unescape(name), nothing
    elseif isexpr(kwdef, :(=))
        lhs, rhs = kwdef.args
        def, name, _ = @something(def_name_defval_from_kwdef_fielddef(lhs), return nothing)
        return def, name, rhs
    elseif kwdef isa Expr && kwdef.head in (:const, :atomic)
        def, name, defval = @something(def_name_defval_from_kwdef_fielddef(kwdef.args[1]), return nothing)
        return Expr(kwdef.head, def), name, defval
    elseif kwdef isa Expr && kwdef.head in (:escape, :var"hygienic-scope")
        def, name, defval = @something(def_name_defval_from_kwdef_fielddef(kwdef.args[1]), return nothing)
        return Expr(kwdef.head, def), name, isnothing(defval) ? defval : Expr(kwdef.head, defval)
    end
end

"""
    Meta.unescape(expr)

Peel away `:escape` expressions and redundant block expressions (see
[`unblock`](@ref)).
"""
function unescape(@nospecialize ex)
    ex = unblock(ex)
    while isexpr(ex, :escape) || isexpr(ex, :var"hygienic-scope")
        ex = unblock(ex.args[1])
    end
    return ex
end


"""
    Meta.unblock(expr)

Peel away redundant block expressions.

Specifically, the following expressions are stripped by this function:
- `:block` expressions with a single non-line-number argument.
- Pairs of `:var"hygienic-scope"` / `:escape` expressions.
"""
function unblock(@nospecialize ex)
    while isexpr(ex, :var"hygienic-scope")
        isexpr(ex.args[1], :escape) || break
        ex = ex.args[1].args[1]
    end
    isexpr(ex, :block) || return ex
    exs = filter(ex -> !(isa(ex, LineNumberNode) || isexpr(ex, :line)), ex.args)
    length(exs) == 1 || return ex
    return unblock(exs[1])
end

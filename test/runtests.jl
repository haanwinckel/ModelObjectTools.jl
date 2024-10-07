using ModelObjectTools
using Test

@testset "ModelObjectTools.jl" begin

    #Create fake variables for Testing
    scalarVarLb = 1.1
    scalarVarUb = 2.5
    addVarInfo(:scalarVar; lb=scalarVarLb, ub=scalarVarUb)
    addVarInfo(:orderedVar; normalization=:ordered)
    addVarInfo(:orderedMatrix; normalization=:ordered)
    addVarInfo(:firstOneVar; normalization=:firstIsOne)
    addVarInfo(:firstZeroVar; normalization=:firstIsZero)
    addVarInfo(:sumToOneVar; lb=0.0, normalization=:sumToOne)
    addVarInfo(:orderedAndBoundsVar; lb=scalarVarLb, ub=scalarVarUb,
        normalization=:ordered)
    addVarInfo(:unrestrictedVar)

    #Test that we can only use @modef when the struct defines
    # default values for all fields.
    expression_ok = :(
        @modef struct Str1
        scalarVar::Float64 = 2.0
        orderedVar::Vector{Float64} = [1.0, 2.0, 3.0]
    end)
    @test eval(expression_ok) isa Any
    expression_problem = :(
        @modef struct Str_problem
        scalarVar::Float64
        orderedVar::Vector{Float64} = [1.0, 2.0, 3.0]
    end)
    @test_throws LoadError eval(expression_problem)

    #Test that we cannot initialize a struct if we forget
    # to register all variable names
    @modef struct Str2
        scalarVar::Float64 = 2.0
        forgottenVar::Float64 = 3.0
        orderedVar::Vector{Float64} = [1.0, 2.0, 3.0]
    end
    @test_throws KeyError Str2()

    #Create another structure with all variables for further testing
    @modef struct MyStruct
        scalarVar::Float64 = 2.0
        orderedVar::Vector{Float64} = [1.0, 2.0, 3.0]
        orderedMatrix::Matrix{Float64} = [1.0 2.0; 3.0 4.0]
        firstOneVar::Vector{Float64} = [1.0, 2.0, 3.0]
        firstZeroVar::Vector{Float64} = [0.0, 2.0, 3.0]
        sumToOneVar::Vector{Float64} = [0.2, 0.5, 0.2, 0.1]
        orderedAndBoundsVar::Vector{Float64} = [1.2, 1.3, 2.2]
        unrestrictedVar::Float64 = 3.0
    end

    m = MyStruct()

    #Can make valid modifications 
    modStr = modelObjectFromAnother(m, :scalarVar => 2.1, :firstZeroVar => [0.0, 1.0, 2.0])
    @test modStr.scalarVar == 2.1
    @test modStr.firstZeroVar == [0.0, 1.0, 2.0]

    #Cannot make invalid modifications
    @test_throws Exception modelObjectFromAnother(m, :scalarVar => [1.8, 1.9]) #Should be a scalar
    @test_throws Exception modelObjectFromAnother(m, :orderedVar => 3.4) #Should be a vector
    @test_throws Exception modelObjectFromAnother(m, :orderedVar => m.orderedVar[end:(-1):1]) #Should be ordered
    @test_throws Exception modelObjectFromAnother(m, :orderedMatrix => [1.0 1000.0; 2.0 999.0]) #Cols should be ordered
    @test_throws Exception modelObjectFromAnother(m, :sumToOneVar => m.sumToOneVar * 0.5) #Should add to one
    @test_throws Exception modelObjectFromAnother(m, :scalarVar => scalarVarLb - 1) #Lower bound 
    @test_throws Exception modelObjectFromAnother(m, :scalarVar => scalarVarUb + 1) #Upper bound 
    @test_throws Exception modelObjectFromAnother(m, :blabla => 5.0) #Not a real field
    @test_throws Exception modelObjectFromAnother(m, :firstOneVar => [0.0, 2.0, 3.0])
    @test_throws Exception modelObjectFromAnother(m, :firstZeroVar => [1.0, 2.0, 3.0])

    function objectApprox(m1, m2)
        return compareModelObjects(m1, m2)[2] < 1e-12
    end

    @testset "Encoding and recoding specific params" for field in fieldnames(MyStruct)
        #println(field)
        m = MyStruct()
        @test vectorRepresentation(m, fields=(field,)) isa AbstractVector{<:Real}
        x = vectorRepresentation(m, fields=(field,))
        @test modelObjectFromVector(m, x, fields=(field,)) isa MyStruct
        val = getproperty(m, field)
        if length(val) == 1
            wrong_m = modelObjectFromAnother(m, field => getproperty(m, field) + 0.001)
        else
            change = zeros(size(val))
            change[end-1] = -0.001
            change[end] = 0.001
            wrong_m = modelObjectFromAnother(m, field => val + change)
        end
        @test !objectApprox(m, wrong_m)
        m2 = modelObjectFromVector(wrong_m, x, fields=(field,))
        @test objectApprox(m, m2)
        @test_throws Exception modelObjectFromVector(m, vcat(x, 1), fields=(field,))
        @test_throws Exception modelObjectFromVector(m, x[1:(end-1)], fields=(field,))
    end

    @testset "Encoding and recoding complete model" begin
        m = MyStruct()
        @test vectorRepresentation(m) isa AbstractVector{<:Real}
        @test modelObjectFromVector(m, vectorRepresentation(m)) isa MyStruct
        wrong_m = modelObjectFromAnother(m, :scalarVar => m.scalarVar + 0.1)
        x = vectorRepresentation(m)
        @test objectApprox(m, modelObjectFromVector(wrong_m, x))
        @test_throws Exception modelObjectFromVector(m, vcat(x, 1.0)) #Extra element in vector
        @test_throws Exception modelObjectFromVector(m, x[1:(end-1)]) #Missing element in vector
    end

    @testset "Changes in vector representation generate valid models" begin
        m = MyStruct()
        x = vectorRepresentation(m)
        for i in eachindex(x)
            for change in [nextfloat(-Inf), -100, -5, -0.1,
                            0.1, 5, 100, prevfloat(Inf)]
                x2 = copy(x)
                x2[i] = x[i] + change
                @test modelObjectFromVector(m, x2) isa MyStruct

                #Will also test successful recovery from reverting the change 
                # in x; but that only works for ``reasonable'' values of the 
                # change in x.
                if abs(change) <= 10
                    m2 = modelObjectFromVector(m, x2)
                    x3 = vectorRepresentation(m2)
                    x3[i] = x3[i] - change
                    m3 = modelObjectFromVector(m2, x3)
                    @test objectApprox(m, m3)
                end
                #The design principle here is: 
                # It should be possible to create objects from any vector of 
                # numbers, even very large (but finite) numbers, because optimization 
                # procedures may often generate such inputs. But making large changes
                # to x may lead to e.g., variables hitting exactly their lower or 
                # upper bounds, and then the vector representation will have infinite
                # values (such that ``reverting'' the original change is not possible).
                # This is not an issue in practice, provided the starting point 
                # in the optimization procedure is not too close to the bounds 
                # (or, for ordered vectors, if the starting value does not have 
                # adjacent values too close to one another).
            end
        end
    end


end

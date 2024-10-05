using ModelObjectTools, StaticArrays, LeastSquaresOptim

#Structure representing model parameters:
@modef struct Params
    beta::Float64 = 4.0
    gamma::SVector{3,Float64} = [1.0, 1.2, 0.2]
    rho::SVector{3,Float64} = [0.2, 0.8, 0.0]
end
#The @modef macro requires every field to have a default
# value. It creates a constructor with optional arguments in 
# the same way that Base.@kwdef does. In addition, it imposes 
# that the object cannot be created with values that do not 
# satisfy some theoretical restrictions which we set here:

addVarInfo(:beta)
addVarInfo(:gamma; lb=0.0, ub=2.0, normalization=:firstIsOne)
addVarInfo(:rho; lb=prevfloat(0.0), normalization=:sumToOne)
#Each call to addVarInfo ``registers'' a parameter name. This is where
# you define theoretical restrictions such as bounds and normalizations.
#Note: bounds are open intervals, (lb, ub). Using ``prevfloat'' 
# in the last line above makes the value 0.0 valid.

#Another structure to hold endogenous equilibrium variables:
@modef struct EqVars
    x::SVector{2,Float64} = [1.0, 2.0]
    y::SVector{3,Float64} = [0.0, 3.0, 4.0]
end

#Register the variable names and impose more restrictions:
addVarInfo(:x; normalization=:ordered)
addVarInfo(:y; normalization=:firstIsZero)
#Note: ``ordered'' also works for matrices. In that case, it 
# requires that each column of the matrix is increasing, as
# you move down rows.

#Now let's define a function that evaluates the equilibrium 
# conditions of the model, that is, the system of equations 
# that has to be solved to find an equilibrium:
function equilibriumConditions(p::Params, ev::EqVars)
    #There are five values in the x and y vectors. But
    # because of the ``firstIsZero'' normalization in y,
    # there are only four choice variables. Thus, there
    # should be four equilibrium conditions in the model.

    eqConds = fill(NaN, 4)
    eqConds[1:3] .= (p.rho .* p.gamma) - (ev.y .+ ev.x[1])
    eqConds[4] = ev.x[2] - p.beta^2
    return eqConds
end

#Now we define a structure that corresponds to an equilibrium 
# of this model. The Equilibrium object can only be created 
# if the equilibrium conditions are satisfied.
const EQ_TOL = 1e-12

struct Equilibrium
    p::Params
    ev::EqVars

    function Equilibrium(p, ev)
        if maximum(abs.(equilibriumConditions(p, ev))) > EQ_TOL
            return nothing
        else
            return new(p, ev)
        end
    end
end

#Now define a function that solves for an equilibrium. Here 
# we use the vectorRepresentation() and modelObjectFromVector() 
# functions provided by the ModelObjectTools package 
# to translate between structures and vectors of real numbers.
function solveForEquilibrium(p::Params; ini_guess=EqVars())
    ini_x = vectorRepresentation(ini_guess)
    function optimizationObjFun(x)
        ev_guess = modelObjectFromVector(EqVars, x)
        return equilibriumConditions(p, ev_guess)
    end
    opt = optimize(optimizationObjFun, ini_x, LevenbergMarquardt(), 
        x_tol = EQ_TOL, f_tol = EQ_TOL, g_tol = EQ_TOL)
    ev_found = modelObjectFromVector(EqVars, opt.minimizer)
    return Equilibrium(p, ev_found)
end

#Now let's put all of this to work.

#Create a specific structure with model parameters. It has 
# default values for all parameters except gamma.
myParams = Params(gamma=[1.0, 0.8, 1.3])

#Display it for easy visualization:
showModelObject(myParams)

#Solve for equilibrium:
eq = solveForEquilibrium(myParams)

#Display the equilibrium variables:
showModelObject(eq.ev)

#Now suppose we want to do comparative statics. We cannot 
# modify the parameter structure p using something like:
# p.beta = 5.0  #Throws an error!
# because p is immutable. The solution is to generate another 
# model object using the following function:
myParams2 = modelObjectFromAnother(myParams, :beta => 5.0)

eq2 = solveForEquilibrium(myParams2; ini_guess = eq.ev)

#Display differences in the equilibrium variables:
compareModelObjects(eq.ev, eq2.ev; verbose = true)

#If one were to estimate that model, the vectorRepresentation()
# and modelObjectFromVector() functions could also be used on the 
# Params object, say, in an optimization procedure that chooses 
# parameters to match simulated moments from an equilibrium.
#And if only a subset of parameters is being estimated, that's fine;
# those functions have an optional ``fields'' requirement that can 
# be used to select the parameters being estimated.
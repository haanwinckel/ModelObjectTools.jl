# ModelObjectTools

[![Build Status](https://github.com/haanwinckel/ModelObjectTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/haanwinckel/ModelObjectTools.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/haanwinckel/ModelObjectTools.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/haanwinckel/ModelObjectTools.jl)

This package provides utilities to help with equilibrium-solving and estimation of models, by allowing for easy encoding and decoding of structures holding model variables into vectors of real numbers---which can be used as choice variables in optimization procedures.

The structures to be encoded and decoded can have elements that are Floats or one- or two-dimensional arrays of Floats. The encoding procedures easily deal with common theoretical restrictions on the variables, such as:
- Lower and upper bounds;
- Requiring vectors to be strictly increasing (or each column of a matrix to be strictly increasing);
- The first element of a vector needs to be zero, or one;
- All elements in a vector must add up to one.

Example:
```
using ModelObjectTools

@modef struct MyStruct
    a = 4.0 #A scalar field
    b = [1.0, 1.2, 0.2] #A vector field
    c = [1.0 10.0; 3.0 11.0; 20.0 12.0] #A matrix field
end

#The @modef macro requires every field to have a default
# value. It creates a constructor with optional arguments in 
# the same way that Base.@kwdef does. In addition, it makes 
# it impossible to create the object if field values do not 
# satisfy theoretical restrictions which we set in the
# following way:

addVarInfo(:a)
addVarInfo(:b; lb=0.0, ub=2.0, normalization=:firstIsOne)
addVarInfo(:c; lb=0.0, normalization=:ordered)

#Each call to addVarInfo ``registers'' a parameter name.
# Field a has no theoretical restrictions. b has bounds 
# of the form [lb, ub] and the first element is required 
# to be one. The matrix field c needs to have all columns 
# being non-decreasing.

st = MyStruct() #structure with default values

x = vectorRepresentation(st)
# x is a 9-element vector of Float64

x[3] = x[3] + 1.0

new_st = modelObjectFromVector(MyStruct, x)
# new_st is a MyStruct similar to st, but with a change 
# in one of the fields (in this example, b is changed).
```

Encoding can also be partial, that is, only a subset of parameters in the struct need to be encoded/decoded.

ModelObjectTools.jl is designed with immutability and [class invariants](https://en.wikipedia.org/wiki/Class_invariant) in mind. Here is an example of an useful class invariant for scientific models:
> An object of type ``Equilibrium'' always corresponds to a combination of parameters and endogenous variables that solves the equilibrium conditions of the model.

A simple, but effective way to achieve that is:
1. When creating the object, check that the candidate solution does satisfy the equilibrium conditions;
2. Make it impossible to modify that object after it is created.

Then, when you use that Equilibrium object as an input to another function (say, one that simulates moments to be compared with data moments), you can be sure that it does correspond to an equilibrium of the model.

One way to make structures in Julia fully immutable is to use SVector or SMatrix from the StaticArrays package.jl package to represent vectors and matrices. Where StaticArrays are not suitable, you can use ReadOnlyArrays.

**Note**: ModelObjectTools.jl is *not* designed for performance. It is primarily aimed at applications where objective function in the optimization procedure is somewhat costly, such that the encoding/decoding of variables is a small part of the computation.

Below is a working example of how ModelObjectTools.jl can be used in combination with StaticArrays to create an equilibrium-solving procedure that yields an immutable Equilibrium object. It also shows other useful functions provided by ModelObjectTools.jl:

```
using ModelObjectTools, StaticArrays, LeastSquaresOptim

@modef struct Params
    beta::Float64 = 4.0
    gamma::SVector{3,Float64} = [1.0, 1.2, 0.2]
    rho::SVector{3,Float64} = [0.2, 0.8, 0.0]
end
addVarInfo(:beta)
addVarInfo(:gamma; lb=0.0, ub=2.0, normalization=:firstIsOne)
addVarInfo(:rho;   lb=0.0, normalization=:sumToOne)

@modef struct EqVars
    x::SVector{2,Float64} = [1.0, 2.0]
    y::SVector{3,Float64} = [0.0, 3.0, 4.0]
end
addVarInfo(:x; normalization=:ordered)
addVarInfo(:y; normalization=:firstIsZero)

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
# modify the parameter structure using something like:
# myParams.beta = 5.0  #Throws an error!
# because myParams is immutable. The solution is to generate another 
# model object using the following function:

myParams2 = modelObjectFromAnother(myParams, :beta => 5.0)

eq2 = solveForEquilibrium(myParams2; ini_guess = eq.ev)

#Display differences in the equilibrium variables:
compareModelObjects(eq.ev, eq2.ev; verbose = true)
```

If one were to estimate that model, the vectorRepresentation()
and modelObjectFromVector() functions could also be used on the 
Params object, say, in an optimization procedure that chooses 
parameters to match simulated moments from an equilibrium.
And if only a subset of parameters is being estimated, that's fine;
those functions have an optional ``fields'' requirement that can 
be used to select the parameters being estimated.
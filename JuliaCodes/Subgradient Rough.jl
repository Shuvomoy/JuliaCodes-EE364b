n=1000# Numbe of variables
m=50 # Number of equality constraints
A=randn(m,n)
b=randn(m,1)
1

# Solving using convex
using Convex
using SCS
xCvx=Variable(n)
problem=minimize( norm(xCvx,1),
#     subject to
                A*xCvx==b)

solve!(problem, SCSSolver())

print(
"The status of the solution is ", problem.status,
"\nwith optimal objective value, f(x*)=", problem.optval,
)

xMinValue=xCvx.value
fMinValue=problem.optval

#Finding an initial feasible point
x1=pinv(A)*b
maxIter=3000


function proj_sub_method_l1norm(
    A::Array{Float64,2},
    b::Array{Float64,2},
    x1::Array{Float64,2},
    a::Float64,
    maxIter::Int64
    )
fSet=Float64[] # We are creating an empty array where we will store the objective values as we progress
push!(fSet,Inf) # Intial value of f is infinity

fBest=Float64[] # It is an array where the last element will give f_best^{(k)} so far
push!(fBest, Inf) # Intial best value of f is infinity

k=1
x=x1
while(k < maxIter)
println("Iteration number=",k)
# Value of the function at kth iteration
fKval=norm(x,1)
# Subgradient of ||x||₁ is sign(x)=(sign(x₁),...,sign(xₙ))
g=(x.>0)-(x.<0)

#step size selection
α=a/k

# Push the current value of f(x^(k)) in fSet
push!(fSet,fKval)

# Note how we are pushing the best f found so far in the last element of f
push!(fBest, min(fBest[end],fKval))
# Subgradient update:
x=x-α*(eye(n)-A'*inv(A*A')*A)*g
k=k+1
end # while
return x, fSet, fBest
end

# Testing if the function is working:

(x, fSet, fBest)=proj_sub_method_l1norm(A, b, x1, 0.1 , maxIter)

using Winston
iters=[1:maxIter]
figProjSub = FramedPlot(
title="The value of f_{best}^{(k)} - f^{*} vs. iteration number",
         xlabel="k",
         ylabel="f_{best}^{(k)} - f^{*}",
yrange=(1e-2,1e0),
ylog=true
)
xAxis=iters
curve1=fBest-fMinValue
bl=Curve(xAxis, curve1, color="blue")
setattr(bl, label="0.1/k")
le=Legend(.8,.95,{bl})
add(figProjSub,bl)

# Data generation
n=50
A=randn(n,n)
P= A*A' # Enforcing Cholesky decomposition thus making P positive definite
eigsP=eigvals(P)
println("P is a positive definite matrix with max eigenvalue ", maximum(eigsP), "and minimum eigen value ", minimum(eigsP))
q=randn(n,1)

# Solve the problem using Convex.jl
using Convex

using SCS

x̄=Variable(n)

obj=(0.5*quad_form(x̄, P))- dot(q,x̄)# quad_form(α, M):=αᵀMα in Convex
# For a detailed list of suppored operations see http://convexjl.readthedocs.org/en/latest/operations.html

constraints=[]# Declaring an empty vecotr of constraints first

# Let us add the constraints x[i]^2 in a loop...
for i in 1:n
  constraints += (x̄[i]^2 <= 1)
end

# Define the problem in
problem=
minimize( obj,
# subject to
constraints
)

# Let's solve it:
solve!(problem,SCSSolver())

# Let's look at the output

print("The status of the solution is ", problem.status)

xMinValue=x̄.value
fMinValue=problem.optval

println( "\nwith optimal objective value, f(x*)=", fMinValue,
"\noptimal solution is x*=", xMinValue)

# Projected subgradient algorithm for primal problem
println("Projected subgradient algorithm for primal problem")

fPrmlSet=Float64[] # We are creating an empty array where we will store the primal objective values as we progress
push!(fPrmlSet,Inf) # Intial value of fPrimal is infinity

fPrmlBest=Float64[] # It is an array where the last element will give f_best^{(k)} so far for the primal objective
push!(fPrmlBest, Inf) # Intial best value of fPrimal is infinity
x=zeros(n,1) # intial starting point is taken to be zero

k=1 # k is the iteration counter, starting at 1

maxIter = 3000 # Maximum number of iterations

while(k < maxIter)

# Evaluate primal objective value
fKval=((0.5*x'*P*x)-q'*x)[1,1]# Julia treats the quantity as 1x1 matrix, so I am taking the [1,1] th element the number itself!

# Calculating the subgradient for the primal function which is essentially the gradient
gK=(P*x)-q

# Step size selcetion: Consider Polyak's step size
#αK=(fKval-fMinValue)/norm(gK)^2
#αK=0.001
αk=1/k

# Push the current value of f(x^(k)) in fSet
push!(fPrmlSet,fKval)

# Note how we are pushing the best f primal found so far in the last element of f
push!(fPrmlBest, min(fPrmlBest[end],fKval))

# Projected subgradient alg
x=x-αK*gK # Find the subgradient update
x=max(min(x,1),-1) # Project it on the set {x: ∀i xᵢ²<=1}
# min(x,1) compares each element of x with 1 and takes the minimum, and max(x,-1)  compares each element of x with -1 and takes the maximum
k=k+1
end # while

x
fPrmlBest

#Subgradient method for the dual problem
println("Subgradient method applied to the dual problem")

gDualSet=Float64[]

push!(gDualSet,-Inf)

gDualBest=Float64[]

push!(gDualBest,-Inf)

λ=ones(n)# Initial dual variable

k=1 # Iteration counter

while (k < maxIter)
# Calculating subgradient
# Calculating x^{(k)}
xStar=inv(P+diagm(vec(2*λ)))*q
#xStar=max(min(xStar,1),-1)
h=xStar.^2-1

# Calculating dual objective value
gKval=(-0.5*q'*xStar-sum(λ))[1,1]
push!(gDualSet,gKval)

push!(gDualBest, max(gKval,gDualBest[end]))
# Calculating primal objective value

# step size selection
sK=1/k

λ=max(0,λ+sK*h)

k=k+1
end

# Let's plot how both of the projected subgradient methods work
using Gadfly

curvePrimal=abs((fPrmlBest-fMinValue)/(fMinValue))*100
curveDual=abs((gDualBest-fMinValue)/fMinValue)*100

iters=1:maxIter
plot(
  layer(x=iters,y=curvePrimal,Geom.line, Theme(default_color=color("red"))),
        layer(x=iters, y=curveDual,Geom.line,Theme(default_color=color("green")))
  )

curvePrimal

fPrmlBest


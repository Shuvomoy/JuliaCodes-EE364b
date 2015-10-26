1
# Alternating subgradient method
n=20

m=100

A=randn(m,n)

b=abs(randn(m,1)) # b is made non-negative, so that 0 is a feasible point

c=-A'*abs(randn(m,1))
# This step deserves some comments.
# From linear programming we know that, an LP described by
# minimize c'x
# subject to Ax<=b
# is unbounded if and only if there exists an extreme ray d such that c'd<0
# Now the set of the recession cone (set of all rays) of {x |Ax<=b} is of the form
# {d | Ad <= 0}, as the set of extreme rays \subset {d | Ad <= 0}, so any extreme ray
# will satisfy Ad=-q, where q>=0, now if we take c=-A'p, where p>=0, then
# c'd=(-A'p)'*(d)=-p'Ad=(-p')*(-q)=pq>=0 as inner product of two nonnegative vectors
# is always positive, thus having c=-A'p will prevent unbounded below situation, and by
# picking b nonnegative we have ensured that the problem is feasible, so combining these two will
#result in a finite optimum.

#Solve the LP using Convex

using Convex

using SCS

xCvx=Variable(n)

problem=minimize(    c'*xCvx,
        #subject to
                    A*xCvx<=b)

solve!(problem,SCSSolver())

print(
"The status of the solution is ", problem.status,
"\nwith optimal objective value, f(x*)=", problem.optval,
)
xMinValue=x.value
fMinValue=problem.optval


# Function that will implement projected subgradient method
# ---------------------------------------------------------
function alternate_subgradient_ineq_LP(
    A::Array{Float64,2},
    b::Array{Float64,2},
    c::Array{Float64,2},
    x1::Array{Float64,2},
    maxIter::Int64
    )
fSet=Float64[] # We are creating an empty array where we will store the objective values as we progress
push!(fSet,Inf) # Intial value of f is infinity

fBest=Float64[] # It is an array where the last element will give f_best^{(k)} so far
push!(fBest, Inf) # Intial best value of f is infinity

k=1
x=x1

while(k < maxIter)
    # Print the current iteration

# Checking the feasibility of the current iterate
constraintVector=A*x-b
h=maximum(constraintVector)
ind=indmax(constraintVector)

# Determining the subgradient, updating fSet and fBest, and determining the step size depending on the feasibility of the current iterate
if(h>0)# infeasible
  g=A[ind,:]'
  α=(h+ϵ)/(norm(g)^2)
  push!(fSet,Inf)
  push!(fBest,fBest[end])# Best f value will not change
else #feasible
  g=c
  fKval=(c'*x)[1,1]
  α=(fKval-fMinValue)/(norm(g)^2)
  push!(fSet,fKval)
  push!(fBest, min(fBest[end],fKval))
end

# updating subgradient
x=x-α*g
if (mod(k,100)==0)
  println("At iteration number=",k)
  println("Current best value of f=", fBest[end])
end
k=k+1
end # while
return x, fSet, fBest
end # function

 # This will be our initial feasible point to start the alternating subgradient method
x1=zeros(n,1)
ϵ=1e-3
maxIter=3000
typeof(x1)

(x2,f2,fBest2)=alternate_subgradient_ineq_LP(A,b,c,x1,maxIter)
A
b
c
typeof(maxIter)


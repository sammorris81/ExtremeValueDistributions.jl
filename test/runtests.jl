using ExtremeValueDistributions
using Base.Test

tests = ["generalizedextreme.jl", "generalizedpareto.jl"]

print_with_color(:blue, "Running tests:\n")

srand(345678)

for t in tests
    test_fn = "$t.jl"
    print_with_color(:green, "* $test_fn\n")
    include(test_fn)
end

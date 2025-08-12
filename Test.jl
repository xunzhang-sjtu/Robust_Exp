# using LinearAlgebra
# using Distributions
# using Optim
# using Random
# using StatsFuns
# using DataFrames
# using StatsBase
# include("Estimate.jl")
# # 设置随机种子
# Random.seed!(1)


# # 模型设置
# N = 3           # 选项数（不含baseline）
# d_u = 1         # u 的维度
# d_p = N         # p 的维度
# S = 5000         # 样本数量


# a_star = [rand(Uniform(-1, 1), d_u) for n in 1:N]
# b_star = [rand(Uniform(-1, 1), d_p) for n in 1:N]
# a_star = [round.(a; digits=2) for a in a_star]
# b_star = [round.(b; digits=2) for b in b_star]
# println("a_star: ", a_star) 
# println("b_star: ", b_star) 

# # Step 2: 生成样本 (u, p)，标准正态分布
# U_train = [randn(d_u) for s in 1:S];
# P_train = [randn(d_p) for s in 1:S];

# Estimate_MNL_Para(U_train, P_train, S, N, d_u, d_p, a_star, b_star)


println("==== COPT 环境检查脚本 ====")

# 1. 检查 COPT_LICENSE_FILE 环境变量
if haskey(ENV, "COPT_LICENSE_FILE")
    println("✅ COPT_LICENSE_FILE 已设置: ", ENV["COPT_LICENSE_FILE"])
else
    println("⚠️ COPT_LICENSE_FILE 未设置")
end

# 2. 检查动态库搜索路径（DYLD_LIBRARY_PATH）
if haskey(ENV, "DYLD_LIBRARY_PATH")
    println("✅ DYLD_LIBRARY_PATH 已设置: ", ENV["DYLD_LIBRARY_PATH"])
else
    println("⚠️ DYLD_LIBRARY_PATH 未设置，可能找不到 libcopt.dylib")
end

# 3. 检查 PATH 中是否包含 COPT 可执行文件路径
if occursin("copt", join(split(ENV["PATH"], ":"), "\n"))
    println("✅ PATH 中包含 copt 目录")
else
    println("⚠️ PATH 中未包含 copt 可执行文件所在目录")
end

# 4. 测试终端是否能运行 copt
println("==== 尝试运行 `copt -v` 查看版本 ====")
try
    run(`copt -v`)
catch e
    println("❌ 无法运行 `copt -v`，错误信息: ", e)
end

# 5. 测试是否能加载 COPT 动态库
println("==== 测试 Julia 能否加载 libcopt.dylib ====")
try
    Libdl.dlopen("libcopt.dylib")
    println("✅ 成功加载 libcopt.dylib")
catch e
    println("❌ 无法加载 libcopt.dylib: ", e)
end

# 6. 测试 JuMP + COPT
println("==== 测试 JuMP 调用 COPT ====")
try
    using JuMP, COPT
    model = Model(COPT.Optimizer)
    @variable(model, x >= 0)
    @objective(model, Max, x)
    optimize!(model)
    println("✅ JuMP 成功调用 COPT")
catch e
    println("❌ JuMP 调用 COPT 失败: ", e)
end

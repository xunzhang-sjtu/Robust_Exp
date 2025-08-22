using LinearAlgebra
using Distributions
using StatsPlots   # 提供 boxplot，基于 Plots
using Plots
using DataFrames, Colors


function plot_boxplot(RO_coef_chosen,Rev_ETO,Rev_RO)
    data = [Rev_ETO, [Rev_RO["RO_coef=$(RO_coef)"] for RO_coef in RO_coef_chosen]...]
    labels = ["ETO"; ["RO_$(RO_coef)" for RO_coef in RO_coef_chosen]]
    # --- 1) 展开成长表 ---
    df = DataFrame(
        value = vcat(data...),
        group = repeat(labels, inner = length(data[1]))
    )

    mycolors = palette(:tab10)[1:length(labels)]   # tab10 最多10个颜色

    # --- 3) 绘制箱线图 ---
    @df df boxplot(:group, :value;
        group = :group,
        palette = mycolors,
        legend = false,
        ylabel = "Revenue",
        # title = "Revenue Distribution"
    )

    # --- 4) 计算均值 ---
    means = combine(groupby(df, :group), :value => mean => :mean_val)

    for i in 1:length(labels)
        scatter!([labels[i]], [means.mean_val[i]];
            color = :red,
            marker = (:star, 10),
            label = "")
    end

    # # --- 6) 绘制均值连线（这里必须把类别转为数值 1:N） ---
    plot!(labels, means.mean_val;
        seriestype = :line,
        color = :red,
        lw = 2,
        ls = :dash,
        label = "Mean trend"
    )
end
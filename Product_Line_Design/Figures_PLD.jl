function line_plot_RPLD_vs_ETOPLD(profit_ETO,profit_RO,gamma_list,include_std,fig_name,is_display)
    """
    profit comparison between RPLD and ETOPLD
    """
    RO_ETO_Ratio_Mean = (mean(profit_RO,dims=1)/mean(profit_ETO))[1,:];
    RO_ETO_Ratio_Std = (std(profit_RO,dims=1)/std(profit_ETO))[1,:];

    end_index = length(gamma_list)
    plot(gamma_list[1:end_index], RO_ETO_Ratio_Mean[1:end_index], marker=:o, xlabel=L"\gamma", label="Average Profit", xticks=(gamma_list[1:end_index], string.(gamma_list[1:end_index])))
    if include_std
        plot!(gamma_list[1:end_index], RO_ETO_Ratio_Std[1:end_index], marker=:o, xlabel=L"\gamma", ylabel="RPD/ETOPD", label="Std of Profit")
    end
    hline!([1.0], linestyle=:dash, color=:red, label="")
    if is_display
        display(current())
    end
    savefig(fig_name)
end


function boxplot_RPLD_vs_ETOPLD(data,labels,fig_name)
    """
    Boxplot comparison between RPLD and ETOPLD
    """
    # --- 1) 展开成长表 ---
    df = DataFrame(
        value = vcat(data...),
        group = repeat(labels, inner = length(data[1]))
    )

    mycolors = ["#f47e20","#f59898","#1f78b4","#329f48","#6b3e97","#f47e20","#d2ab82"]
    # --- 3) 绘制箱线图 ---
    @df df boxplot(:group, :value;
        group = :group,
        palette = mycolors,
        legend = false,
        ylabel = "Normalized Revenue",
        outliers = false,
        # title = "Revenue Distribution"
    )

    # --- 4) 计算均值 ---
    means = combine(groupby(df, :group), :value => mean => :mean_val)

    # for i in 1:length(labels)
    for i in eachindex(labels)
        scatter!([labels[i]], [means.mean_val[i]];
            color = :red,
            marker = (:star, 10),
            label = "")
    end

    # # # --- 6) 绘制均值连线（这里必须把类别转为数值 1:N） ---
    # plot!(labels, means.mean_val;
    #     seriestype = :line,
    #     color = :red,
    #     lw = 2,
    #     ls = :dash,
    #     label = "Mean trend"
    # )
    display(current())
    savefig(fig_name)
end

function hist_profit_distribution(profit_ETO, profit_RO, gamma_index,gamma_list,fig_name)
    gamma = gamma_list[gamma_index]
    """
    Plot histograms of profit distributions for ETO and RO methods.
    """
    mycolors = ["#f47e20","#f59898","#1f78b4","#329f48","#6b3e97","#f47e20","#d2ab82"]
    histogram(profit_ETO, color = mycolors[1],alpha=0.5, label="ETO", xlabel="Profit", ylabel="Frequency", nbins=20)
    histogram!(profit_RO[:,gamma_index], color = mycolors[4],alpha=0.75, label="RO ($gamma)", nbins=20)
    display(current())
    savefig(fig_name)
end

# RO_ETO_Ratio_Mean = (mean(profit_RO,dims=1)/mean(profit_ETO))[1,:];
# RO_ETO_Ratio_Std = (std(profit_RO,dims=1)/std(profit_ETO))[1,:];

# end_index = length(gamma_list)
# xvals = gamma_list[1:end_index]

# # 计算25%和75%分位点
# RO_ETO_Ratio_Q25 = [quantile(profit_RO[:,i] ./ profit_ETO, 0.25) for i in 1:end_index]
# RO_ETO_Ratio_Q75 = [quantile(profit_RO[:,i] ./ profit_ETO, 0.75) for i in 1:end_index]

# plot(xvals, RO_ETO_Ratio_Mean[1:end_index], marker=:o, xlabel=L"\gamma", label="Average Profit", xticks=(xvals, string.(xvals)))
# plot!(xvals, RO_ETO_Ratio_Std[1:end_index], marker=:o, xlabel=L"\gamma", ylabel="RPD/ETOPD", label="Std of Profit")
# plot!(xvals, RO_ETO_Ratio_Q25, marker=:diamond, linestyle=:dash, color=:green, label="25% Quantile")
# plot!(xvals, RO_ETO_Ratio_Q75, marker=:diamond, linestyle=:dash, color=:orange, label="75% Quantile")
# hline!([1.0], linestyle=:dash, color=:red, label="")
# display(current())
# # savefig(string(data_dir, "RPLD_vs_ETOPLD_lambda=$lambda.pdf"))

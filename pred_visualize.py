


def plot_matchup_matrix(proj, num_games = 3, savename="ideal.png", matchup_df = None, week=""):
    """
    Generates a matrix of predicted outcomes if every player played every other player

    Args:
        proj (dict): dictionary of manager name to projected stat values
        num_games (int, optional): _description_. Defaults to 3.
        savename (str, optional): _description_. Defaults to "ideal.png".
        matchup_df (pd dataframe, optional): 

    Returns:
        _type_: _description_
    """

    if matchup_df is None:
        managers = list(proj.keys())
    else:
        managers = matchup_df['manager'].values

    probMat = np.zeros((len(managers), len(managers)))
    for i1, m1 in enumerate(managers):
        for i2, m2 in enumerate(managers):
            if i2 > i1:
                p, s, m = prob_victory(proj, m1, m2, matchup_df=matchup_df)
                probMat[i1, i2] = p[0]
                probMat[i2, i1] = p[1]
            elif i2 == i1:
                probMat[i1, i2] = np.nan


    

    # create labels for the axes
    if matchup_df is None:
        yAxisLabels = managers
    else:
        yAxisLabels = matchup_df[['manager', 'teamName']].apply(lambda x: x[0] + '\n' + x[1],axis=1)

    xAxisLabels = managers

    # do plotting
    sn.set(font_scale=1.2)
    f, ax = plt.subplots(figsize=(20,10))
    ax = sn.heatmap(probMat, annot=np.round(probMat*100)/100, fmt='', xticklabels = xAxisLabels,
            yticklabels = yAxisLabels, cmap='RdYlGn',cbar=False)

    # highlight actual matchup
    if not matchup_df is None:
        # add in patches to mark who actually played who in that week
        # get number of unique matchups:
        for m in matchup_df['matchupNumber'].unique():
            i,j = matchup_df[matchup_df['matchupNumber']==m].index
            ax.add_patch(Rectangle((i,j), 1, 1, fill=False, edgecolor='blue', lw=3))
            ax.add_patch(Rectangle((j,i), 1, 1, fill=False, edgecolor='blue', lw=3))

    if num_games is None:
        if week == "":
            f.suptitle(f"NBA Fantasy Predicted Results", fontsize = 30)
        else:
            f.suptitle(f"NBA Fantasy Predicted Results (Week {week})", fontsize = 30)
    else:
        f.suptitle(f"NBA Fantasy Ideal Matchups (All Players Play {num_games} Games, Ignore Injuries, Don't Count Players on IL)", fontsize = 30)

    if savename != "":
        plt.savefig(savename)
        plt.close(f)

    return probMat

def plot_matchup_summary(proj, p1, p2, matchup_df = None, savename=None):
    """
    Plots a summary of the specified matchup, showing the 90% confidence interval
    for each stat, showing the probability that either player will 
    win the category.


    Args:
        proj (dict): dictionary that maps player to projected stats
        p1 (str): name of the first player
        p2 (str): name of the second player
        matchup_df (pandas df, optional): matchup df with the current score for midweek analysis. Defaults to None.
    """

    p,s,m = prob_victory(proj, p1, p2, matchup_df=matchup_df)
    
    f, ax = plt.subplots(nrows=3,ncols=3, figsize=(20,14))

    vic = [np.round(x*100,2) for x in p]
    f.suptitle(f"Probability of Victory - {p1}: {vic[0]}%, {p2}: {vic[1]}%, Tie: {vic[2]}%", fontsize=26)

    # epsilon = 0.0001
    epsilon = 0.05

    if not matchup_df is None:
        grouped = matchup_df.groupby("manager")

    
   
    for ip, p in enumerate((p1,p2)):
        for i, stat in enumerate(s):
            if "%" in stat:
                mu = m[stat][p][0]
                sigma = m[stat][p][1]
                x = (norm.ppf(epsilon, loc=mu, scale=sigma),
                    norm.ppf(1-epsilon,loc=mu, scale=sigma))
                # prob = norm.pmf(x, loc=mu, scale=sigma)
            else:
                mu = proj[p][stat]
                x = np.arange(poisson.ppf(epsilon, mu),
                    poisson.ppf(1-epsilon, mu)+1)
                # prob = poisson.pmf(x, mu)
            row = int(i/3)
            col = i-int(i/3)*3
            a = ax[row,col]

            if ip == 0:
                vic = np.array([np.round(x*100,2) for x in s[stat]])
                a.set_title(f"{stat} - {p1}: {vic[0]}%, {p2}: {vic[1]}%, Tie: {vic[2]}%")
            # Add current stats
            exp_val = mu
            if "%" not in stat and not matchup_df is None:
                exp_val+=grouped.get_group(p)[stat].iloc[0]
            a.errorbar(exp_val, (1-ip)*0.01, xerr=np.array((mu-x[0], x[-1]-mu)).reshape(2,1), label=p, capsize = 4, fmt = 'o', alpha=0.75)
            a.set_ylim([-0.005,0.015])
            a.set_yticks([])
            if ip == 1:
                a.legend()
    
    if savename is None:
        savename = f"{p1}_vs_{p2}.png"
    plt.savefig(savename)

    return

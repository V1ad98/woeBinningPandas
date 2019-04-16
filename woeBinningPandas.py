import pandas as pd
import numpy as np
import math
import warnings
import copy

germancredit = pd.read_csv('GermanCredit.csv')
df = germancredit[['credit_risk', 'amount','duration','savings','purpose']]

def woe_binning_2 (df, target_var, pred_var, min_perc_total, min_perc_class, stop_limit, abbrev_fact_levels, bad, good):
    cutpoints_backup = False
    stop_limit_exceeded = False
    list_level_a_collected = False
    iv_total_collect = pd.DataFrame([[np.nan]], columns=['iv_total_collect'])
    
    #### Build subsets with target and predictor variable
    df = df[[target_var, pred_var]] # used for final binning
    dfrm = df[[target_var, pred_var]]# used for iterative merging of bins
    dfrm.columns = ['target_var','predictor_var']

    #### Check if numerical variable or factor was provided as predictor and apply appropriate binning technique

    ### Binning in case a numerical variable was selected
    if len(dfrm.iloc[:,0].drop_duplicates()) == 2 and (dfrm.iloc[:,1].dtypes.kind in 'bifc') == True:
        

        ## Derive number of initial bins from min.perc.total parameter
        max_bins = math.trunc(1/min_perc_total)
        
        ## Derive cutpoints for bins (with similar frequency)
        cutpoints = dfrm.predictor_var.quantile(np.arange(0,max_bins+1)/max_bins).reset_index(drop=True)
        innercutpoints = [-np.inf] + list(cutpoints[1:len(cutpoints)-1]) + [np.inf]  # add -Inf, +Inf to cutpoints
        cutpoints = list(dict.fromkeys(innercutpoints))
        
        ## Calculate initial crosstab from binned variable and target variable
        ## to identify and merge sparse bins
        
        
        # Compute binned variable from cutpoints and add it to the subset data frame
        dfrm["predictor_var_binned"] = pd.cut(dfrm["predictor_var"], cutpoints, right=True, labels = None,
            retbins=False, precision=10, include_lowest=False)
            
            
        # Compute crosstab from binned variable and target variable and covert it to a data frame
        freq_table = pd.crosstab(dfrm["predictor_var_binned"],dfrm["target_var"], dropna=True)
        freq_table = freq_table.reset_index(drop=False)
        missing=pd.DataFrame({'predictor_var_binned': ["Missing"],
                              0.0: [dfrm.isnull().sum(axis = 0)[0]],
                              1.0: [dfrm.isnull().sum(axis = 0)[1]]})
        freq_table =freq_table.append(missing,ignore_index=True, sort=False)
        woe_dfrm = pd.DataFrame(freq_table) # Convert frequency table to data frame
        woe_dfrm = woe_dfrm.set_index(["predictor_var_binned"])
        
        # Compute columns percents for target classes from crosstab frequencies
        woe_dfrm["col_perc_a"] = woe_dfrm[good]/sum(woe_dfrm[good])
        woe_dfrm["col_perc_b"] = woe_dfrm[bad]/sum(woe_dfrm[bad])
        
        # Correct column percents in case of 0 frequencies (in case of no NA skip last row)
        if df.iloc[:,1].isnull().values.any()==False:
            if min(woe_dfrm.iloc[:,0].drop(woe_dfrm.index[len(woe_dfrm)-1]))==0 or min(woe_dfrm.iloc[:,1].drop(woe_dfrm.index[len(woe_dfrm)-1]))==0:
                woe_dfrm["col_perc_a"] = (woe_dfrm["col_perc_a"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)/sum(woe_dfrm["col_perc_a"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)
                woe_dfrm["col_perc_b"] = (woe_dfrm["col_perc_b"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)/sum(woe_dfrm["col_perc_b"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)
                        
        else:
            if min(woe_dfrm.iloc[:,0])==0 or min(woe_dfrm.iloc[:,1])==0:
                woe_dfrm["col_perc_a"] = (woe_dfrm["col_perc_a"] + 0.0001)/sum(woe_dfrm["col_perc_a"] + 0.0001)
                woe_dfrm["col_perc_b"] = (woe_dfrm["col_perc_b"] + 0.0001)/sum(woe_dfrm["col_perc_b"] + 0.0001)

        # Check for bins (without last regular and without NA bin) if frequencies < percentage limit specified above
    # (in reverse order to remain correct reference to cutpoints)
        for i in reversed(range(0,len(woe_dfrm.index)-2)):
            if woe_dfrm["col_perc_a"].iloc[i]<min_perc_class or woe_dfrm["col_perc_b"].iloc[i]<min_perc_class or (woe_dfrm.iloc[i][0] + woe_dfrm.iloc[i][1])/(sum(woe_dfrm[0]) + sum(woe_dfrm[1]))<min_perc_total:
                # Remove cutpoint 
                del cutpoints[i+1]
                # Compute binned variable from cutpoints and add it to the subset data frame
                dfrm["predictor_var_binned"] = pd.cut(dfrm["predictor_var"], cutpoints, right=True, labels = None,
                    retbins=False, precision=10, include_lowest=False)
                # Compute crosstab from binned variable and target variable and covert it to a data frame   
                freq_table = pd.crosstab(dfrm["predictor_var_binned"],dfrm["target_var"], dropna=True)
                freq_table = freq_table.reset_index(drop=False)
                missing=pd.DataFrame({'predictor_var_binned': ["Missing"],
                                      0.0: [dfrm.isnull().sum(axis = 0)[0]],
                                      1.0: [dfrm.isnull().sum(axis = 0)[1]]})
                freq_table =freq_table.append(missing,ignore_index=True, sort=False)
                woe_dfrm = pd.DataFrame(freq_table) # Convert frequency table to data frame
                woe_dfrm = woe_dfrm.set_index(["predictor_var_binned"])
                # Compute columns percents for target classes from crosstab frequencies
                woe_dfrm["col_perc_a"] = woe_dfrm[good]/sum(woe_dfrm[good])
                woe_dfrm["col_perc_b"] = woe_dfrm[bad]/sum(woe_dfrm[bad])
                # Correct column percents in case of 0 frequencies (in case of no NA skip last row)
                if df.iloc[:,1].isnull().values.any()==False:
                    if min(woe_dfrm.iloc[:,0].drop(woe_dfrm.index[len(woe_dfrm)-1]))==0 or min(woe_dfrm.iloc[:,1].drop(woe_dfrm.index[len(woe_dfrm)-1]))==0:
                        woe_dfrm["col_perc_a"] = (woe_dfrm["col_perc_a"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)/sum(woe_dfrm["col_perc_a"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)
                        woe_dfrm["col_perc_b"] = (woe_dfrm["col_perc_b"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)/sum(woe_dfrm["col_perc_b"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)
                                
                else:
                    if min(woe_dfrm.iloc[:,0])==0 or min(woe_dfrm.iloc[:,1])==0:
                        woe_dfrm["col_perc_a"] = (woe_dfrm["col_perc_a"] + 0.0001)/sum(woe_dfrm["col_perc_a"] + 0.0001)
                        woe_dfrm["col_perc_b"] = (woe_dfrm["col_perc_b"] + 0.0001)/sum(woe_dfrm["col_perc_b"] + 0.0001)
            # Stop in case 3 cutpoints (-Inf, x, +Inf) are reached
            if len(cutpoints)==3:
                break
          
            # Check for last regular bin if frequencies < percentage limit specified above (only in case number of cutpoints > 3
        if len(cutpoints)>3:
            if woe_dfrm["col_perc_a"][len(woe_dfrm.index)-2]<min_perc_class or woe_dfrm["col_perc_b"][len(woe_dfrm.index)-2]<min_perc_class or (woe_dfrm.iloc[len(woe_dfrm.index)-2,0] + woe_dfrm.iloc[len(woe_dfrm.index)-2,1])/(sum(woe_dfrm[0])+sum(woe_dfrm[1]))<min_perc_total:
                # Remove cutpoint
                del cutpoints[len(woe_dfrm.index)-2]
                # Compute binned variable from cutpoints and add it to the subset data frame
                dfrm["predictor_var_binned"] = pd.cut(dfrm["predictor_var"], cutpoints, right=True, labels = None,
                    retbins=False, precision=10, include_lowest=False)
                # Compute crosstab from binned variable and target variable and covert it to a data frame 
                freq_table = pd.crosstab(dfrm["predictor_var_binned"],dfrm["target_var"], dropna=True)
                freq_table = freq_table.reset_index(drop=False)
                missing=pd.DataFrame({'predictor_var_binned': ["Missing"],
                                      0.0: [dfrm.isnull().sum(axis = 0)[0]],
                                      1.0: [dfrm.isnull().sum(axis = 0)[1]]})
                freq_table =freq_table.append(missing,ignore_index=True, sort=False)
                woe_dfrm = pd.DataFrame(freq_table) # Convert frequency table to data frame
                woe_dfrm = woe_dfrm.set_index(["predictor_var_binned"])
                #woe_dfrm = woe_dfrm[['good', 'bad']]  # Select columns with raw frequencies only
                # Compute columns percents for target classes from crosstab frequencies
                woe_dfrm["col_perc_a"] = woe_dfrm[good]/sum(woe_dfrm[good])
                woe_dfrm["col_perc_b"] = woe_dfrm[bad]/sum(woe_dfrm[bad])
                # Correct column percents in case of 0 frequencies (in case of no NA skip last row)
                if df.iloc[:,1].isnull().values.any()==False:
                    if min(woe_dfrm.iloc[:,0].drop(woe_dfrm.index[len(woe_dfrm)-1]))==0 or min(woe_dfrm.iloc[:,1].drop(woe_dfrm.index[len(woe_dfrm)-1]))==0:
                        woe_dfrm["col_perc_a"] = (woe_dfrm["col_perc_a"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)/sum(woe_dfrm["col_perc_a"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)
                        woe_dfrm["col_perc_b"] = (woe_dfrm["col_perc_b"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)/sum(woe_dfrm["col_perc_b"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)
                                
                else:
                    if min(woe_dfrm.iloc[:,0])==0 or min(woe_dfrm.iloc[:,1])==0:
                        woe_dfrm["col_perc_a"] = (woe_dfrm["col_perc_a"] + 0.0001)/sum(woe_dfrm["col_perc_a"] + 0.0001)
                        woe_dfrm["col_perc_b"] = (woe_dfrm["col_perc_b"] + 0.0001)/sum(woe_dfrm["col_perc_b"] + 0.0001)
                        
        ## After sparse bins are merged:
        ## Merge bins with similar WOE values and calculate corresponding WOE table and IV step by step
        ## until 2 bins are left (i.e. 3 cutpoints: -Inf, middle cutpoint, +Inf)
        while len(cutpoints)>2:
    
            # Compute binned variable from cutpoints and add it to the subset data frame
            dfrm["predictor_var_binned"] = pd.cut(dfrm["predictor_var"], cutpoints, right=True, labels = None,
                    retbins=False, precision=10, include_lowest=False)
            
            # Compute crosstab from binned variable and target variable and covert it to a data frame
            freq_table = pd.crosstab(dfrm["predictor_var_binned"],dfrm["target_var"], dropna=True)
            freq_table = freq_table.reset_index(drop=False)
            missing=pd.DataFrame({'predictor_var_binned': ["Missing"],
                                  0.0: [dfrm.isnull().sum(axis = 0)[0]],
                                  1.0: [dfrm.isnull().sum(axis = 0)[1]]})
            freq_table =freq_table.append(missing,ignore_index=True, sort=False)
            woe_dfrm = pd.DataFrame(freq_table) # Convert frequency table to data frame
            woe_dfrm = woe_dfrm.set_index(["predictor_var_binned"])
            
            # Compute columns percents for target classes from crosstab frequencies
            woe_dfrm["col_perc_a"] = woe_dfrm[good]/sum(woe_dfrm[good])
            woe_dfrm["col_perc_b"] = woe_dfrm[bad]/sum(woe_dfrm[bad])
            # Correct column percents in case of 0 frequencies (in case of no NA skip last row)
            if df.iloc[:,1].isnull().values.any()==False:
                if min(woe_dfrm.iloc[:,0].drop(woe_dfrm.index[len(woe_dfrm)-1]))==0 or min(woe_dfrm.iloc[:,1].drop(woe_dfrm.index[len(woe_dfrm)-1]))==0:
                    woe_dfrm["col_perc_a"] = (woe_dfrm["col_perc_a"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)/sum(woe_dfrm["col_perc_a"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)
                    woe_dfrm["col_perc_b"] = (woe_dfrm["col_perc_b"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)/sum(woe_dfrm["col_perc_b"].drop(woe_dfrm.index[len(woe_dfrm)-1]) + 0.0001)
                                
            else:
                if min(woe_dfrm.iloc[:,0])==0 or min(woe_dfrm.iloc[:,1])==0:
                    woe_dfrm["col_perc_a"] = (woe_dfrm["col_perc_a"] + 0.0001)/sum(woe_dfrm["col_perc_a"] + 0.0001)
                    woe_dfrm["col_perc_b"] = (woe_dfrm["col_perc_b"] + 0.0001)/sum(woe_dfrm["col_perc_b"] + 0.0001)   

            woe_dfrm["woe"] = 100*np.log(woe_dfrm["col_perc_a"]/woe_dfrm["col_perc_b"])
    
            woe_dfrm_list = list(woe_dfrm["woe"]) #dataframe to list
            woe_lag = list(woe_dfrm["woe"]) #dataframe to list
            
            woe_lag.insert(0, np.nan)  #add Nan [0]
            woe_lag.pop(len(woe_lag)-1) #delete Nan
            
            woe_dfrm_list=pd.DataFrame(woe_dfrm_list) #list to dataframe
            woe_lag= pd.DataFrame(woe_lag) #list to dataframe
            
            woe_diff = (woe_dfrm_list-woe_lag).abs()
            
            woe_dfrm["woe_lag"] = list(woe_lag.iloc[:,0]) #add column woe_lag to woe_dfrm
            woe_dfrm["woe_diff"] = list(woe_diff.iloc[:,0]) #add column woe_diff to woe_dfrm
            
            woe_dfrm["iv_bins"] = (woe_dfrm["col_perc_a"]-woe_dfrm["col_perc_b"])*woe_dfrm["woe"]/100
            
            # Calculate total IV for current binning
            iv_total = sum(woe_dfrm.fillna(0)['iv_bins'])
            iv_total = pd.DataFrame([[iv_total]], columns=['iv_total'])
            
            # Collect total IVs for different binning solutions
            if np.isnan(iv_total_collect.iloc[0][0])==False:
                iv_total_collect = pd.concat([iv_total_collect, iv_total], axis=1)
            else:
                iv_total_collect.iloc[0][0] = iv_total.iloc[0][0]
            
            # In case IV decreases by more than percentage specified by stop.limit parameter above
            # restore former binning solution (cutpoints) and leave loop
            if len(iv_total_collect.columns)>1:
                actual_iv_decrease = ((iv_total_collect.iloc[:,len(iv_total_collect.columns)-2])-(iv_total_collect.iloc[:,len(iv_total_collect.columns)-1]))/(iv_total_collect.iloc[:,len(iv_total_collect.columns)-2])
        
                if actual_iv_decrease.iloc[0] > stop_limit and stop_limit_exceeded == False:
                    cutpoints_final = cutpoints_backup
                    woe_dfrm_final = woe_dfrm_backup
                    stop_limit_exceeded = True   # indicates that stop limit is exceeded to prevent overriding the final solution
            
            # Save first cutpoint solution and corresponding WOE values as final solution (is used in case no WOE merging will be applied)
            if cutpoints_backup == False:
                cutpoints_final = cutpoints
                woe_dfrm_final = woe_dfrm

            # Saves binning solution after last merging step in case the IV stop limit was not exceeded
            if stop_limit_exceeded == False and len(cutpoints)==3:
                cutpoints_final = cutpoints
                woe_dfrm_final = woe_dfrm

            # Save backups of current cutpoints and corresponding WOE values before merging to be able to retrieve solution in case IV decrease is too strong
            cutpoints_backup = copy.deepcopy(cutpoints)
            woe_dfrm_backup = woe_dfrm
    
            # Determine the index of the minimum WOE difference between adjacent bins and
            # merge bins with minimum WOE difference (apart from the last 'Missing' bin)    
            min_woe_diff = woe_dfrm["woe_diff"].drop(woe_dfrm.index[len(woe_dfrm)-1])==min(woe_dfrm["woe_diff"].drop(woe_dfrm.index[len(woe_dfrm)-1]).fillna(+np.inf))
            min_woe_diff = min_woe_diff.reset_index(drop=True).index[min_woe_diff == True].tolist()
            min_woe_diff = min_woe_diff[0]
            del cutpoints[min_woe_diff] 
            
        ## Compute final IV
        iv_total_final = sum(woe_dfrm_final.fillna(0)["iv_bins"])
        
        ## Save final binning solution via look-up-table for deployment
        lower_cutpoints_final_dfrm = pd.DataFrame(cutpoints_final, columns=['cutpoints_final'])
        
        upper_cutpoints_final_dfrm = pd.DataFrame(cutpoints_final[1:]+['Missing'], columns=['upper_cutpoints_final_dfrm'])
        iv_total_final = pd.DataFrame([[iv_total_final]], columns=['iv_total_final'])
        look_up_table = pd.concat([woe_dfrm_final.iloc[:,3].reset_index(drop=False), lower_cutpoints_final_dfrm, upper_cutpoints_final_dfrm, iv_total_final], axis=1).set_index(["predictor_var_binned"])
        look_up_table = pd.concat([look_up_table, woe_dfrm_final.iloc[:,0], woe_dfrm_final.iloc[:,1], woe_dfrm_final.iloc[:,2], woe_dfrm_final.iloc[:,6]], axis=1) # add column with final total Information Value
        look_up_table["iv_total_final"] = look_up_table["iv_total_final"].fillna(method = 'ffill')
        
        if bad == 0 and good == 1:
            look_up_table = look_up_table.rename(index=str, columns={0: "bad", 1: "good"})
        elif good == 0 and bad==1:    
            look_up_table = look_up_table.rename(index=str, columns={0: "good", 1: "bad"})  
        
        binning = pd.concat([(woe_dfrm_final["woe"]).reset_index(drop=False), look_up_table["cutpoints_final"].reset_index(drop=True), look_up_table["upper_cutpoints_final_dfrm"].reset_index(drop=True), look_up_table["iv_total_final"].reset_index(drop=True), look_up_table["good"].reset_index(drop=True), look_up_table["bad"].reset_index(drop=True), look_up_table["col_perc_a"].reset_index(drop=True), look_up_table["col_perc_b"].reset_index(drop=True), woe_dfrm_final["iv_bins"].reset_index(drop=True)], axis=1, sort=False).set_index(["predictor_var_binned"])
    
    
    
    
    
    
    
    ### Binning in case a factor was selected        
    if len(dfrm.iloc[:,0].drop_duplicates()) == 2 and (dfrm.iloc[:,1].dtype == 'object')==True:
        dfrm.iloc[:,1] = dfrm.iloc[:,1].astype('category')            
        ## Copy predictor variable to prepare binning/recoding
        dfrm["predictor_var_binned"] = dfrm["predictor_var"]
        
        ## Handling of NAs
        if dfrm["predictor_var_binned"].isnull().values.any()==True:
            dfrm["predictor_var_binned"] = dfrm["predictor_var_binned"].cat.add_categories(["Missing"])   # add factor level 'Missing'
            dfrm["predictor_var_binned"] = dfrm["predictor_var_binned"].fillna("Missing")   # replace NA with string 'Missing'
            
        ## Prepare binned factor in INPUT data (levels may be merged in subsequent steps)
        df[pred_var+"_binned"] = df[pred_var].astype('category')
        #df.loc[df[pred_var+"_binned"]] = df[pred_var].astype('category')
    
        # Handling of NAs
        if df.iloc[:,len(df.columns)-1].isnull().values.any()==True:
            df.iloc[:,len(df.columns)-1]=df.iloc[:,len(df.columns)-1].cat.add_categories(["Missing"]) # add factor level 'Missing'
            df.iloc[:,len(df.columns)-1]=df.iloc[:,len(df.columns)-1].fillna("Missing")  # replace NA with string 'Missing'
    
        
        ## Calculate initial crosstab from binned variable and target variable
        ## to identify and merge sparse bins
        
        # Compute crosstab from binned variable and target variable and covert it to a data frame
        freq_table = pd.crosstab(dfrm["predictor_var_binned"],dfrm["target_var"])
        woe_dfrm = pd.DataFrame(freq_table) # Convert frequency table to data frame
        # Compute WOE and information value (IV) from crosstab frequencies
        woe_dfrm["col_perc_a"] = woe_dfrm[good]/sum(woe_dfrm[good])
        woe_dfrm["col_perc_b"] = woe_dfrm[bad]/sum(woe_dfrm[bad])
        
        # Correct column percents in case of 0 frequencies
        if min(woe_dfrm.iloc[:,0])==0 or min(woe_dfrm.iloc[:,1])==0:
            woe_dfrm["col_perc_a"] = (woe_dfrm["col_perc_a"] + 0.0001)/sum(woe_dfrm["col_perc_a"] + 0.0001)
            woe_dfrm["col_perc_b"] = (woe_dfrm["col_perc_b"] + 0.0001)/sum(woe_dfrm["col_perc_b"] + 0.0001)
    
        # Merge factor levels with frequencies < percentage limit specified above to "misc. level" (associated with pos. and neg. WOE values)
        woe_dfrm["sparse_merge"]=np.nan
        woe_dfrm.loc[(woe_dfrm["col_perc_a"]<min_perc_class) | (woe_dfrm["col_perc_b"]<min_perc_class) | (((woe_dfrm.iloc[:,0]+woe_dfrm.iloc[:,1])/((woe_dfrm.iloc[:,0].sum())+(woe_dfrm.iloc[:,1].sum())))<min_perc_total), "sparse_merge"] = 1
        woe_dfrm_sparse_subset = woe_dfrm[woe_dfrm["sparse_merge"]==1]
        
        woe_dfrm_sparse_subset.loc[(woe_dfrm_sparse_subset["col_perc_a"]<= woe_dfrm_sparse_subset["col_perc_b"]),"sparse_merge"] =-1
               
        woe_dfrm_sparse_subset_pos = woe_dfrm_sparse_subset[woe_dfrm_sparse_subset["sparse_merge"]==1]
        woe_dfrm_sparse_subset_neg = woe_dfrm_sparse_subset[woe_dfrm_sparse_subset["sparse_merge"]==-1]
        
        if len(list(woe_dfrm_sparse_subset_pos.index))>0:
            for i in range (0,len(list(woe_dfrm_sparse_subset_pos.index))):
                dfrm["predictor_var_binned"]= dfrm["predictor_var_binned"].cat.remove_categories(list(woe_dfrm_sparse_subset_pos.index)[i])
                
            dfrm["predictor_var_binned"] = dfrm["predictor_var_binned"].cat.add_categories(["misc. level pos."])   # add factor level 'Missing'
            dfrm["predictor_var_binned"] = dfrm["predictor_var_binned"].fillna("misc. level pos.")   # replace NA with string 'Missing'
          
        if len(list(woe_dfrm_sparse_subset_neg.index))>0:    
            for i in range (0,len(list(woe_dfrm_sparse_subset_neg.index))):
                dfrm["predictor_var_binned"]= dfrm["predictor_var_binned"].cat.remove_categories(list(woe_dfrm_sparse_subset_neg.index)[i]) 
            
            dfrm["predictor_var_binned"] = dfrm["predictor_var_binned"].cat.add_categories(["misc. level neg."])   # add factor level 'Missing'
            dfrm["predictor_var_binned"] = dfrm["predictor_var_binned"].fillna("misc. level neg.")   # replace NA with string 'Missing'
        
        ## After sparse levels are merged:
        ## Merge levels with similar WOE values and calculate corresponding WOE table and IV step by step until
        ## 2 regular bins (+ Missing or 'misc. level') are left
    
        while len(dfrm["predictor_var_binned"].cat.categories)>3:
        
            # Compute crosstab from binned variable and target variable and covert it to a data frame
            freq_table = pd.crosstab(dfrm["predictor_var_binned"],dfrm["target_var"])
            woe_dfrm = pd.DataFrame(freq_table) # Convert frequency table to data frame
            # Compute WOE and information value (IV) from crosstab frequencies
            woe_dfrm["col_perc_a"] = woe_dfrm[good]/sum(woe_dfrm[good])
            woe_dfrm["col_perc_b"] = woe_dfrm[bad]/sum(woe_dfrm[bad])
        
            # Correct column percents in case of 0 frequencies
            if min(woe_dfrm.iloc[:,0])==0 or min(woe_dfrm.iloc[:,1])==0:
                woe_dfrm["col_perc_a"] = (woe_dfrm["col_perc_a"] + 0.0001)/sum(woe_dfrm["col_perc_a"] + 0.0001)
                woe_dfrm["col_perc_b"] = (woe_dfrm["col_perc_b"] + 0.0001)/sum(woe_dfrm["col_perc_b"] + 0.0001)   
    
    
            woe_dfrm["woe"] = 100*np.log(woe_dfrm["col_perc_a"]/woe_dfrm["col_perc_b"])
            woe_dfrm = woe_dfrm.sort_values(by=['woe'])
        
            woe_dfrm_list = list(woe_dfrm["woe"]) #dataframe to list
            woe_lag = list(woe_dfrm["woe"]) #dataframe to list
            
            woe_lag.insert(0, np.nan)  #add Nan [0]
            woe_lag.pop(len(woe_lag)-1) #delete Nan
            
            woe_dfrm_list=pd.DataFrame(woe_dfrm_list) #list to dataframe
            woe_lag= pd.DataFrame(woe_lag) #list to dataframe
            
            woe_diff = (woe_dfrm_list-woe_lag).abs()
            
            woe_dfrm["woe_lag"] = list(woe_lag.iloc[:,0]) #add column woe_lag to woe_dfrm
            woe_dfrm["woe_diff"] = list(woe_diff.iloc[:,0]) #add column woe_diff to woe_dfrm
            
            woe_dfrm["iv_bins"] = (woe_dfrm["col_perc_a"]-woe_dfrm["col_perc_b"])*woe_dfrm["woe"]/100
            
            # Calculate total IV for current binning
            iv_total = sum(woe_dfrm.fillna(0)['iv_bins'])
            iv_total = pd.DataFrame([[iv_total]], columns=['iv_total'])
    
            # Collect total IVs for different binning solutions
            if np.isnan(iv_total_collect.iloc[0][0])==False:
                iv_total_collect = pd.concat([iv_total_collect, iv_total], axis=1)
            else:
                iv_total_collect.iloc[0][0] = iv_total.iloc[0][0]
            
            # In case IV decreases by more than percentage specified by stop.limit parameter above
            # restore former binning solution (cutpoints) and leave loop
            if len(iv_total_collect.columns)>1:
                actual_iv_decrease = ((iv_total_collect.iloc[:,len(iv_total_collect.columns)-2])-(iv_total_collect.iloc[:,len(iv_total_collect.columns)-1]))/(iv_total_collect.iloc[:,len(iv_total_collect.columns)-2])
        
                if actual_iv_decrease.iloc[0] > stop_limit and stop_limit_exceeded == False:
                    stop_limit_exceeded = True   # indicates that stop limit is exceeded to prevent overriding the final solution
            
            # Merge until 2 regular bins remain  
            if len(dfrm["predictor_var_binned"].cat.categories)>3:
                
                # Merge levels with most similar WOE values
                min_woe_diff = woe_dfrm["woe_diff"]==min(woe_dfrm["woe_diff"].fillna(+np.inf))
                min_woe_diff_index = min_woe_diff.reset_index(drop=True).index[min_woe_diff == True].tolist()[0]
                
                dfrm["predictor_var_binned"]= dfrm["predictor_var_binned"].cat.remove_categories(min_woe_diff.index[min_woe_diff_index])
                dfrm["predictor_var_binned"]= dfrm["predictor_var_binned"].cat.remove_categories(min_woe_diff.index[min_woe_diff_index-1]) 
                dfrm["predictor_var_binned"] = dfrm["predictor_var_binned"].cat.add_categories([min_woe_diff.index[min_woe_diff_index]+ " + " +min_woe_diff.index[min_woe_diff_index-1]])
                dfrm["predictor_var_binned"] = dfrm["predictor_var_binned"].fillna(min_woe_diff.index[min_woe_diff_index]+ " + "+min_woe_diff.index[min_woe_diff_index-1])
                
                
                # Save names of the factor levels that are merged
                list_level_a = [woe_dfrm.index[min_woe_diff_index]]
                list_level_b = [woe_dfrm.index[min_woe_diff_index-1]]
                
                # Collect names of the factor levels that are merged in lists (until stop criteria is reached)
                if list_level_a_collected == False:
                    list_level_a_collected = list_level_a
                    list_level_b_collected = list_level_b
                else:
                    if stop_limit_exceeded == False:
                        list_level_a_collected = list_level_a_collected + list_level_a
                        list_level_b_collected = list_level_b_collected + list_level_b
    
                    else:
                        list_level_a_collected = list_level_a_collected[0:len(list_level_a_collected)]
                        list_level_b_collected = list_level_b_collected[0:len(list_level_b_collected)]
                                    
        ### Apply FINAL binning to INPUT data
        
        ## Merge factor levels
        # Merge sparse levels
        df.iloc[:,len(df.columns)-1] = df.iloc[:,len(df.columns)-1].cat.add_categories(["misc. level pos.","misc. level neg."])
        for i in range (0,len(list(woe_dfrm_sparse_subset_pos.index))):
            df.iloc[:,len(df.columns)-1]= df.iloc[:,len(df.columns)-1].cat.remove_categories(list(woe_dfrm_sparse_subset_pos.index)[i])
        df.iloc[:,len(df.columns)-1] = df.iloc[:,len(df.columns)-1].fillna("misc. level pos.")
        for i in range (0,len(list(woe_dfrm_sparse_subset_neg.index))):
            df.iloc[:,len(df.columns)-1]= df.iloc[:,len(df.columns)-1].cat.remove_categories(list(woe_dfrm_sparse_subset_neg.index)[i])
        df.iloc[:,len(df.columns)-1] = df.iloc[:,len(df.columns)-1].fillna("misc. level neg.")
        
        # Merge levels with similar WOE values
        if list_level_a_collected != False:
            for i in range(0,len(list_level_a_collected)): 
                df.iloc[:,len(df.columns)-1]= df.iloc[:,len(df.columns)-1].cat.remove_categories(list_level_a_collected[i])
                df.iloc[:,len(df.columns)-1]= df.iloc[:,len(df.columns)-1].cat.remove_categories(list_level_b_collected[i])
                df.iloc[:,len(df.columns)-1]= df.iloc[:,len(df.columns)-1].cat.add_categories([list_level_a_collected[i]+ " + " + list_level_b_collected[i]])
                df.iloc[:,len(df.columns)-1]= df.iloc[:,len(df.columns)-1].fillna(list_level_a_collected[i]+ " + " + list_level_b_collected[i])
    
        ## Repeat generating WOE table for selected binning solution
        
        # Compute crosstab from binned variable and target variable and covert it to a data frame
        freq_table_final = pd.crosstab(df.iloc[:,len(df.columns)-1],dfrm["target_var"])
        woe_dfrm_final = pd.DataFrame(freq_table_final) # Convert frequency table to data frame
    
        # Compute WOE and information value (IV) from crosstab frequencies
        woe_dfrm_final["col_perc_a"] = woe_dfrm_final[good]/sum(woe_dfrm_final[good])
        woe_dfrm_final["col_perc_b"] = woe_dfrm_final[bad]/sum(woe_dfrm_final[bad])
        # Correct column percents in case of 0 frequencies
        if min(woe_dfrm_final.iloc[:,0])==0 or min(woe_dfrm_final.iloc[:,1])==0:
                woe_dfrm_final["col_perc_a"] = (woe_dfrm_final["col_perc_a"] + 0.0001)/sum(woe_dfrm_final["col_perc_a"] + 0.0001)
                woe_dfrm_final["col_perc_b"] = (woe_dfrm_final["col_perc_b"] + 0.0001)/sum(woe_dfrm_final["col_perc_b"] + 0.0001)  
         
        woe_dfrm_final["woe"] = 100*np.log(woe_dfrm_final["col_perc_a"]/woe_dfrm_final["col_perc_b"])
        woe_dfrm_final = woe_dfrm_final.sort_values(by=['woe'])
        woe_dfrm_final["iv_bins"] = (woe_dfrm_final["col_perc_a"]-woe_dfrm_final["col_perc_b"])*woe_dfrm_final["woe"]/100
        iv_total_final = sum(woe_dfrm_final.fillna(0)['iv_bins'])
        iv_total_final = pd.DataFrame([[iv_total_final]], columns=['iv_total_final'])
        
           
        ## Add variable with corresponding WOE values for final binning
        
        # Add final binned (numerical) variable with WOE values (via left join with WOE table)
    
        df = pd.merge(df,woe_dfrm_final["woe"].to_frame(), how='left', left_on=df.columns[len(df.columns)-1], right_index=True)
        df= df.rename(index=str, columns={df.columns[len(df.columns)-1]: pred_var+"_binned_woe"})
    
        ## Save final binning solution via look-up-table for deployment
        df[pred_var] = df[pred_var].astype('category')
        df[pred_var]=df[pred_var].cat.add_categories(["Missing"]) # add factor level 'Missing'
        df[pred_var] = df[pred_var].fillna("Missing")   # replace NA with string 'Missing'
        
        look_up_table = df.groupby([df[pred_var],df[df.columns[len(df.columns)-2]]])
        look_up_table = look_up_table[[df.columns[len(df.columns)-1]]].mean().dropna().reset_index(drop=False)
        
        look_up_table =pd.concat([look_up_table.iloc[:,1],look_up_table.drop(columns=[look_up_table.columns[1]])], axis=1)
        look_up_table = look_up_table.rename(index=str, columns={look_up_table.columns[0]: "Group_2", look_up_table.columns[1]: "Group_1"})
        
        look_up_table = pd.concat([look_up_table.reset_index(drop=True), iv_total_final], axis=1)   # add column with final total Information Value
        look_up_table["iv_total_final"] = look_up_table["iv_total_final"].fillna(method = 'ffill')
        look_up_table = look_up_table.rename(index=str, columns={look_up_table.columns[2]: "woe"})
        
        
        look_up_table = pd.merge(look_up_table, woe_dfrm_final.drop(columns=['woe']), how='left', left_on=look_up_table.columns[0], right_index=True)
        look_up_table = look_up_table.sort_values(by=['woe',look_up_table.columns[0]]) # sort by woe value and merged bin name
        

        # In case the misc. level consists only of only NA rename it 'Missing'
        if len(look_up_table[look_up_table.iloc[:,1]=="Missing"])==1 and len(look_up_table[look_up_table.iloc[:,0]=="misc. level neg."])==1:
            if look_up_table[look_up_table.iloc[:,1]=="Missing"].index[0] == look_up_table[(look_up_table.iloc[:,0]=="misc. level neg.") & (look_up_table.iloc[:,1]=="Missing")].index[0]:
                look_up_table.iloc[:,0] = look_up_table.iloc[:,0].cat.add_categories(["Missing"])   # add factor level 'Missing'
                look_up_table.iloc[:,0] = look_up_table.iloc[:,0].cat.remove_categories("misc. level neg.")
                look_up_table.iloc[:,0] = look_up_table.iloc[:,0].fillna("Missing")   # replace NA with string "Missing"
            
        if len(look_up_table[look_up_table.iloc[:,1]=="Missing"])==1 and len(look_up_table[look_up_table.iloc[:,0]=="misc. level pos."])==1:
            if look_up_table[look_up_table.iloc[:,1]=="Missing"].index[0] == look_up_table[(look_up_table.iloc[:,0]=="misc. level pos.") & (look_up_table.iloc[:,1]=="Missing")].index[0]:   
                look_up_table.iloc[:,0] = look_up_table.iloc[:,0].cat.add_categories(["Missing"])   # add factor level 'Missing'
                look_up_table.iloc[:,0] = look_up_table.iloc[:,0].cat.remove_categories("misc. level pos.")
                look_up_table.iloc[:,0] = look_up_table.iloc[:,0].fillna("Missing")   # replace NA with string "Missing"

        # Abbreviate long factor levels (in case they are longer than specified or longer than 1000 characters)
        if abbrev_fact_levels==0 and 1000<look_up_table.iloc[:,1].str.len().max():
            abbrev_fact_levels = 1000
    
            
        if bad == 0 and good == 1:
            look_up_table = look_up_table.rename(index=str, columns={0: "bad", 1: "good"})
        elif good == 0 and bad== 1:    
            look_up_table = look_up_table.rename(index=str, columns={0: "good", 1: "bad"})  
            
        binning=look_up_table
            
             
    #### Check for correct variable specification and
    #### generate requested output, in case specification is correct
    
    ### Display warning message in case of incorrect predictor variable specification
    
    if (dfrm.iloc[:,1].dtypes.kind in 'bifc') == False and (dfrm.iloc[:,1].dtypes=="category")==False:
    	warnings.warn("Incorrect variable specification.\nPredictor variable needs to be a numeric variable or a factor.")

    
    ### Generate requested output, in case specification is correct
    
    else:
        ## Function passes the final binning solution as look-up table
        look_up_table
        
    return binning

def woe_binning (df, target_var, pred_var, min_perc_total, min_perc_class, stop_limit, abbrev_fact_levels, event_class):
#### Warning message and defaults in case parameters are not specified
    if df.isnull().values.any()==True or type(target_var) is str == False or type(pred_var) is str == False:
        warnings.warn("Incorrect specification of data frame and/or variables.")
    
    if pd.isnull(min_perc_total)==True:
        min_perc_total=0.05
        
    if min_perc_total<0.0001 or min_perc_total>0.2 or (str(min_perc_total).replace('.','',1).isdigit()) == False:
        warnings.warn("Incorrect parameter specification; accepted min.perc.total parameter range is 0.0001-0.2. Parameter was set to default (0.05).")
        min_perc_total=0.05

    if pd.isnull(min_perc_class)==True:
        min_perc_class=0
        
    if min_perc_class<0 or min_perc_class>0.2 or (str(min_perc_class).replace('.','',1).isdigit()) == False:
        warnings.warn("Incorrect parameter specification; accepted min.perc.class parameter range is 0-0.2. Parameter was set to default (0).")
        min_perc_class=0
        
    if pd.isnull(stop_limit)==True:
        stop_limit=0.1
        
    if stop_limit<0 or stop_limit>0.5 or (str(stop_limit).replace('.','',1).isdigit()) == False:
        warnings.warn("Incorrect parameter specification; accepted stop.limit parameter range is 0-0.05. Parameter was set to default (0.1).")
        stop_limit=0.1
        
    if pd.isnull(abbrev_fact_levels)==True:
        abbrev_fact_levels=200
        
    if abbrev_fact_levels<0 or abbrev_fact_levels>1000:
        warnings.warn("Incorrect parameter specification; accepted abbrev.fact.levels parameter range is 0-10000. Parameter was set to default (200).")
        abbrev_fact_levels=200

    #### Display warning message in case of incorrect target variable specification
    if len(df[target_var].drop_duplicates().isna())!=2:
        warnings.warn("Incorrect variable specification.\nTarget variable must have two distinct values (NAs are accepted).")

    #### Display warning message in case none of the target classes matches the specified event.class parameter
    if pd.isnull(event_class)==False:
        if df[target_var].drop_duplicates().iloc[0]==event_class or df[target_var].drop_duplicates().iloc[1]==event_class==False:
            warnings.warn("None of the target classes matches the specified event.class parameter.")
    
    #### In case bad class was specified assign 'good' and 'bad' codes (the latter will be associated with negative WOE values then)
    if pd.isnull(event_class)==False: 
        if df[target_var].drop_duplicates().iloc[0]==event_class:
            bad = df[target_var].drop_duplicates().iloc[0]
            good = df[target_var].drop_duplicates().iloc[1]
        else:
            bad = df[target_var].drop_duplicates().iloc[1]
            good = df[target_var].drop_duplicates().iloc[0]
            
    else:
        bad = df[target_var].drop_duplicates().iloc[0]
        good = df[target_var].drop_duplicates().iloc[1]
        
    bad = int(bad)
    good = int(good)

    #### Subset: consider only cases without NA in target variable
    df = df[df[target_var].isna()==False]
    
    #### Call actual binning function and put binning solutions together with respective variable names into a list
    woe_binning_2(df, target_var, pred_var, min_perc_total, min_perc_class, stop_limit, abbrev_fact_levels, bad, good)

    return woe_binning_2(df, target_var, pred_var, min_perc_total, min_perc_class, stop_limit, abbrev_fact_levels, bad, good)

#woe_binning (df, target_var, pred_var, min_perc_total, min_perc_class, stop_limit, abbrev_fact_levels, event_class)     
binning = woe_binning(df, 'credit_risk', 'purpose', 0.05, 0, 0.1, 50, 'bad')

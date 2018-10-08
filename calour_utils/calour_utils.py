import numpy as np
import scipy.stats
from statsmodels.sandbox.stats.multicomp import multipletests
import matplotlib.pyplot as plt

import calour as ca


def equalize_groups(exp, group_field, equal_fields):
    '''Normalize an experiment so all groups have the same number of samples from each equal_field

    Parameters
    ----------
    group_field: str
        the field by which samples are divided into groups (at least 2 groups)
    equal_field: list of str
        list of fields for which each of the groups should have the same amount of samples for each value.
        if more than one supplied, the combination is created as a unique value

    Returns
    -------
        Experiment, with equal number of samples for each value of equal_fields in each group
    '''
    exp = exp.copy()
    jfield = equal_fields[0]
    if len(equal_fields) > 1:
        cname = '__calour_joined'
        for cefield in equal_fields[1:]:
            exp = exp.join_metadata_fields(jfield, cefield, cname)
            jfield = cname
            cname += 'X'
        jfield = cname
    exp = exp.join_metadata_fields(group_field, jfield, '__calour_final_field')
    allexp = []
    for cval in exp.sample_metadata[jfield].unique():
        cexp = exp.filter_samples(jfield, cval)
        cexp = cexp.downsample('__calour_final_field', inplace=True)
        allexp.append(cexp)
    res = allexp.pop()
    if len(allexp) > 1:
        for cexp in allexp:
            res = res.join_experiments(cexp)
    return res


def get_ratios(exp, id_field, group_field, group1, group2, min_thresh=5):
    '''get a new experiment made of the ratios between different group_field values
    for the same id_field

    Parameters
    ----------
    exp : Experiment
    id_field: str
        name of the field containing the individual id. ratios are calculated
        for samples with the same id_field (i.e. the individual id)
    group_field: str
        name of the field with the two groups to calculate the ratio of
        (i.e. sample_site)
    group1: str
        value of group_field for group1 (nominator)
    group2: str
        value of group_field for group1 (denominator)

    Returns
    -------
    calour.Experiment
        with only samples from group1 that have group1 and group2 values.
        Data contains the ratio of group1/group2
    '''
    data = exp.get_data(sparse=False)
    newexp = exp.copy()
    newexp.sparse = False
    keep = []
    for cid in exp.sample_metadata[id_field].unique():
        pos1 = np.where((exp.sample_metadata[id_field] == cid) & (exp.sample_metadata[group_field] == group1))[0]
        pos2 = np.where((exp.sample_metadata[id_field] == cid) & (exp.sample_metadata[group_field] == group2))[0]
        if len(pos1) != 1:
            print('not 1 sample for group1: %s' % cid)
            continue
        if len(pos2) != 1:
            print('not 1 sample for group2: %s' % cid)
            continue
        cdat1 = data[pos1, :]
        cdat2 = data[pos2, :]
        cdat1[cdat1 < min_thresh] = min_thresh
        cdat2[cdat2 < min_thresh] = min_thresh
        newexp.data[pos1, :] = np.log2(cdat1 / cdat2)
        keep.append(pos1[0])
    print('found %d ratios' % len(keep))
#     print(keep)
    newexp = newexp.reorder(keep, axis='s')
    return newexp


def get_sign_pvals(exp, alpha=0.1, min_present=5):
    '''get p-values for a sign-test with the data in exp
    data should come from get_ratios()
    does fdr on it
    '''
    exp = exp.copy()
    # get rid of bacteria that don't have enough non-zero ratios
    keep = []
    for idx in range(exp.data.shape[1]):
        cdat = exp.data[:, idx]
        npos = np.sum(cdat > 0)
        nneg = np.sum(cdat < 0)
        if npos + nneg >= min_present:
            keep.append(idx)
    print('keeping %d features with enough ratios' %len(keep))
    exp=exp.reorder(keep,axis='f')
    pvals = []
    esize = []
    for idx in range(exp.data.shape[1]):
        cdat = exp.data[:,idx]
        npos = np.sum(cdat>0)
        nneg = np.sum(cdat<0)
        pvals.append(scipy.stats.binom_test(npos,npos+nneg))
        esize.append((npos-nneg)/(npos+nneg))
    plt.figure()
    sp = np.sort(pvals)
    plt.plot(np.arange(len(sp)),sp)
    plt.plot([0,len(sp)],[0,1],'k')
    reject = multipletests(pvals, alpha=alpha, method='fdr_bh')[0]
    index = np.arange(len(reject))
    esize = np.array(esize)
    pvals = np.array(pvals)
    exp.feature_metadata['esize']=esize
    exp.feature_metadata['pval']=pvals
    index = index[reject]
    okesize = esize[reject]
    new_order = np.argsort(okesize)
    newexp = exp.reorder(index[new_order], axis='f', inplace=False)
    print('found %d significant' % len(newexp.feature_metadata))
    return newexp


def show_wordcloud(exp, ignore_exp=None, server='http://127.0.0.1:5000'):
    '''open the wordcloud html page from dbbact for all sequences in exp
    
    File is saved into 'wordcloud.html'

    Parameters
    ----------
    exp: AmpliconExperiment
    ignore_exp: None or list of int, optional
        expids to ignore when drawing the wordcloud
    '''
    import requests
    import webbrowser
    import os
    
    print('getting wordcloud for %d sequences' % len(exp.feature_metadata))
    params={}
    params['sequences']=list(exp.feature_metadata.index.values)
    params['ignore_exp']=ignore_exp
    res=requests.post(server+'/sequences_wordcloud',json=params)
    
    if res.status_code!=200:
        print('failed')
        print(res.status_code)
        print(res.reason)

    print('got output')
    with open('wordcloud.html','w') as fl:
        fl.write(res.text)
    webbrowser.open('file://'+os.path.realpath('wordcloud.html'), new=True)


def collapse_correlated(exp,min_corr=0.95):
    '''merge features that have very correlated expression profile
    useful after dbbact.sample_enrichment()
    all correlated featuresIDs are concatenated to a single id

    Returns
    -------
    Experiment, with correlated features merged
    '''
    import numpy as np
    data = exp.get_data(sparse=False, copy=True)
    corr = np.corrcoef(data,rowvar=False)
    use_features=set(np.arange(corr.shape[0]))
    feature_ids = {}
    orig_ids = {}
    for idx, cfeature in enumerate(exp.feature_metadata.index.values):
        feature_ids[idx] = str(cfeature)
        orig_ids[idx] = str(cfeature)

    da = exp.feature_metadata['_calour_diff_abundance_effect']
    for idx in range(corr.shape[0]):
        if idx not in use_features:
            continue
        corr_pos = np.where(corr[idx,:]>=min_corr)[0]
        for idx2 in corr_pos:
            if idx2==idx:
                continue
            if idx2 in use_features:
                id1 = orig_ids[idx]
                id2 = orig_ids[idx2]
                if abs(da[id1]) < abs(da[id2]):
                    pos1 = idx2
                    pos2 = idx
                else:
                    pos1 = idx
                    pos2 = idx2
                feature_ids[pos1] = feature_ids[pos1]+'; '+feature_ids[pos2]
#                 data[:, idx] = data[:, idx] + data[:, idx2]
                use_features.remove(idx2)
                del feature_ids[idx2]
    keep_pos = list(use_features)
    newexp = exp.copy()
    newexp.data = data
    newexp = newexp.reorder(keep_pos, axis='f', inplace=True)
    feature_ids_list = [feature_ids[idx] for idx in keep_pos]
    newexp.feature_metadata['_featureid'] = feature_ids_list
    newexp.feature_metadata.set_index('_featureid', drop=False, inplace=True)
    return newexp


def plot_violin(exp, field, features=None, downsample=True, num_keep=None, **kwargs):
    '''Plot a violin plot for the distribution of frequencies for a (combined set) of features
    
    Parameters
    ----------
    exp: Experiment
    field: str
        Name of the field to plot for
    features: list of str or None, optional
        None to sum frequencies of all features. Otherwise sum frequencies of features in list.
    downsample: bool, optional
        True to run exp.downsample on the field so all groups have same number of samples.
    num_keep: int or None, optional
        The minimal group size for downsample, or None to use smallest group size
    **kwargs: additional parameters to pass to pyplot.violinplot
    
    Returns
    -------
    figure
    '''
    import matplotlib.pyplot as plt

    if downsample:
        exp=exp.downsample(field, num_keep=num_keep)
    if features is not None:
        exp=exp.filter_ids(features)
    data = exp.get_data(sparse=False).sum(axis=1)
    group_freqs = []
    group_names = []
    for cgroup in exp.sample_metadata[field].unique():
        group_names.append(cgroup)
        group_freqs.append(data[exp.sample_metadata[field]==cgroup])
    fig = plt.figure()
    plt.violinplot(group_freqs,**kwargs)
    plt.xticks(np.arange(1,len(group_names)+1),group_names)
    return fig


def splot(exp,field,**kwargs):
    '''
    Plot a sorted version of the experiment exp based on field
    '''
    tt = exp.sort_samples(field)
    res = tt.plot(sample_field=field,gui='qt5',**kwargs)
    return res


def sort_by_bacteria(exp, seq, inplace=False):
    import numpy as np
    '''sort samples according to the frequency of a given bacteria
    the selected bacteria frequency field is named "bf" (in the sample_metadata)
    '''
    spos=np.where(exp.feature_metadata.index.values==seq)[0][0]
    bf=exp.get_data(sparse=False,copy=True)[:,spos].flatten()
    if inplace:
        newexp=exp
    else:
        newexp=exp.copy()
    newexp.sample_metadata['bf']=bf
    newexp=newexp.sort_samples('bf')
    return newexp

from collections import defaultdict

import numpy as np
import scipy.stats
from statsmodels.sandbox.stats.multicomp import multipletests
# import matplotlib.pyplot as plt
import pandas as pd

import calour as ca
from calour.util import _to_list


def equalize_groups(exp, group_field, equal_fields, random_seed=None):
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
    exp = exp.join_metadata_fields(group_field, jfield, '__calour_final_field')
    samples = []
    for cval in exp.sample_metadata[jfield].unique():
        cexp = exp.filter_samples(jfield, cval)
        if len(cexp.sample_metadata['__calour_final_field'].unique()) == 1:
            continue
        cexp = cexp.downsample('__calour_final_field', inplace=True, random_state=random_seed)
        samples.extend(cexp.sample_metadata.index.values)
    res = exp.filter_ids(samples, axis='s')
    return res


def merge_general(exp, field, val1, val2, new_field=None, v1_new=None, v2_new=None):
    '''merge a field with multiple values into a new field with only two values
    All samples with values not in val1, val2 are filtered away

   Parameters
   ----------
    exp:
        calour.Experiment
    field : str
        the field to merge
    val1, val2: list of str
        the values to merge together
    new_field : str or None (optional)
        name of the new field. if None, new field will be field+"_merged"
    v1_new, v2_new: str or None, optional
        name of new values for merged val1, val2
        if None, will use "_".join(val1)

    Returns
    -------
    newexp: calour.Experiment, with values in 2 categories - yes/no
    '''
    if new_field is None:
        new_field = field + '_merged'
    newexp = exp.copy()
    newexp.sample_metadata[new_field] = newexp.sample_metadata[field].copy()
    if v1_new is None:
        v1_new = '+'.join(map(str, val1))
    if v2_new is None:
        v2_new = '+'.join(map(str, val2))
    newexp.sample_metadata[new_field].replace(val1, v1_new, inplace=True)
    newexp.sample_metadata[new_field].replace(val2, v2_new, inplace=True)
    newexp = newexp.filter_samples(new_field, [v1_new, v2_new], inplace=True)
    return newexp


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
    print('keeping %d features with enough ratios' % len(keep))
    exp = exp.reorder(keep, axis='f')
    pvals = []
    esize = []
    for idx in range(exp.data.shape[1]):
        cdat = exp.data[:, idx]
        npos = np.sum(cdat > 0)
        nneg = np.sum(cdat < 0)
        pvals.append(scipy.stats.binom_test(npos, npos + nneg))
        esize.append((npos - nneg) / (npos + nneg))
    # plt.figure()
    # sp = np.sort(pvals)
    # plt.plot(np.arange(len(sp)),sp)
    # plt.plot([0,len(sp)],[0,1],'k')
    reject = multipletests(pvals, alpha=alpha, method='fdr_bh')[0]
    index = np.arange(len(reject))
    esize = np.array(esize)
    pvals = np.array(pvals)
    exp.feature_metadata['esize'] = esize
    exp.feature_metadata['pval'] = pvals
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
    params = {}
    params['sequences'] = list(exp.feature_metadata.index.values)
    params['ignore_exp'] = ignore_exp
    res = requests.post(server + '/sequences_wordcloud', json=params)

    if res.status_code != 200:
        print('failed')
        print(res.status_code)
        print(res.reason)

    print('got output')
    with open('wordcloud.html', 'w') as fl:
        fl.write(res.text)
    webbrowser.open('file://' + os.path.realpath('wordcloud.html'), new=True)


def collapse_correlated(exp, min_corr=0.95):
    '''merge features that have very correlated expression profile
    useful after dbbact.sample_enrichment()
    all correlated featuresIDs are concatenated to a single id

    Returns
    -------
    Experiment, with correlated features merged
    '''
    import numpy as np
    data = exp.get_data(sparse=False, copy=True)
    corr = np.corrcoef(data, rowvar=False)
    use_features = set(np.arange(corr.shape[0]))
    feature_ids = {}
    orig_ids = {}
    for idx, cfeature in enumerate(exp.feature_metadata.index.values):
        feature_ids[idx] = str(cfeature)
        orig_ids[idx] = str(cfeature)

    da = exp.feature_metadata['_calour_diff_abundance_effect']
    for idx in range(corr.shape[0]):
        if idx not in use_features:
            continue
        corr_pos = np.where(corr[idx, :] >= min_corr)[0]
        for idx2 in corr_pos:
            if idx2 == idx:
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
                feature_ids[pos1] = feature_ids[pos1] + '; ' + feature_ids[pos2]
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
        exp = exp.downsample(field, num_keep=num_keep)
    if features is not None:
        exp = exp.filter_ids(features)
    data = exp.get_data(sparse=False).sum(axis=1)
    group_freqs = []
    group_names = []
    for cgroup in exp.sample_metadata[field].unique():
        group_names.append(cgroup)
        group_freqs.append(data[exp.sample_metadata[field] == cgroup])
    fig = plt.figure()
    plt.violinplot(group_freqs, **kwargs)
    plt.xticks(np.arange(1, len(group_names) + 1), group_names)
    return fig


def splot(exp, field, **kwargs):
    '''
    Plot a sorted version of the experiment exp based on field
    '''
    tt = exp.sort_samples(field)
    res = tt.plot(sample_field=field, gui='qt5', **kwargs)
    return res


def sort_by_bacteria(exp, seq, inplace=True):
    import numpy as np
    '''sort samples according to the frequency of a given bacteria
    '''
    spos = np.where(exp.feature_metadata.index.values == seq)[0][0]
    bf = exp.get_data(sparse=False, copy=True)[:, spos].flatten()
    if inplace:
        newexp = exp
    else:
        newexp = exp.copy()
    newexp.sample_metadata['bf'] = bf
    newexp = newexp.sort_samples('bf')
    return newexp


def metadata_enrichment(exp, field, val1, val2=None, ignore_vals=set(['Unspecified', 'Unknown']), use_fields=None, alpha=0.05):
    '''Test for metadata enrichment over all metadata fields between the two groups

    Parameters
    ----------
    exp: Experiment
    field: str
        the field to divide the samples
    val1: str or list of str
        first group values for field
    val2: str or list of str or None, optional
        second group values or None to select all not in group1
    ignore_vals: set of str
        the values in the metadata field to ignore
    use_fields: list of str or None, optional
        list of fields to test for enrichment on None to test all
    alpha: float
        the p-value cutoff


    Returns
    -------

    '''
    exp1 = exp.filter_samples(field, val1)
    if val2 is None:
        exp2 = exp.filter_samples(field, val1, negate=True)
    else:
        exp2 = exp.filter_samples(field, val2)
    tot_samples = len(exp.sample_metadata)
    s1 = len(exp1.sample_metadata)
    s2 = len(exp2.sample_metadata)

    if use_fields is None:
        use_fields = exp.sample_metadata.columns

    for ccol in use_fields:
        for cval in exp.sample_metadata[ccol].unique():
            if cval in ignore_vals:
                continue
            num1 = np.sum(exp1.sample_metadata[ccol] == cval)
            num2 = np.sum(exp2.sample_metadata[ccol] == cval)
            if num1 + num2 < 20:
                continue
            p0 = (num1 + num2) / tot_samples
            pv1 = scipy.stats.binom_test(num1, s1, p0)
            pv2 = scipy.stats.binom_test(num2, s2, p0)
            if (pv1 < alpha):
                print('column %s value %s enriched in group1. p0=%f, num1=%f/%f (e:%f) num2=%f/%f (e:%f). pval %f' % (ccol, cval, p0, num1, s1, s1 * p0, num2, s2, s2 * p0, pv1))
            if (pv2 < alpha):
                print('column %s value %s enriched in group2. p0=%f, num1=%f/%f (e:%f) num2=%f/%f (e:%f). pval %f' % (ccol, cval, p0, num1, s1, s1 * p0, num2, s2, s2 * p0, pv2))


def filter_singletons(exp, field, min_number=2):
    '''Filter away samples that have <min_number of similar values in field

    Used to remove singleton twins from the twinsuk study
    '''
    counts = exp.sample_metadata[field].value_counts()
    counts = counts[counts >= min_number]
    newexp = exp.filter_samples(field, list(counts.index.values))
    return newexp


def numeric_to_categories(exp, field, new_field, values, inplace=True):
    '''convert a continuous field to categories

    Parameters
    ----------
    exp: calour.Experiment
    field: str
        the continuous field name
    new_field: str
        name of the new categoriezed field name
    values: int or list of float
        the bins to categorize by. each number is the lowest number for the bin. a new bin is created for <first number

    Returns
    calour.Experiment with new metadata field new_field
    '''
    if not inplace:
        exp = exp.copy()
    sm = exp.sample_metadata
    exp.sample_metadata[new_field] = '>%s' % values[-1]
    for cval in values[::-1]:
        exp.sample_metadata.loc[sm[field] <= cval, new_field] = str(cval)
    return exp


def taxonomy_from_db(exp):
    '''add taxonomy to each feature based on dbbact
    '''
    exp = exp.add_terms_to_features('dbbact')
    if len(exp.exp_metadata['__dbbact_taxonomy']) == 0:
        print('did not obtain taxonomy from add_terms_to_features')
    exp.feature_metadata['taxonomy'] = 'na'
    for ck, cv in exp.exp_metadata['__dbbact_taxonomy'].items():
        exp.feature_metadata.loc[ck, 'taxonomy'] = cv
    return exp


def focus_features(exp, ids, inplace=False, focus_feature_field='_calour_util_focus'):
    '''Reorder the bacteria so the focus ids are at the beginning (top)

    Parameters
    ----------
    exp: calour.Experiments
    ids: str or list of str
        the feature ids to focus

    Returns
    -------
    calour.Experiment
        reordered
    '''
    ids = _to_list(ids)
    pos = []
    for cid in ids:
        if cid in exp.feature_metadata.index:
            pos.append(exp.feature_metadata.index.get_loc(cid))
    neworder = np.arange(len(exp.feature_metadata))
    neworder = np.delete(neworder, pos)
    neworder = pos + list(neworder)
    newexp = exp.reorder(neworder, axis='f', inplace=inplace)
    # create the new feature_metadata field denoting which are focued
    ff = ['focus'] * len(pos) + ['orig'] * (len(neworder) - len(pos))
    newexp.feature_metadata[focus_feature_field] = ff
    return newexp


def alpha_diversity_as_feature(exp):
    data = exp.get_data(sparse=False, copy=True)
    data[data < 1] = 1
    entropy = []
    for idx in range(np.shape(data)[0]):
        entropy.append(np.sum(data[idx, :] * np.log2(data[idx, :])))
    alpha_div = entropy
    newexp = exp.copy()
    newexp.sample_metadata['_alpha_div'] = alpha_div
    # newexp.add_sample_metadata_as_features('_alpha_div')
    return newexp


def filter_16s(exp, seq='TACG', minreads=5000):
    '''Filter an experiment keeping only samples containing enough sequences starting with seq
    '''
    # get the sequences starting with seq
    okseqs = [x for x in exp.feature_metadata.index.values if x[:len(seq)] == seq]

    # count how many reads from the okseqs
    texp = exp.filter_ids(okseqs)
    dat = texp.get_data(sparse=False)
    numok = dat.sum(axis=1)

    newexp = exp.reorder(numok >= minreads, axis='s')
    return newexp


def create_ko_feature_file(ko_file='ko00001.json', out_file='ko_feature_map.tsv'):
    '''Create a feature metadata file for kegg ontologies for picrust2

    Parameters
    ----------
    ko_file: str, optional
        name of the kegg ontology json file to import.
        get it from https://www.genome.jp/kegg-bin/get_htext?ko00001
    out_file: str, optional
        name of the feature mapping file to load into calour
        it contains level and name fields.

    NOTE: if term appears in several levels, it will just keep the first one.
    '''
    import json

    with open(ko_file) as f:
        tt = json.load(f)
    found = set()
    outf = open(out_file, 'w')
    outf.write('ko\tname\tlevel1\tlevel2\tlevel3\n')
    for c1 in tt['children']:
        l1name = c1['name']
        for c2 in c1['children']:
            l2name = c2['name']
            for c3 in c2['children']:
                l3name = c3['name']
                if 'children' in c3:
                    for c4 in c3['children']:
                        l4name = c4['name']
                        zz = l4name.split()
                        if zz[0] in found:
                            print('duplicate id %s' % l4name)
                            continue
                        found.add(zz[0])
                        outf.write(zz[0] + '\t')
                        outf.write(' '.join(zz[1:]) + '\t')
                        outf.write(l1name + '\t')
                        outf.write(l2name + '\t')
                        outf.write(l3name + '\n')
                else:
                    # print('no children for level3 %s' % c3)
                    pass
    print('saved to %s' % out_file)


def add_taxonomy(exp):
    '''Add DBBact derived taxonomy to sequences in the experiment
    The taxonomy is added as exp.feature_metadata.taxonomy
    NOTE: can erase the current taxonomy
    NOTE: will also fill the exp_metadata dbbact fields

    Parameters:
    -----------
    exp: calour.Experiment

    Returns:
    --------
    exp: same as the input (modification is inplace)
    '''
    exp.add_terms_to_features('dbbact', get_taxonomy=True)
    exp.feature_metadata['taxonomy'] = pd.Series(exp.exp_metadata['__dbbact_taxonomy'])
    return exp


def plot_experiment_terms(exp, weight='binary', min_threshold=0.005, show_legend=False, sort_legend=True):
    '''Plot the distribution of most common terms in the experiment
    Using the dbbact annotations. For each sequence, take the strongest term (based on f-score) and plot the
    distribution of such terms for the entire set of sequences in the experiment

    Parameters
    ----------
    exp: calour.Experiment
    weight: str, optional NOT IMPLEMENTED
        how to weigh the frequency of each bacteria. options are:
        'binary': just count the number of bacteria with each term
        'linear': weigh by mean frequency of each bacteria
    min_threshold: float, optional
        Join together to 'other' all terms with < min_treshold of sequences containing them
    show_legend: bool, optional
        True to show legend with pie slice names, false to showin slices
    sort_legend: bool, optional
        True to sort the legend by the pie slice size

    Returns
    -------
    '''
    import matplotlib.pyplot as plt

    exp = exp.add_terms_to_features('dbbact')
    ct = exp.feature_metadata['common_term'].value_counts()
    dat = exp.get_data(sparse=False)
    feature_sum = dat.sum(axis=0)
    terms = exp.feature_metadata['common_term']

    ct = defaultdict(float)
    for idx, cseq in enumerate(exp.feature_metadata.index.values):
        cterm = terms[cseq]
        if weight == 'binary':
            ct[cterm] += 1
        elif weight == 'linear':
            ct[cterm] += feature_sum[idx]
        else:
            raise ValueError('weight=%s not supported. please use binary/linear' % weight)

    # convert to fraction
    all_sum = sum(ct.values())
    for cterm, ccount in ct.items():
        ct[cterm] = ct[cterm] / all_sum

    # join all terms < min_threshold
    c = {}
    c['other'] = 0
    for cterm, cval in ct.items():
        if cval < min_threshold:
            c['other'] += cval
        else:
            c[cterm] = cval
    plt.figure()
    labels = c.keys()
    values = []
    for clabel in labels:
        values.append(c[clabel])
    if show_legend:
        patches, texts = plt.pie(values, radius=0.5)

        percent = np.array(values)
        percent = 100 * percent / percent.sum()
        labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(labels, percent)]

        # sort according to pie slice size
        if sort_legend:
            patches, labels, dummy = zip(*sorted(zip(patches, labels, values), key=lambda x: x[2], reverse=True))

        # plt.legend(patches, labels, loc='left center', bbox_to_anchor=(-0.1, 1.), fontsize=8)
        plt.legend(patches, labels)
    else:
        plt.pie(values, labels=labels)


def read_qiime2(data_file, sample_metadata_file=None, feature_metadata_file=None, rep_seqs_file=None, **kwargs):
    '''Read a qiime2 generated table (even if it was run without the --p-no-hashedfeature-ids flag)
    This is a wrapper for calour.read_amplicon(), that can unzip and extract biom table, feature metadata, rep_seqs_file qza files generated by qiime2

    Parameters
    ----------
    data_file: str
        name of qiime2 deblur/dada2 generated feature table qza or biom table
    sample_metadata_file: str or None, optional
        name of tab separated mapping file
    feature_metadata_file: str or None, optional
        can be the taxonomy qza or tsv generated by qiime2 feature classifier
    rep_seqs_file: str or None, optional
        if not none, name of the qiime2 representative sequences qza file (the --o-representative-sequences file name in qiime2 dada2/deblur)
    **kwargs:
        to be passed to calour.read_amplicon

    Returns
    -------
    calour.AmpliconExperiment
    '''
    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:
        data_file = filename_from_zip(tempdir, data_file, 'data/feature-table.biom')
        feature_metadata_file = filename_from_zip(tempdir, feature_metadata_file, 'data/taxonomy.tsv')
        rep_seqs_file = filename_from_zip(tempdir, rep_seqs_file, 'data/dna-sequences.fasta')
        expdat = ca.read_amplicon(data_file, sample_metadata_file=sample_metadata_file, feature_metadata_file=feature_metadata_file, **kwargs)
        if rep_seqs_file is not None:
            seqs = []
            with open(rep_seqs_file) as rsf:
                for cline in rsf:
                    # take the sequence from the header
                    if cline[0] != '>':
                        continue
                    seqs.append(cline[1:])
            expdat.feature_metadata['_orig_id'] = expdat.feature_metadata['_feature_id']
            expdat.feature_metadata['_feature_id'] = seqs
            expdat.feature_metadata = expdat.feature_metadata.set_index('_feature_id')

    return expdat


def filename_from_zip(tempdir, data_file, internal_data):
    '''get the data filename from a regular/qza filename

    Parameters
    ----------
    tmpdir: str
        name of the directory to extract the zip into
    data_file: str
        original name of the file (could be '.qza' or not)
    internale_data: str
        the internal qiime2 qza file name (i.e. 'data/feature-table.biom' for biom table etc.)

    Returns
    -------
    str: name of data file to read.
    '''
    import zipfile

    if data_file is None:
        return data_file
    if not data_file.endswith('.qza'):
        return data_file
    fl = zipfile.ZipFile(data_file)
    internal_name = None
    for fname in fl.namelist():
        if fname.endswith(internal_data):
            internal_name = fname
            break
    if internal_name is None:
        raise ValueError('No biom table in qza file %s. is it a qiime2 feature table?' % data_file)
    data_file = fl.extract(internal_name, tempdir)
    return data_file

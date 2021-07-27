from collections import defaultdict
from logging import getLogger, NOTSET, basicConfig
from pkg_resources import resource_filename
from logging.config import fileConfig

import numpy as np
import scipy.stats
from statsmodels.sandbox.stats.multicomp import multipletests
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.ensemble

import calour as ca
from calour.util import _to_list
from calour.training import plot_scatter

try:
    # get the logger config file location
    log_file = resource_filename(__package__, 'log.cfg')
    # log = path.join(path.dirname(path.abspath(__file__)), 'log.cfg')
    # set the logger output according to log.cfg
    # setting False allows other logger to print log.
    fileConfig(log_file, disable_existing_loggers=False)
except:
    print('failed to load logging config file')
    basicConfig(format='%(levelname)s:%(message)s')

logger = getLogger(__package__)
# set the log level to the same as calour module if present
try:
    clog = getLogger('calour')
    calour_log_level = clog.getEffectiveLevel()
    if calour_log_level != NOTSET:
        logger.setLevel(calour_log_level)
except:
    print('calour module not found for log level setting. Level not set')


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
            exp = exp.join_metadata_fields(jfield, cefield, cname, axis='s')
            jfield = cname
            cname += 'X'
    exp = exp.join_metadata_fields(group_field, jfield, '__calour_final_field', axis='s')
    samples = []
    for cval in exp.sample_metadata[jfield].unique():
        cexp = exp.filter_samples(jfield, cval)
        if len(cexp.sample_metadata['__calour_final_field'].unique()) == 1:
            continue
        cexp = cexp.downsample('__calour_final_field', inplace=True, random_seed=random_seed)
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

    Parameters
    ----------
    exp: calour.Experiment
        An experiment that contains reatios values in the data(e.g. the result of calour_utils.get_ratios() )
    alpha: float, optional
        the FDR threshold for the features to keep
    min_present: int, optional
        ony look at features present in at least min_present samples

    Returns
    -------
    calour.Experiment with features significantly higher/lower (i.e. log ratios >0/<0) that expected by chance
    feature_metadata will include esize and pval fields
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
    res = multipletests(pvals, alpha=alpha, method='fdr_bh')
    reject = res[0]
    qvals = res[1]
    index = np.arange(len(reject))
    esize = np.array(esize)
    pvals = np.array(pvals)
    exp.feature_metadata['esize'] = esize
    exp.feature_metadata['pval'] = pvals
    exp.feature_metadata['qval'] = qvals
    index = index[reject]
    okesize = esize[reject]
    new_order = np.argsort(okesize)
    new_order = np.argsort((1 - pvals[reject]) * np.sign(okesize))
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
    tmp_field = '_calour_' + field + '_num'
    values = np.sort(values)[::-1]
    if not inplace:
        exp = exp.copy()
    # keep only numeric values (all other are 0)
    exp.sample_metadata[tmp_field] = pd.to_numeric(exp.sample_metadata[field], errors='coerce')
    exp.sample_metadata[tmp_field] = exp.sample_metadata[tmp_field].fillna(0)
    new_field_num = new_field + '_num'
    sm = exp.sample_metadata
    exp.sample_metadata[new_field] = '>%s' % values[0]
    exp.sample_metadata[new_field_num] = values[0]
    for idx, cval in enumerate(values):
        if idx < len(values) - 1:
            exp.sample_metadata.loc[sm[tmp_field] <= cval, new_field] = '%s-%s' % (values[idx + 1], cval)
        else:
            exp.sample_metadata.loc[sm[tmp_field] <= cval, new_field] = '<%s' % (values[idx])
        exp.sample_metadata.loc[sm[tmp_field] <= cval, new_field_num] = cval
    return exp


def taxonomy_from_db(exp):
    '''add taxonomy to each feature based on dbbact
    '''
    exp = exp.add_terms_to_features('dbbact', get_taxonomy=True)
    if len(exp.databases['dbbact']['taxonomy']) == 0:
        print('did not obtain taxonomy from add_terms_to_features')
    exp.feature_metadata['taxonomy'] = 'na'
    for ck, cv in exp.databases['dbbact']['taxonomy'].items():
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
    exp.feature_metadata['taxonomy'] = pd.Series(exp.databases['dbbact']['taxonomy'])
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


def genetic_distance(data, labels):
    '''calculate the std within each family
    used by get_genetic for testing bacteria significantly associated with family
    '''
    distances = np.zeros(np.shape(data)[0])
    for cidx in np.unique(labels):
        pos = np.where(labels == cidx)[0]
        if len(pos) > 1:
            distances -= np.std(data[:, pos], axis=1) / np.mean(data[:, pos], axis=1)
            # distances -= np.std(data[:, pos], axis=1)
    return distances


def get_genetic(exp, field, alpha=0.1, numperm=1000, fdr_method='dsfdr'):
    '''Look for features that depend on family/genetics by comparing within family std/mean to random permutations

    Parameters
    ----------
    field: str
        the field that has the same value for members of same family
    '''
    cexp = exp.filter_abundance(0, strict=True)
    data = cexp.get_data(copy=True, sparse=False).transpose()
    data[data < 4] = 4
    labels = exp.sample_metadata[field].values

    # remove samples that don't have similar samples w.r.t field
    remove_samps = []
    remove_pos = []
    for cidx, cval in enumerate(np.unique(labels)):
        pos = np.where(labels == cval)[0]
        if len(pos) < 2:
            remove_samps.append(cval)
            remove_pos.append(cidx)
    if len(remove_pos) > 0:
        labels = np.delete(labels, remove_pos)
        data = np.delete(data, remove_pos, axis=1)
        print('removed singleton samples %s' % remove_samps)

    print('testing with %d samples' % len(labels))
    keep, odif, pvals, qvals = ca.dsfdr.dsfdr(data, labels, method=genetic_distance, transform_type='log2data', alpha=alpha, numperm=numperm, fdr_method=fdr_method)
    print('Positive correlated features : %d. Negative correlated features : %d. total %d'
          % (np.sum(odif[keep] > 0), np.sum(odif[keep] < 0), np.sum(keep)))
    newexp = ca.analysis._new_experiment_from_pvals(cexp, exp, keep, odif, pvals, qvals)
    return newexp
    # return keep, odif, pvals


def filter_contam(exp, field, blank_vals, negate=False):
    '''Filter suspected contaminants based on blank samples

    Filter by removing features that have lower mean in samples compared to blanks

    Parameters
    ----------
    exp: calour.AmpliconExperiment
    field: str
        name of the field identifying blank samples
    blank_vals: str or list of str
        the values for the blank samples in the field
    negate: bool, optional
        False (default) to remove contaminants, True to keep only contaminants

    Returns
    -------
    calour.AmpliconExperiment with only features that are not contaminants (if negate=False) or contaminants (if negate=True)
    '''
    bdata = exp.filter_samples(field, blank_vals).get_data(sparse=False)
    sdata = exp.filter_samples(field, blank_vals, negate=True).get_data(sparse=False)
    bmean = bdata.mean(axis=0)
    smean = sdata.mean(axis=0)
    okf = smean > bmean
    print('found %d contaminants' % okf.sum())
    if negate:
        okf = (okf is False)
    newexp = exp.reorder(okf, axis='f')
    return newexp


def order_samples(exp, field, order):
    '''Order samples according to a custom order in field. non-specified values in order are maintained as is

    Parameters
    ----------
    exp: Calour.Experiment
    field: str
        name of the field to order by
    order: list of str
        the requested order of values in the field

    Returns
    -------
    Calour.Experiment
    '''
    newexp = exp.copy()
    newexp.sample_metadata['__order_field'] = 999999
    for idx, cval in enumerate(order):
        newexp.sample_metadata.loc[newexp.sample_metadata[field] == cval, '__order_field'] = idx
    newexp = newexp.sort_samples('__order_field')
    return newexp


def test_picrust_enrichment(dd_exp, picrust_exp, **kwargs):
    '''find enrichment in picrust2 terms comparing 2 groups

    Parameters
    ----------
    dd_exp: calour.AmpliconExperiment
        the differential abundance results (on bacteria)
    picrust_exp: calour.Experiment
        The picrust2 intermediate file (EC/KO). load it using:
        picrust_exp=ca.read('./EC_predicted.tsv',data_file_type='csv',sample_in_row=True, data_table_sep='\t', normalize=None)
        NOTE: rows are KO/EC, columns are bacteria
    **kwargs:
        passed to diff_abundance. can include: alpha, method, etc.

    Returns
    -------
    ca.Experiment with the enriched KO/EC terms
        The original group the bacteria (column) is in the '__group' field
    '''
    vals = dd_exp.feature_metadata['_calour_direction'].unique()
    if len(vals) != 2:
        raise ValueError('Diff abundance groups contain !=2 values')
    id1 = dd_exp.feature_metadata[dd_exp.feature_metadata['_calour_direction'] == vals[0]]
    id2 = dd_exp.feature_metadata[dd_exp.feature_metadata['_calour_direction'] == vals[1]]
    picrust_exp.sample_metadata['__picrust_test'] = ''
    picrust_exp.sample_metadata.loc[picrust_exp.sample_metadata.index.isin(id1.index), '__group'] = vals[0]
    picrust_exp.sample_metadata.loc[picrust_exp.sample_metadata.index.isin(id2.index), '__group'] = vals[1]
    tt = picrust_exp.filter_samples('__group', [vals[0], vals[1]])
    tt = tt.diff_abundance('__group', vals[0], vals[1], **kwargs)
    tt.sample_metadata = tt.sample_metadata.merge(dd_exp.feature_metadata, how='left', left_on='_sample_id', right_on='_feature_id')
    return tt


def uncorrelate(exp, normalize=False, random_seed=None):
    '''remove correlations between features in the experiment, by permuting samples of each bacteria

    Parameters
    ----------
    exp: calour.Experiment
        the experiment to permute
    normalize: False or int, optional
        if not int, normalize each sample after the uncorrelation to normalize reads
    random_seed: int or None, optional
        if not None, seed the numpy random seed with it

    Returns
    -------
    calour.Experiment
        the permuted experiment (each feature randomly permuted along samples)
    '''
    exp = exp.copy()
    exp.sparse = False
    if random_seed is not None:
        np.random_seed(random_seed)
    for idx in range(len(exp.feature_metadata)):
        exp.data[:, idx] = np.random.permutation(exp.data[:, idx])
    if normalize:
        exp.normalize(10000, inplace=True)
    return exp


def plot_dbbact_terms(exp, region=None, only_exact=False, collapse_per_exp=True, ignore_exp=None, num_terms=50, ignore_terms=[]):
    from sklearn.cluster import AffinityPropagation, OPTICS
    from sklearn import metrics
    import matplotlib.pyplot as plt

    logger.debug('plot_dbbact_terms for %d features' % len(exp.feature_metadata))

    ignore_terms = set(ignore_terms)
    exp = exp.add_terms_to_features('dbbact')
    terms_per_seq = {}
    all_terms = defaultdict(float)
    sequences = exp.feature_metadata.index.values
    # sequences=sequences[:20]
    for cseq in sequences:
        anno = exp.exp_metadata['__dbbact_sequence_annotations'][cseq]
        expterms = {}
        for idx, cannoid in enumerate(anno):
            canno = exp.exp_metadata['__dbbact_annotations'][cannoid]
            # test if region is the same if we require exact region
            if only_exact:
                if region != canno['primer']:
                    continue
            if ignore_exp is not None:
                if canno['expid'] in ignore_exp:
                    continue
            # get the experiment from where the annotation comes
            # if we don't collapse by experiment, each annotation gets a fake unique expid
            if collapse_per_exp:
                cexp = canno['expid']
            else:
                cexp = idx

            if canno['annotationtype'] == 'contamination':
                canno['details'].append(('all', 'contamination'))
            for cdet in canno['details']:
                cterm = cdet[1]
                if cterm in ignore_terms:
                    continue
                if cdet[0] in ['low']:
                    cterm = '-' + cterm
                if cexp not in expterms:
                    expterms[cexp] = defaultdict(float)
                expterms[cexp][cterm] += 1
        cseq_terms = defaultdict(float)
        for cexp, cterms in expterms.items():
            for cterm in cterms.keys():
                cseq_terms[cterm] += 1
        for cterm, ccount in cseq_terms.items():
            all_terms[cterm] += ccount
        terms_per_seq[cseq] = cseq_terms

    all_terms_sorted = sorted(all_terms, key=all_terms.get, reverse=True)

    use_terms = all_terms_sorted[:num_terms]
    use_terms_set = set(use_terms)

    # +1 since we have 'other'
    outmat = np.zeros([len(use_terms) + 1, len(sequences)])
    for seqidx, cseq in enumerate(sequences):
        for cterm in all_terms.keys():
            if cterm in use_terms_set:
                idx = use_terms.index(cterm)
            else:
                idx = len(use_terms)
            outmat[idx, seqidx] += terms_per_seq[cseq][cterm]

    term_names = use_terms + ['other']
    texp = ca.AmpliconExperiment(outmat, pd.DataFrame(term_names, columns=['term'], index=term_names), pd.DataFrame(sequences, columns=['_feature_id'], index=sequences))
    texp.sample_metadata['_sample_id'] = texp.sample_metadata['term']
    ww = texp.normalize()

    print('clustering')
    af = AffinityPropagation().fit(ww.get_data(sparse=False))
    cluster_centers_indices = af.cluster_centers_indices_
    print('found %d clusters' % len(cluster_centers_indices))
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    texp.sample_metadata['cluster'] = labels
    www = texp.aggregate_by_metadata('cluster')

    # now cluster the features
    bb = OPTICS(metric='l1')
    scaled_exp = www.scale(axis='f')
    fitres = bb.fit(scaled_exp.get_data(sparse=False).T)
    bbb = fitres.labels_
    www.feature_metadata['cluster'] = bbb
    www2 = www.aggregate_by_metadata('cluster', axis='f')

    # and plot the pie charts
    # prepare the labels
    ll = www.sample_metadata['_calour_merge_ids'].values
    labels = []
    for clabel in ll:
        clabel = clabel.split(';')
        clabel = clabel[:4]
        clabel = ';'.join(clabel)
        labels.append(clabel)

    plt.figure()
    sqplots = np.ceil(np.sqrt(len(www2.feature_metadata)))
    for idx, cid in enumerate(www2.feature_metadata.index.values):
        plt.subplot(sqplots, sqplots, idx + 1)
        ttt = www2.filter_ids([cid])
        num_features = len(ttt.feature_metadata['_calour_merge_ids'].values[0].split(';'))
        tttdat = ttt.get_data(sparse=False).T[0, :]
        if np.sum(tttdat) > 0:
            tttdat = tttdat / np.sum(tttdat)
        plt.pie(tttdat, radius=1,counterclock=False)
        plt.title(num_features)
    plt.figure()
    plt.pie(ttt.get_data(sparse=False).T[0,:], radius=1)
    plt.legend(labels)

    # # merge the original terms used in each cluster
    # details = []
    # for cmerge_ids in www.sample_metadata['_calour_merge_ids'].values:
    #     cdetails = ''
    #     cmids = cmerge_ids.split(';')
    #     for cmid in cmids:
    #         cdetails += ww.sample_metadata['term'][int(cmid)] + ', '
    #     details.append(cdetails)
    # www.sample_metadata['orig_terms'] = details

    # www = www.cluster_features()

    # www.plot(gui='qt5', xticks_max=None, sample_field='term')
    return www

    # ww=ww.cluster_data(axis=1,transform=ca.transforming.binarize)
    # ww=ww.cluster_data(axis=0,transform=ca.transforming.binarize)
    # ww.plot(gui='qt5', xticks_max=None,sample_field='term')
    # return texp


def trim_seqs(exp, new_len):
    '''trim sequences in the Experiment to length new_len, joining sequences identical on the short length

    Parameters
    ----------
    exp: calour.AmpliconExperiment
        the experiment to trim the sequences (features)
    new_len: the new read length per sequence

    Returns
    -------
    new_exp: calour.AmpliconExperiment
        with trimmed sequences
    '''
    new_seqs = [cseq[:new_len] for cseq in exp.feature_metadata.index.values]
    new_exp = exp.copy()
    new_exp.feature_metadata['new_seq'] = new_seqs
    new_exp = new_exp.aggregate_by_metadata('new_seq', axis='f', agg='sum')
    new_exp.feature_metadata = new_exp.feature_metadata.reindex(new_exp.feature_metadata['new_seq'])
    new_exp.feature_metadata['_feature_id'] = new_exp.feature_metadata['new_seq']
    return new_exp


def filter_features_exp(exp, ids_exp, insert=True):
    '''Filter features in Experiment exp based on experiment ids_exp.
    If insert==True, also insert blank features if feature in ids_exp does not exist in exp

    Parameters
    ----------
    exp: calour.Experiment
        the experiment to filter
    ids_exp: calour.Experiment
        the experiment used to get the ids to filter by
    insert: bool, optional
        True to also insert blank features if feature from ids_exp does not exist in exp

    Returns
    -------
    newexp: calour.Experiment
        exp, filtered and ordered according to ids_exp'''
    if insert:
        texp = exp.join_experiments(ids_exp, field='orig_exp')
    else:
        texp = exp.copy()
    texp = texp.filter_ids(ids_exp.feature_metadata.index)
    texp = texp.filter_samples('orig_exp', 'exp')
    texp.description = exp.description
    drop_cols = [x for x in texp.sample_metadata.columns if x not in exp.sample_metadata.columns]
    texp.sample_metadata.drop(drop_cols, axis='columns', inplace=True)
    return texp


def regress_fit(exp, field, estimator=sklearn.ensemble.RandomForestRegressor(), params=None):
    '''fit a regressor model to an experiment
    Parameters
    ----------
    field : str
        column name in the sample metadata, which contains the variable we want to fit
    estimator : estimator object implementing `fit` and `predict`
        scikit-learn estimator. e.g. :class:`sklearn.ensemble.RandomForestRegressor`
    params: dict of parameters to supply to the estimator

    Returns
    -------
    model: the model fit to the data
    '''
    X = exp.get_data(sparse=False)
    y = exp.sample_metadata[field]

    if params is None:
        # use sklearn default param values for the given estimator
        params = {}

    # deep copy the model by clone to avoid the impact from last iteration of fit.
    model = sklearn.base.clone(estimator)
    model = model.set_params(**params)
    model.fit(X, y)
    return model


def regress_predict(exp, field, model):
    pred = model.predict(exp.data)
    df = pd.DataFrame({'Y_PRED': pred, 'Y_TRUE': exp.sample_metadata[field].values, 'SAMPLE': exp.sample_metadata[field].index.values, 'CV': 0})
    plot_scatter(df, cv=False)
    return df


def classify_fit(exp, field, estimator=sklearn.ensemble.RandomForestClassifier()):
    '''fit a classifier model to the experiment

    Parameters
    ----------
    exp: calour.Experiment
        the experiment to classify
    field: str
        the field to classify
    estimator : estimator object implementing `fit` and `predict`
        scikit-learn estimator. e.g. :class:`sklearn.ensemble.RandomForestRegressor`

    Returns
    -------
    model: the model fit to the data
    '''
    X = exp.get_data(sparse=False)
    y = exp.sample_metadata[field]

    # deep copy the model by clone to avoid the impact from last iteration of fit.
    model = sklearn.base.clone(estimator)
    model.fit(X, y)
    return model


def classify_predict(exp, field, model, predict='predict_proba', plot_it=True):
    '''predict classes on an experiment based on a model from calour_util.classify_fit()

    Parameters
    ----------
    exp: calour.Experiment
    field: str
        Name of the field containing the categories to predict
    model: sklearn.base
        an sklean classifier trained on the fit data. Usually from calour_util.classify_fit()
    predict: str, optional
        the field to use from the model
    plot_it: bool, optional
        True to plot various metrics for the fitted classifier (ROC etc.)
        False to skip plotting

    Returns
    -------
    pandas.Dataframe with the following fields:
    'Y_TRUE': the real value of the samples
    'CV':
    'SAMPLE' the sample id (also the index)
    all-field values: the probability of the sample being this value

    '''
    X = exp.get_data(sparse=False)
    y = exp.sample_metadata[field]
    pred = getattr(model, predict)(X)
    if pred.ndim > 1:
        df = pd.DataFrame(pred, columns=model.classes_)
    else:
        df = pd.DataFrame(pred, columns=['Y_PRED'])
    df['Y_TRUE'] = y.values
    df['CV'] = 1
    df['SAMPLE'] = y.index.values
    # df = pd.DataFrame({'Y_PRED': pred, 'Y_TRUE': exp.sample_metadata[field].values, 'SAMPLE': exp.sample_metadata[field].index.values, 'CV': 0})
    if plot_it:
        ca.training.plot_roc(df, cv=False)
        ca.training.plot_prc(df)
        ca.training.plot_cm(df)
    # roc_auc = classify_get_roc(df)
    # print('ROC is %f' % roc_auc)
    return df


def classify_get_roc(result):
    '''Get the ROC for the given prediction
    NOTE: currently implemented only for 2 classes
    '''
    from sklearn.metrics import roc_auc_score

    classes = np.unique(result['Y_TRUE'].values)
    if len(classes) > 2:
        raise ValueError('More than 2 values for real values. cannot calculate ROC (need to update the function....)')

    roc_auc = roc_auc_score(result['Y_TRUE'].values == classes[0], result[classes[0]])
    return roc_auc
    # for cls in classes:
    #     y_true = result['Y_TRUE'].values == cls
    #     fpr, tpr, thresholds = roc_curve(y_true.astype(int), result[cls])
    #     if np.isnan(fpr[-1]) or np.isnan(tpr[-1]):
    #         logger.warning(
    #             'The class %r is skipped because the true positive rate or '
    #             'false positive rate computation failed. This is likely because you '
    #             'have either no true positive or no negative samples for this class' % cls)
    #     roc_auc2 = auc(fpr, tpr)
    #     roc_auc = roc_auc_score(y_true.astype(int), result[cls])
    #     if roc_auc2 != roc_auc:
    #         raise ValueError('roc_auc %f != self calculated roc %f' % (roc_auc, roc_auc2))
    #     # print(accuracy_score(y_true.astype(int), result[cls]))
    #     # print(balanced_accuracy_score(y_true.astype(int), result[cls]))
    #     print(roc_auc)


def classify_get_accuracy(result, threshold=0.5):
    '''Get the accuracy for the given prediction
    NOTE: currently implemented only for 2 classes
    Parameters
    '''
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    classes = np.unique(result['Y_TRUE'].values)
    if len(classes) > 2:
        raise ValueError('More than 2 values for real values. cannot calculate accuracy (need to update the function....)')
    classes.sort()
    result['predicted_class'] = classes[1]
    result.loc[result[classes[0]] >= threshold, 'predicted_class'] = classes[0]

    # ascore = accuracy_score(result['Y_TRUE'] == classes[0], result[classes[0]] >= threshold)
    ascore = balanced_accuracy_score(result['Y_TRUE'], result['predicted_class'])
    return ascore


def equalize_sample_groups(exp, field):
    '''Filter samples, so equal number of samples with each value in field remain.

    Parameters
    ----------
    exp: calour.Experiment
        the experiment to equalize
    field: str
        the field to equalize by

    Returns
    -------
    newexp: calour.Experiment
        with similar number of samples for each field value
    '''
    num_keep = exp.sample_metadata[field].value_counts().min()
    logger.info('keeping %d samples with each value' % num_keep)
    vals = exp.sample_metadata[field].values
    num_val = defaultdict(int)
    keep = []
    for idx, cval in enumerate(vals):
        if num_val[cval] < num_keep:
            num_val[cval] += 1
            keep.append(idx)
    newexp = exp.reorder(keep, axis='s')
    return newexp


def _paired_diff(data, shuffle_pairs):
    '''Calculate the sample2-sample1 diff over all samples

    NOT IMPLEMENTED'''
    pass


def paired_test(exp, pair_field, order_field, ):
    '''Perform a paired test for pairs of samples

    Parameters
    ----------
    exp: calour.Experiment
    pair_field: str
        Name of the sample field by which samples are paired (2 samples with each value)
        NOTE: values with != 2 samples will be dropped

    Returns
    -------
    '''
    # keep only pairs
    drop_values = []
    for cval, cexp in exp.iterate(pair_field):
        if len(cexp.sample_metadata) != 2:
            logger.info('Value %s has %d samples' % (cval, len(cexp.sample_metadata)))
            drop_values.append(cval)
    if len(drop_values) > 0:
        logger.warning('Dropping %d values with != 2 samples' % len(drop_values))
        exp = exp.filter_samples(pair_field, drop_values, negate=True)

    # prepare the shuffle pairs
    shuffle_pairs = []
    for cval in exp.sample_metadata[pair_field].unique():
        shuffle_pairs.append(np.where(exp.sample_metadata[pair_field].values == cval))
    return exp, shuffle_pairs


def plot_taxonomy(exp, field='taxonomy', num_show=10, show_legend=True, normalize=False):
    '''Plot a taxonomy bar plot (can also be used for terms)

    Parameters
    ----------
    exp: ca.Experiment
    field: str, optional
        name of the feature field to use for the bars (i.e. 'taxonomy' or 'term')
    num_show: int, optional
        number of taxa to show
    show_legend: bool, optional
        True to plot the legend, False to not plot
    '''
    f = plt.figure()
    if normalize:
        exp = exp.normalize(axis='s')
    exp = exp.sort_abundance()
    e1 = exp.reorder(np.arange(-1, -(num_show + 1), -1), axis='f')
    exp = e1
    term_num = []
    terms = exp.feature_metadata.index.values
    data = exp.get_data(sparse=False)
    for cid in range(len(exp.feature_metadata)):
        term_num.append(data[:, cid])
    cbottom = np.zeros(len(term_num[0]))
    for cid in range(len(term_num)):
        ctn = term_num[cid]
        plt.bar(np.arange(len(ctn)), ctn, bottom=cbottom)
        cbottom += ctn
    if show_legend:
        plt.legend(terms)
    return f


def cluster_by_terms(exp, min_score_threshold=0.1, filter_ratio=1.01):
    '''Cluster the experiment features by dbbact terms associated with them

    Parameters
    ----------
    exp: calour.AmpliconExperiment
    min_score_threshold: float, optional
        the minimum term score to keep (anything lower will be changed to it)
    filter_ratio: float, optional
        throw away all features (after applying min_score) with mean > filter_ratio * min_score

    Returns:
    --------
    exp: calour.AmpliconExperiment
        with features clustered by dbbact terms
    '''
    # create an experiment of terms (as features) X features (as samples)
    db = db = ca.database._get_database_class('dbbact')
    logger.info('getting per-feature terms for %d terms' % len(exp.feature_metadata))
    texp = db.sample_term_scores(exp, term_type='fscore', axis='f')
    texp.data[texp.data < min_score_threshold] = min_score_threshold
    texp = texp.filter_by_data('abundance', axis='s', cutoff=min_score_threshold * filter_ratio)
    logger.info('after filtering terms with < %f score threshold, %d remaining' % (min_score_threshold, len(texp.sample_metadata)))
    texp = texp.cluster_data(axis='s', metric='canberra')
    new_exp = exp.filter_ids(texp.sample_metadata.index)
    return new_exp

from collections import defaultdict, OrderedDict
from logging import getLogger, NOTSET, basicConfig
from pkg_resources import resource_filename
from logging.config import fileConfig
import os

import numpy as np
import scipy.stats
from statsmodels.sandbox.stats.multicomp import multipletests
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib as mpl
import pandas as pd
import sklearn
import sklearn.ensemble

from IPython.display import display

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

    if isinstance(equal_fields, str):
        equal_fields = [equal_fields]

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
        if len(cexp.sample_metadata[group_field].unique()) < len(exp.sample_metadata[group_field].unique()):
            logger.info('value %s in not present in all groups' % cval)
            continue
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
            logger.info('not 1 sample for group1: %s' % cid)
            continue
        if len(pos2) != 1:
            logger.info('not 1 sample for group2: %s' % cid)
            continue
        cdat1 = data[pos1, :]
        cdat2 = data[pos2, :]
        cdat1[cdat1 < min_thresh] = min_thresh
        cdat2[cdat2 < min_thresh] = min_thresh
        newexp.data[pos1, :] = np.log2(cdat1 / cdat2)
        keep.append(pos1[0])
    logger.info('found %d ratios' % len(keep))
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
        if len(exp.feature_metadata) == 0:
            raise ValueError('No features remaining after filtering. did you supply a correct list?')
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


def alpha_diversity_as_feature(exp, method='entropy'):
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

    logger.info('clustering')
    af = AffinityPropagation().fit(ww.get_data(sparse=False))
    cluster_centers_indices = af.cluster_centers_indices_
    logger.info('found %d clusters' % len(cluster_centers_indices))
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


def trim_seqs(exp, new_len, pos='end'):
    '''trim sequences in the Experiment to length new_len, joining sequences identical on the short length

    Parameters
    ----------
    exp: calour.AmpliconExperiment
        the experiment to trim the sequences (features)
    new_len: the new read length per sequence (if pos=='end' or length to trim (if pos=='start'))
    pos: str, optional
        'end' to trim from end of sequence
        'start' to trim from start of sequence

    Returns
    -------
    new_exp: calour.AmpliconExperiment
        with trimmed sequences
    '''
    if pos == 'end':
        new_seqs = [cseq[:new_len] for cseq in exp.feature_metadata.index.values]
    elif pos == 'start':
        new_seqs = [cseq[new_len:] for cseq in exp.feature_metadata.index.values]
    new_exp = exp.copy()
    new_exp.feature_metadata['new_seq'] = new_seqs
    new_exp = new_exp.aggregate_by_metadata('new_seq', axis='f', agg='sum')
    # new_exp.feature_metadata = new_exp.feature_metadata.reindex(new_exp.feature_metadata['new_seq'])
    new_exp.feature_metadata = new_exp.feature_metadata.set_index('new_seq',drop=False)
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
    db = ca.database._get_database_class('dbbact')
    logger.info('getting per-feature terms for %d terms' % len(exp.feature_metadata))
    texp = db.sample_term_scores(exp, term_type='fscore', axis='f')
    texp.data[texp.data < min_score_threshold] = min_score_threshold
    texp = texp.filter_by_data('abundance', axis='s', cutoff=min_score_threshold * filter_ratio)
    logger.info('after filtering terms with < %f score threshold, %d remaining' % (min_score_threshold, len(texp.sample_metadata)))
    texp = texp.cluster_data(axis='s', metric='canberra')
    new_exp = exp.filter_ids(texp.sample_metadata.index)
    return new_exp


def plot_diff_term_tree(exp, term, relation='both', keep_only_diff=True, simplify=True, colormap='coolwarm'):
    '''Plot a cytoscape tree of a given term and it's parents/children
    NOTE: requires ipycytoscape

    Parameters
    ----------
    exp: ca.AmpliconExperiment
        result of the plot_diff_abundance_enrichment(). Run it with: term_type='parentterm', alpha=1, add_single_exp_warning=False .

    term: str
        The term to plot the tree for (i.e. excreta)

    Returns
    -------
    graph: ipycytoscape.CytoscapeWidget() representation of the graph
    '''
    import ipycytoscape
    import networkx as nx

    # get the parent/child terms graph
    db = ca.database._get_database_class('dbbact')
    family = db.db.get_term_family_graph([term], relation=relation)
    g = nx.node_link_graph(family)

    # add size/color and pval/odif to graph
    cm = plt.get_cmap(colormap)
    qq = exp.feature_metadata.copy()
    qq.index = qq.index.str.lower()
    for cnode in g.nodes:
        if 'name' not in g.nodes[cnode]:
            print(g.nodes[cnode])
            g.nodes[cnode]['name'] = str(cnode)
            continue
        cname = g.nodes[cnode]['name']
        if cname in qq.index:
            g.nodes[cnode]['pval'] = qq.loc[cname]['pvals']
            g.nodes[cnode]['odif'] = qq.loc[cname]['odif']
            if qq.loc[cname]['odif'] > 0:
                    pvcol = 0.5
            else:
                    pvcol = -0.5
            ccolor = ((1 - qq.loc[cname]['pvals']) * pvcol + 0.5)
            g.nodes[cnode]['color'] = list(np.array(cm(ccolor * 255)) * 255)[:3]
            g.nodes[cnode]['size'] = np.max([np.abs(qq.loc[cname]['odif']) * 500, 1])

    # delete term nodes without odif field filter (so we won't get a huge forest)
    if keep_only_diff:
        while True:
            done = True
            for cnode in g:
                if 'odif' in g.nodes[cnode]:
                    continue
                if len(g.out_edges(cnode)) > 0:
                    for cin in g.in_edges(cnode):
                        g.add_edge(list(cin)[0], list(g.out_edges(cnode))[0][1])
                g.remove_node(cnode)
                done = False
                break
            if done:
                break

    # simplify the graph, removing terms that have 1 parent, 1 child
    if simplify:
        done = False
        while not done:
            done = True
            for cnode in g.nodes():
                if len(g.in_edges(cnode)) != 1:
                    continue
                if len(g.out_edges(cnode)) != 1:
                    continue
                g.add_edge(list(g.in_edges(cnode))[0][0], list(g.out_edges(cnode))[0][1])
                g.remove_node(cnode)
                done = False
                break

    # plot the graph
    directed = ipycytoscape.CytoscapeWidget()
    directed.graph.add_graph_from_networkx(g)
    directed.set_style([{'selector': 'node',
                         'css': {'content': 'data(name)',
                                 'text-valign': 'center',
                                 'color': 'white',
                                 'text-outline-width': 2,
                                 'text-outline-color': 'green',
                                 'background-color': 'data(color)',
                                 'width': 'data(size)',
                                 'height': 'data(size)'}
                         },
                        {'selector': 'edge',
                         'style': {'width': 4,
                                   'line-color': '#9dbaea',
                                   'target-arrow-shape': 'triangle',
                                   'target-arrow-color': '#9dbaea',
                                   'curve-style': 'bezier'}},
                        {'selector': ':selected',
                         'css': {'background-color': 'black',
                                 'line-color': 'black',
                                 'target-arrow-color': 'black',
                                 'source-arrow-color': 'black',
                                 'text-outline-color': 'black'}}
                        ])
    directed.set_layout(name='dagre')
    directed.set_tooltip_source('name')
    display(directed)
    return directed


def bicluster(exp, cluster_method='std', min_prevalence=0.05, std_thresh=None, transform='none', max_iterations=20, subset=None, start_axis='f', random_seed=None):
    '''Do unsupervised bi-clustering

    Parameters
    ----------
    exp: calour.AmpliconExperiment
    cluster_method: str, optional
        ther clustering method to use. options are:
        'barkai': based on the method of Ihmels et al (https://doi.org/10.1038/ng941). iterative clustering using stable clusters with lower standard deviation than expected by chance
        'std': standard deviation difference between the two clusters
        'linear': use two-line fit to identify cluster break point
    min_prevalence: float or None, optional
        the minimal prevalence for features to keep for the clustering (for filter_prevalence()) before the biclustering
        if None, do not do filter_prevalence()
    std_thresh: float or None, optional
        if not None, the mean/std threshold to keep in the cluster
        if None, randomize a threshold before the run
    transform: str, optional
        the transform on the data before the buclustering. options:
            'binarydata': transform to presence/absence
            'rankdata': rank each feature on the samples
            'log2data': log2 pf the data. numbers<1 are changed to 1
            'none': no transform
    max_iterations: int, optional
        the number of feature/sample iterations to perform
    subset: int or list if int or None, optional
        the number of samples or features to include in the initial seed samples for the algorithm (depending of start_axis).
        if list of int, these are the sample positions (in the sample_metadata or feature_metadata dataframe) to use as the initial samples
        If None, randomly select a part of the number of samples or features to use (uniform)
    start_axis: 'f' or 's', optional
        the starting axis - 'f' to first cluster features, 's' to first cluster samples
    random_seed: int or None, optional
        if not none, set the numpy random seed prior to running the clustering

    Returns
    -------
    exp: calour.Experiment with '_bicluster' field in sample_metadata and feature_metadata (with values 1 for the cluster, 0 for the rest) or None if clustering failed
    '''
    import scipy as sp
    if random_seed is not None:
        np.random.seed(random_seed)
    exp = exp.copy()
    if min_prevalence is not None:
        exp = exp.filter_prevalence(min_prevalence)

    # The function for two-lines fit to a curve. used with the 'linear' option in the cluster_method
    def two_lines(x, breakpoint, slope1, offset1, slope2):
        res = np.zeros(len(x))
        l1x = np.where(x < breakpoint)[0]
        l2x = np.where(x >= breakpoint)[0]
        res[l1x] = x[l1x] * slope1 + offset1
        offset2 = slope1 * breakpoint + offset1
        res[l2x] = (x[l2x] - breakpoint) * slope2 + offset2
        return res

    # if we want to first cluster features, transform the experiment at the beginning and end
    if start_axis == 'f':
        exp.data = exp.data.T
        fmd = exp.feature_metadata
        exp.feature_metadata = exp.sample_metadata
        exp.sample_metadata = fmd

    data = exp.get_data(sparse=False, copy=True)
    if transform == 'binarydata':
        data = (data > 0)
    elif transform == 'rankdata':
        data = scipy.stats.rankdata(data, axis=0)
    elif transform == 'log2data':
        data[data < 1] = 1
        data = np.log2(data)
    elif transform == 'none':
        pass
    else:
        raise ValueError('transform %s not supported' % transform)

    # prepare the scaled data (mean 0 std 1) on each axis separately
    data_norm_per_feature = sklearn.preprocessing.scale(data, axis=0)
    data_norm_per_sample = sklearn.preprocessing.scale(data, axis=1)
    # data_norm_per_sample = data.copy()

    num_samples = len(exp.sample_metadata)
    num_features = len(exp.feature_metadata)

    if subset is None:
        subset = int(num_samples * np.random.rand())
        logger.debug('subset is None, so randmoly selected initial number of samples: %d' % subset)

    if isinstance(subset, list) or isinstance(subset, tuple) or isinstance(subset, np.ndarray):
        samples = subset
    else:
        samples = np.random.permutation(np.arange(num_samples))
        samples = samples[:subset]
    features = np.arange(num_features)

    if std_thresh is None:
        # std_thresh = np.abs(np.random.rand()*2 + 2)
        std_thresh = np.abs(np.random.rand() * 0.3 + 0.7)
        logger.debug('std_thresh is None so randomly set std threshold: %f' % std_thresh)

    for citer in range(max_iterations):
        # initialize the per-sample scaling score (for the naama barkai algorithm)
        # https://www.nature.com/articles/ng941z
        for caxis, cdim in zip(['s', 'f'], [0, 1]):
            if caxis == 's':
                cdata = data_norm_per_feature.copy()
                mask = np.ones([num_samples], dtype=bool)
                mask[samples] = False
                cg = cdata[samples, :]
                cng = cdata[mask, :]
            else:
                cdata = data_norm_per_sample.copy()
                # cdata = cdata * np.tile(sample_scores, )
                mask = np.ones([num_features], dtype=bool)
                mask[features] = False
                cg = cdata[:, features]
                cng = cdata[:, mask]

            # identify the samples or features to keep. Put their indices in the variable "ok"

            if cluster_method == 'std':
                # get the std on the other dimension (i.e. for each feature if we are looking on the samples)
                cstd = np.std(cdata, axis=cdim)
                mean_in = np.mean(cg, axis=cdim)
                mean_out = np.mean(cng, axis=cdim)
                diff = (mean_in - mean_out) / (cstd + 0.0000001)
                ok = np.where(diff > std_thresh)[0]

            elif cluster_method == 'barkai':
                cstd = np.std(cdata, axis=cdim)
                cstd = np.std(cg, axis=cdim)
                mean_in = np.mean(cg, axis=cdim)
                mean_all = np.mean(cdata, axis=cdim)
                diff = (mean_in - mean_all) / (cstd + 0.0000001)
                ok = np.where(diff > std_thresh / np.sqrt(len(features)))[0]
                # if citer % 6 == 0:
                #     std_thresh = std_thresh * 1.1

            elif cluster_method == 'dsfdr':
                labels = np.zeros(mask.shape)
                labels[mask] = 1
                # print(cdata.shape)
                # print(labels.shape)
                if caxis == 'f':
                    ccdata = cdata
                else:
                    ccdata = cdata.T
                reject, tstat, pvals, qvals = ca.dsfdr.dsfdr(ccdata, labels)
                # print('keeping %d for axis %s' % (np.sum(reject), caxis))
                ok = np.where(reject)[0]

            elif cluster_method == 'linear':
                cvn = np.mean(cg, axis=cdim) - np.mean(cng, axis=cdim)
                ci = np.argsort(cvn)
                cv = cvn[ci]
                cv = np.log2(1 + cv - np.min(cv))

                midpoint = int(len(cv) / 2)
                initial_breakpoint = int(len(cv) * 3 / 4)
                # res = scipy.optimize.curve_fit(two_lines, np.arange(len(cv)), cv, p0=(initial_breakpoint, (cv[midpoint] - cv[0]) * 2 / len(cv), 0, (cv[-1] - cv[midpoint]) * 2 / len(cv)))
                score=[]
                d1s=[]
                d2s=[]
                for cbp in range(int(len(cv)*2 / 3),len(cv)-5):
                    x1 = np.arange(cbp)
                    res = sp.stats.linregress(x1, cv[:cbp])
                    a1=res.slope
                    b1=res.intercept
                    # a1,b1=np.polyfit(x1,cv[:cbp],1)
                    d1 = np.sum(np.abs(cv[:cbp] - (a1*x1+b1)))

                    x2 = np.arange(cbp,len(cv))
                    # a2,b2=np.polyfit(x2,cv[cbp:],1)
                    res2 = sp.stats.linregress(x2, cv[cbp:])
                    a2=res2.slope
                    b2=res2.intercept
                    d2 = np.sum(np.abs(cv[cbp:] - (a2*x2+b2)))
                    score.append(d1+d2)
                    d1s.append(d1)
                    d2s.append(d2)
                # plt.figure()
                # plt.plot(np.arange(len(score)),score,'b')
                # plt.plot(np.arange(len(score)),d1s,'r')
                # plt.plot(np.arange(len(score)),d2s,'k')
                # plt.legend(['total','d1','d2'])
                minpos=np.argmin(score)+int(len(cv)*2 / 3)
                # print(minpos)
                # plt.figure()
                # xp = np.arange(len(cv))
                # plt.plot(xp,cv,'.k')
                # # plt.plot(xp,two_lines(xp,*res[0]),'r')
                # plt.plot(minpos,cv[minpos],'ob')
                # plt.title(caxis)
                # ok = ci[int(res[0][0]):]
                ok = ci[minpos:]
            else:
                raise ValueError('cluster_method %s not supported' % cluster_method)
            if caxis == 's':
                features = ok
            else:
                samples = ok
            if len(ok) == 0:
                logger.debug('empty cluster reached')
                return None

    # print('samples: %d, features: %d' % (len(samples), len(features)))
    if len(samples) == 0 or len(features) == 0:
        logger.info('No stable cluster found')
        return None

    mask = np.zeros([num_samples])
    mask[samples] = 1
    exp.sample_metadata['_bicluster'] = mask
    exp.sample_metadata['_bicluster'] = exp.sample_metadata['_bicluster'].astype(str)
    mask = np.zeros([num_features])
    mask[features] = 1
    exp.feature_metadata['_bicluster'] = mask
    exp.feature_metadata['_bicluster'] = exp.feature_metadata['_bicluster'].astype(str)

    not_samples = np.delete(np.arange(num_samples), samples)
    not_features = np.delete(np.arange(num_features), features)
    exp = exp.reorder(np.hstack([samples, not_samples]), axis='s')
    exp = exp.reorder(np.hstack([features, not_features]), axis='f')

    # now flip back if needed
    if start_axis == 'f':
        exp.data = exp.data.T
        fmd = exp.feature_metadata
        exp.feature_metadata = exp.sample_metadata
        exp.sample_metadata = fmd
    return exp


def bicluster_enrichment(exp, cluster_method='std', min_prevalence=0.05, std_thresh=None, transform='none', max_iterations=20, subset=None, start_axis='f', random_seed=None, alpha=0.1, num_clusters=0):
    '''Do unsupervised bi-clustering, and test the resulting cluster for feature (via dbBact) and sample (via all the metadata fields) enrichment

    Parameters
    ----------
    exp: calour.AmpliconExperiment
    cluster_method: str, optional
        ther clustering method to use. options are:
        'barkai': based on the method of Ihmels et al (https://doi.org/10.1038/ng941). iterative clustering using stable clusters with lower standard deviation than expected by chance
        'std': standard deviation difference between the two clusters

    min_prevalence: float or None, optional
        the minimal prevalence for features to keep for the clustering (for filter_prevalence()) before the biclustering
        if None, do not do filter_prevalence()
    std_thresh: float or None, optional
        if not None, the mean/std threshold to keep in the cluster
        if None, randomize a threshold before the run
    transform: str, optional
        the transform on the data before the buclustering. options:
            'binarydata': transform to presence/absence
            'rankdata': rank each feature on the samples
            'log2data': log2 pf the data. numbers<1 are changed to 1
            'none': no transform
    max_iterations: int, optional
        the number of feature/sample iterations to perform
    subset: int or list if int or None, optional
        the number of samples to include in the initial seed samples for the algorithm.
        if list of int, these are the sample positions (in the sample_metadata dataframe) to use as the initial samples
        If None, randomly select a part of the number of samples to use (uniform)
    start_axis: 'f' or 's', optional
        the starting axis - 'f' to first cluster features, 's' to first cluster samples
    random_seed: int or None, optional
        if not none, set the numpy random seed prior to running the clustering
    alpha: float, optional
        the alpha dsFDR threshold for the sample metadata enrichment
    num_clusters: int, optional
        if not None, the number of non-overlapping clusters to identify
        if 0, run only once (one cluster)

    Returns
    -------
    exp: calour.AmpliconExperiment
        after the prevalence filtering, and ordered according to the cluster. Cluster info is in the field: '_bicluster' (both for sample_metadata and feature_metadata)
    e: calour.AmpliconExperiment
        The enriched dbBact terms for the features (from plot_diff_abundance_enrichment)
        Negative value in the feature_metadata._calour_stat (or '1.0' in the _calour_dir) are enriched in the cluster, positive values (or '0.0' in _calour_dir) are enriched in samples not in the cluster
    dd: calour.AmpliconExperiment:
        The enriched metadata terms. Features are FIELD_:_VALUE, Samples are the original samples
        Negative value in the feature_metadata._calour_stat (or '1.0' in the _calour_dir) are enriched in the cluster, positive values (or '0.0' in _calour_dir) are enriched in samples not in the cluster
    '''
    if random_seed is not None:
        np.random.seed(random_seed)

    # we do the min_prevalence filtering here since we run multiple times
    if min_prevalence is not None:
        exp = exp.filter_prevalence(min_prevalence)

    orig_exp = exp.copy()

    cluster_features = []
    cluster_samples = []
    finished = False
    num_tries = 0

    while not finished:
        num_tries += 1
        exp = bicluster(orig_exp, cluster_method=cluster_method, min_prevalence=None, std_thresh=std_thresh, transform=transform, max_iterations=max_iterations, subset=subset, start_axis=start_axis, random_seed=random_seed)

        # check if it a new cluster (by examining overlap < 90% with all previous clusters of samples and features)
        if exp is not None:
            cf = set(exp.feature_metadata[exp.feature_metadata._bicluster == '1.0']._feature_id.values)
            ncf = set(exp.feature_metadata[exp.feature_metadata._bicluster == '0.0']._feature_id.values)
            new_features = True
            for cfeat in cluster_features:
                if len(cfeat.intersection(cf)) > 0.9 * np.max([len(cfeat), len(cf)]):
                    new_features = False
                    break
                if len(cfeat.intersection(ncf)) > 0.9 * np.max([len(cfeat), len(ncf)]):
                    new_features = False
                    break
            cs = set(exp.sample_metadata[exp.sample_metadata._bicluster == '1.0']._sample_id.values)
            ncs = set(exp.sample_metadata[exp.sample_metadata._bicluster == '0.0']._sample_id.values)
            new_samples = True
            for csamp in cluster_samples:
                if len(csamp.intersection(cs)) > 0.9 * np.max([len(csamp), len(cs)]):
                    new_samples = False
                    break
                if len(csamp.intersection(ncs)) > 0.9 * np.max([len(csamp), len(ncs)]):
                    new_samples = False
                    break
            logger.debug('got cluster with %d samples, %d features' % (len(cs), len(cf)))
            if new_features and new_samples:
                cluster_features.append(cf)
                cluster_samples.append(cs)
                logger.debug('it is a new cluster. added')
            else:
                logger.debug('cluster overlaps with old cluster')
        # if len(cluster_features) >= num_clusters:
        #     finished = True
        if num_tries > num_clusters * 5:
            finished = True

    if len(cluster_features) == 0:
        logger.warning('did not find any clusters')
        return

    logger.info('found %d unique clusters' % len(cluster_samples))

    # calculate the cluster score for each cluster
    scores = np.zeros([len(cluster_samples)])
    for idx, (csamp, cfeat) in enumerate(zip(cluster_samples, cluster_features)):
        cexp = orig_exp.filter_ids(cfeat)
        mean_in = cexp.filter_ids(csamp, axis='s').data.mean()
        mean_out = cexp.filter_ids(csamp, axis='s', negate=True).data.mean()
        scores[idx] = mean_in / mean_out

    idx = np.argsort(scores)
    cluster_features = [cluster_features[x] for x in idx[::-1]]
    cluster_samples = [cluster_samples[x] for x in idx[::-1]]
    scores = scores[idx]

    # add all annotations once so will speed up the enrichment analysis
    logger.debug('Adding dbBact annotations for enrichment analysis')
    db = ca.database._get_database_class('dbbact')
    db.add_all_annotations_to_exp(orig_exp)

    num_samples = len(orig_exp.sample_metadata)

    exp_list = []
    for csamp, cfeat in zip(cluster_samples, cluster_features):
        exp = orig_exp.copy()
        print('***************************************')
        csamp = list(csamp)
        cfeat = list(cfeat)
        exp.sample_metadata['_bicluster'] = '0.0'
        exp.sample_metadata.loc[csamp, '_bicluster'] = '1.0'
        exp.feature_metadata['_bicluster'] = '0.0'
        exp.feature_metadata.loc[cfeat, '_bicluster'] = '1.0'
        # return exp
        exp = exp.sort_samples('_bicluster')
        exp = exp.sort_by_metadata('_bicluster', axis='f')
        exp.plot(gui='cli', barx_fields=['_bicluster'], bary_fields=['_bicluster'])

        exp_list.append(exp)

        print('*** features')
        dd = exp.diff_abundance('_bicluster', '0.0', '1.0', alpha=1)
        f, e = dd.plot_diff_abundance_enrichment()
    #     print(e.feature_metadata.to_html())
        display(e.feature_metadata)

        # metadata enrichment
        mddat = np.zeros([0, num_samples])
        field_vals = []
        for cfield in exp.sample_metadata.columns:
            cunique = exp.sample_metadata[cfield].unique()
            num_unique = len(cunique)

            # 1 value - no enrichment
            if num_unique == 1:
                continue

            # a lot of values - so look for correlation (if numeric)
            if num_unique > np.max([num_samples / 10, 3]):
                try:
                    cdata = exp.sample_metadata[cfield].astype(float)
                    mddat = np.vstack([mddat, np.array(cdata)])
                    field_vals.append('%s_:_continuous' % (cfield))
                    continue
                except:
                    continue

            # a few categories
            for cval in cunique:
                field_vals.append('%s_:_%s' % (cfield, cval))
                cdata = (exp.sample_metadata[cfield] == cval).astype(bool)
                mddat = np.vstack([mddat, np.array(cdata)])
        # print(field_vals)

        mdexp = ca.Experiment(mddat.T, exp.sample_metadata, pd.DataFrame(field_vals, columns=['_feature_id'], index=field_vals))
        dd = mdexp.diff_abundance('_bicluster', '0.0', '1.0', alpha=alpha)
        print('*** samples')
    #     print(dd.feature_metadata.to_html())
        display(dd.feature_metadata)
        print('------------------------------')
    return exp_list, e, dd


def bicluster_analysis(exp, min_prevalence=0.2, include_subsets=True):
    clusters = []
    cluster_exps = []
    exp = exp.copy()
    subset = None

    if min_prevalence is not None:
        exp = exp.filter_prevalence(min_prevalence)

    # add dbBact annotations (so we do it only once and then use for all dbBact term enrichment per-cluster)
    db = ca.database._get_database_class('dbbact')
    db.add_all_annotations_to_exp(exp, force=True)

    use_exps = [exp]

    # find biclusters
    for i in range(100):
        cexp = use_exps[np.random.randint(len(use_exps))]
        print(cexp)
        # res = bicluster(cexp, cluster_method='barkai', transform='binarydata', max_iterations=20, start_axis='f', subset=None)
        res = bicluster(cexp, cluster_method='barkai', transform='rankdata', max_iterations=20, start_axis='f', subset=None)
        print('clustered')
        if res is None:
            subset = None
            continue
        samples = set(np.where(res.sample_metadata['_bicluster'] == '1.0')[0])
        features = set(np.where(res.feature_metadata['_bicluster'] == '1.0')[0])
        not_samples = set(np.where(res.sample_metadata['_bicluster'] == '0.0')[0])
        not_features = set(np.where(res.feature_metadata['_bicluster'] == '0.0')[0])
        foundit = False
        if len(samples) < 5:
            continue
        if len(features) < 5:
            continue
        for (csamp, cfeat) in clusters:
            if len(csamp.intersection(samples)) > 0.8 * np.max([len(csamp), len(samples)]):
                if len(cfeat.intersection(features)) > 0.8 * np.max([len(cfeat), len(features)]):
                    foundit = True
                    break
            if len(csamp.intersection(not_samples)) > 0.8 * np.max([len(csamp), len(not_samples)]):
                if len(cfeat.intersection(not_features)) > 0.8 * np.max([len(cfeat), len(not_features)]):
                    foundit = True
                    break
        if foundit:
            print('found it')
            subset = None
        else:
            print('not found')
            clusters.append((samples, features))
            cluster_exps.append(res)
            if include_subsets:
                print('adding subset')
                tt = res.filter_samples('_bicluster', '0.0').filter_by_metadata('_bicluster', ['0.0'], axis='f')
                if len(tt.sample_metadata) > 10 and len(tt.feature_metadata) > 10:
                    use_exps.append(tt)
                tt = res.filter_samples('_bicluster', '1.0').filter_by_metadata('_bicluster', ['1.0'], axis='f')
                if len(tt.sample_metadata) > 10 and len(tt.feature_metadata) > 10:
                    use_exps.append(tt)
                print('ok')
            subset = np.random.permutation(np.where(res.sample_metadata == '0.0')[0])
            if len(subset) > 20:
                subset = subset[:20]
            print('ok2')
    print('found %d biclusters' % len(clusters))

    for idx, cexp in enumerate(cluster_exps):
        print('idx: %d (%r)' % (idx, cexp))
        cexp = cexp.sort_samples('_bicluster')
        cexp = cexp.sort_by_metadata('_bicluster', axis='f')
        alpha = 0.1
        num_samples = len(cexp.sample_metadata)

        cexp.plot(gui='cli', barx_fields=['_bicluster'], bary_fields=['_bicluster'])

        print('*** features')
        dd = cexp.diff_abundance('_bicluster', '0.0', '1.0', alpha=1)
        f, e = dd.plot_diff_abundance_enrichment()
    #     print(e.feature_metadata.to_html())
        # display(e.feature_metadata)

        # metadata enrichment
        mddat = np.zeros([0, num_samples])
        field_vals = []
        for cfield in cexp.sample_metadata.columns:
            cunique = cexp.sample_metadata[cfield].unique()
            num_unique = len(cunique)

            # 1 value - no enrichment
            if num_unique == 1:
                continue

            # a lor of values - so look for correlation (if numeric)
            if num_unique > np.max([num_samples / 10, 3]):
                try:
                    cdata = cexp.sample_metadata[cfield].astype(float)
                    mddat = np.vstack([mddat, np.array(cdata)])
                    field_vals.append('%s_:_continuous' % (cfield))
                    continue
                except:
                    continue

            # a few categories
            for cval in cunique:
                field_vals.append('%s_:_%s' % (cfield, cval))
                cdata = (cexp.sample_metadata[cfield] == cval).astype(bool)
                mddat = np.vstack([mddat, np.array(cdata)])

        mdexp = ca.Experiment(mddat.T, cexp.sample_metadata, pd.DataFrame(field_vals, columns=['_feature_id'], index=field_vals))
        dd = mdexp.diff_abundance('_bicluster', '0.0', '1.0', alpha=alpha)
        print('*** samples')
    #     print(dd.feature_metadata.to_html())
        display(dd.feature_metadata)

        #     cu.splot(exp,'_bicluster',barx_fields=['_bicluster'],bary_fields=['_bicluster'])
    return cluster_exps


def health_index(exp, method=None, bad_features=None, good_features=None, field_name='_health_index', use_features='both', return_filtered=False):
    '''Calcuulate the per-sample health index (from "Meta-analysis defines predominant shared microbial responses in various diseases and a specific inflammatory bowel disease signal", https://doi.org/10.1186/s13059-022-02637-7)

    Parameters
    ----------
    exp: ca.AmpliconExperiment
        the experiment to calculate the per-sample health index for.
        NOTE: must be V4 region
    method: str or None, optional
        the method to normalize the experiment prior to calculating the health index
        options are:
            None: no transform (the freq method for health index)
            'binarydata': do presence/absence based health index
            'rankdata': transform each feature to rank (across all samples)
            'rankdata2': transform each feature to rank (across all features in each sample)
    bad_features: str or None, optional
        if None, use the default health-index bacteria file (from Abbas-Egbariya, Haya, et al. "Meta-analysis defines predominant shared microbial responses in various diseases and a specific inflammatory bowel disease signal." Genome Biology 23.1 (2022): 1-23.)
        if not None, location of the feature_metadata file (tsv, row per feature, first column is the feature sequence) of features that go up in disease (i.e. higher in not healthy)
    good_features: str of None, optional
        similar to bad_features, but for the features that are lower in disease (i.e. higher in healthy)
    field_name: str, optional
        name of the sample_metadata column for the health index results
    use_features: str, optional
        'both': use up and down regulated bacteria
        'healthy': use only the bacteria higher in health
        'disease': use only bacteria lower in health
    return_filtered: bool, optional
        if True, return the experiment with only the health index features instead of all the features
    Returns
    -------
    ca.AmpliconExperiment
        with a new sample_metadata field named '_health_index'
    '''
    # load the non-specific bacteria
    data_dir = resource_filename(__package__, 'data')
    if bad_features is None:
        bad_features = os.path.join(data_dir, 'nonspecific-up_feature.txt')
    if good_features is None:
        good_features = os.path.join(data_dir, 'nonspecific-down_feature.txt')

    ns_good = pd.read_csv(good_features, sep='\t', index_col=0)
    ns_good['dir'] = 'good'
    ns_bad = pd.read_csv(bad_features, sep='\t', index_col=0)
    ns_bad['dir'] = 'bad'
    nsf = ns_good.merge(ns_bad, how='outer')

    print('%d good, %d bad' % (len(ns_good), len(ns_bad)))

    # create a single experiment with the good and bad features
    newexp = exp.copy()
    newexp = newexp.filter_ids(nsf._feature_id.values)
    newexp.feature_metadata['_health_index_group'] = 'good'
    newexp.feature_metadata.loc[newexp.feature_metadata.index.isin(ns_bad['_feature_id']), '_health_index_group'] = 'bad'

    newexp.sparse = False

    # the epsilon to add in case we get 0 bacteria in the good or bad groups
    eps = 0.1
    if method is None:
        eps = 1
    elif method == 'binarydata':
        newexp.data = (newexp.data > 0)
    elif method == 'rankdata':
        newexp.data = scipy.stats.rankdata(newexp.data, axis=0)
    elif method == 'rankdata2':
        newexp.data = scipy.stats.rankdata(newexp.data, axis=1)
    else:
        raise ValueError('method %s not supported' % method)

    bad_ids = nsf[nsf['dir'] == 'bad']['_feature_id'].values
    good_ids = nsf[nsf['dir'] == 'good']['_feature_id'].values

    bad_exp = newexp.filter_ids(bad_ids)
    bad_score = bad_exp.data.sum(axis=1)
    good_exp = newexp.filter_ids(good_ids)
    good_exp.save('./tt')
    good_score = good_exp.data.sum(axis=1)
    logger.info('found %d bad, %d good' % (len(bad_exp.feature_metadata), len(good_exp.feature_metadata)))
    # we do "-" so high is healthy
    if use_features == 'both':
        dbi = np.log2((good_score + eps) / (bad_score + eps))
    elif use_features == 'disease':
        dbi = - np.log2((bad_score + eps))
    elif use_features == 'healthy':
        dbi = np.log2((good_score + eps))
    else:
        raise ValueError('use_features method not supported (%s)' % use_features)

    if return_filtered:
        exp = newexp
        print('returning filtered')
    else:
        exp = exp.copy()
    exp.sample_metadata[field_name] = dbi
    return exp


def plot_sample_term_scatter(exp, term1, term2, ignore_exp=True, transform='rankdata', field=None):
    db = ca.database._get_database_class('dbbact')
    db.add_all_annotations_to_exp(exp, max_id=None, force=False, get_parents=True)
    exp = exp.copy()
    sequence_terms = exp.databases['dbbact']['sequence_terms']
    sequence_annotations = exp.databases['dbbact']['sequence_annotations']
    annotations = exp.databases['dbbact']['annotations']
    term_info = exp.databases['dbbact']['term_info']
    focus_terms = [term1, term2]
    # focus_terms=None
    if focus_terms is not None:
        focus_terms = set(focus_terms)
        ok_annotations = {}
        for cid, cannotation in annotations.items():
            # check if an
            found_terms = set()
            for cdetail in cannotation['details']:
                if cdetail[1] in focus_terms:
                    found_terms.add(cdetail[1])
            if len(found_terms) > 0:
                ok_annotations[cid] = cannotation
        logger.info('keeping %d out of %d annotations with all the terms (%s)' % (len(ok_annotations), len(annotations), focus_terms))
        for k, v in sequence_annotations.items():
            nv = [x for x in v if x in ok_annotations]
            sequence_annotations[k] = nv

    # change the sequence annotations from dict to list of tuples
    sequence_annotations = [(k, v) for k, v in sequence_annotations.items()]

    # set the experiments to ignore in the wordcloud
    if ignore_exp is True:
        if exp is None:
            raise ValueError('Cannot use ignore_exp=True when exp is not supplied')
        ignore_exp = db.db.find_experiment_id(datamd5=exp.info['data_md5'], mapmd5=exp.info['sample_metadata_md5'], getall=True)
        if ignore_exp is None:
            logger.warn('No matching experiment found in dbBact. Not ignoring any experiments')
        else:
            logger.info('Found %d experiments (%s) matching current experiment - ignoring them.' % (len(ignore_exp), ignore_exp))
    if ignore_exp is None:
        ignore_exp = []

    # we need to rekey the annotations with an str (old problem...)
    annotations = {str(k): v for k, v in annotations.items()}

    logger.info('Getting per-sequence term f-scores')
    seq_scores = {}
    for cseq, canno in sequence_annotations:
        cseqannotations = [(cseq, canno)]
        res = db.get_enrichment_score(annotations=annotations, seqannotations=cseqannotations, term_info=term_info)
        cf = res[0]
        seq_scores[cseq] = {}
        seq_scores[cseq][term1] = cf.get(term1, 0)
        seq_scores[cseq][term2] = cf.get(term2, 0)

    term1vec = np.array([seq_scores[cseq][term1] for cseq in exp.feature_metadata.index.values])
    term2vec = np.array([seq_scores[cseq][term2] for cseq in exp.feature_metadata.index.values])

    cdata = exp.get_data(copy=True, sparse=False)
    if transform == 'log2data':
        cdata[cdata < 1] = 1
        cdata = np.log2(cdata)
    elif transform == 'binarydata':
        cdata = cdata > 0
    elif transform == 'rankdata':
        cdata = scipy.stats.rankdata(cdata, axis=0)

    samp_term1 = np.matmul(cdata, term1vec)
    samp_term2 = np.matmul(cdata, term2vec)

    exp.sample_metadata['_coord_term1'] = samp_term1
    exp.sample_metadata['_coord_term2'] = samp_term2

    plt.figure()
    labels = []
    for cval, cexp in exp.iterate(field):
        labels.append(cval)
        plt.plot(cexp.sample_metadata['_coord_term1'], cexp.sample_metadata['_coord_term2'], '.')
    plt.legend(labels)
    plt.xlabel(term1)
    plt.ylabel(term2)
    plt.title(transform)


def metadata_correlation(exp, value_field, alpha=0.1, ok_columns=None, bad_values=[], filter_na=True):
    '''Test correlation/enrichment between sample metadata columns and a given value_field which is a sample metadata column (e.g. _health_index following cu.health_index() ).
    Enrichment is performed on categorical metadata fields (comparing mann-whitney of value_field in the groups)
    Correlation is performed on numeric metadata fields (spearman of value field and the metadata fields)

    Parameters
    ----------
    exp: calour.Experiment
    value_field: str
        the name of the sample metadata field to compare to all the other fields (e.g. "_health_index")
    alpha: float, optional
        the FDR level (to correct for the multiple metadata fields tested)
    ok_columns: list of str or None, optional
        if not None, test only fields appeating in ok_columns instead of all the sample_metadata fields
    bad_values: list of str, optional
        values to not include in the testing (e.g. 'unknwon')
    filter_na: bool, optional
        True to remove samples with nan in their metadata field

    Returns
    -------
    fields
    stats
    q (corrected p-values)
    names
    '''
    amd = exp.sample_metadata.copy()
    names = []
    pvals = []
    fields = []
    stats = []
    num_skipped = 0
    if ok_columns is None:
        ok_columns = amd.columns
    for cfield in amd.columns:
        if ok_columns is not None:
            if cfield not in ok_columns:
                continue
        md = amd.copy()
        # get rid of bad field values
        for cignore in bad_values:
            md = md[md[cfield] != cignore]
        if filter_na:
            md = md[md[cfield].notna()]

        if len(md[cfield].unique()) == 1:
            logger.debug('field %s contains 1 value. skipping' % cfield)
            num_skipped += 1
            continue
        if len(md[cfield].unique()) >= 3:
            # print('>=3 values for field %s' % cfield)
            if not pd.to_numeric(md[cfield], errors='coerce').notnull().all():
                num_skipped += 1
                logger.debug('field %s is not numeric and contains >= 3 unique values. skipping' % cfield)
                continue
            if len(md) < 10:
                pass
                # continue
            cres = scipy.stats.spearmanr(md[value_field], md[cfield])
            names.append('COR: (%d) %s: %s' % (len(md), cfield, cres))
            ccres = cres[0]
        elif len(md[cfield].unique()) == 2:
            # print('2 values for field %s' % cfield)
            vals = sorted(md[cfield].unique())
            cv1 = md[md[cfield] == vals[0]]
            cv2 = md[md[cfield] == vals[1]]
            cres = scipy.stats.mannwhitneyu(cv1[value_field], cv2[value_field], alternative='two-sided')
            names.append('BIN: %s: %s (%d, %f), %s (%d, %f) %s' % (cfield, vals[0], len(cv1), np.median(cv1[value_field]), vals[1], len(cv2), np.median(cv2[value_field]), cres))
            ccres = np.median(cv1[value_field]) - np.median(cv2[value_field])
        else:
            logger.debug('no values for field %s' % cfield)
            num_skipped += 1
            continue
        pvals.append(cres[1])
        fields.append(cfield)
        stats.append(ccres)
    if num_skipped > 0:
        logger.info('skipped %d (out of %d) fields with inappropriate values' % (num_skipped, len(amd.columns)))
    if len(pvals) == 0:
        logger.error('no fields with matching values detected')
        return [], [], [], []
    reject, q, aaa, bbb = multipletests(pvals, alpha=alpha, method='fdr_bh')
    # reject, q = statsmodels.stats.multitest.fdrcorrection(pvals,alpha=alpha,method='indep')
    for idx, cname in enumerate(names):
        if reject[idx]:
            print(cname)
    return fields, np.array(stats), np.array(q), names


# for the group_dependence method
def variance_stat(data, labels):
    cstat = 0.1
    for clab in list(set(labels)):
        cdat = data[:, labels == clab]
        ccstat = np.var(cdat, axis=1)
        cstat += ccstat
    return np.var(data, axis=1) / cstat


def group_dependece(exp: ca.Experiment, field, method='variance', transform=None,
                    numperm=1000, alpha=0.1, fdr_method='dsfdr', random_seed=None):
    '''Find features with non-random group distribution based on within-group variance

    The permutation based p-values and multiple hypothesis correction is implemented.

    Parameters
    ----------
    field: str
        The field to test by. Values are converted to numeric.
    method : str or function
        the method to use for the statistic. options:

        * 'variance': sum of within-group variances
        * callable: the callable to calculate the statistic (its input are
          sample-by-feature numeric numpy.array and 1D numeric
          numpy.array of sample metadata; output is a numpy.array of float)
    transform : str or None
        transformation to apply to the data before caluculating the statistic.

        * 'rankdata' : rank transfrom each OTU reads
        * 'log2data' : calculate log2 for each OTU using minimal cutoff of 2
        * 'normdata' : normalize the data to constant sum per samples
        * 'binarydata' : convert to binary absence/presence
    alpha : float
        the desired FDR control level (type I error rate)
    numperm : int
        number of permutations to perform
    fdr_method : str
        method to compute FDR. Allowed methods include:

        * 'dsfdr': discrete FDR
        * 'bhfdr': Benjamini-Hochberg FDR method
        * 'byfdr' : Benjamini-Yekutielli FDR method
        * 'filterBH' : Benjamini-Hochberg FDR method with filtering
    random_seed : int, np.radnom.Generator instance or None, optional, default=None
        set the random number generator seed for the random permutations
        If int, random_seed is the seed used by the random number generator;
        If Generator instance, random_seed is set to the random number generator;
        If None, then fresh, unpredictable entropy will be pulled from the OS

    Returns
    -------
    Experiment
        The experiment with only correlated features, sorted according to correlation coefficient.

        * '{}' : the non-adjusted p-values for each feature
        * '{}' : the FDR-adjusted q-values for each feature
        * '{}' : the statistics (correlation coefficient if
          the `method` is 'spearman' or 'pearson'). If it
          is larger than zero for a given feature, it indicates this
          feature is positively correlated with the sample metadata;
          otherwise, negatively correlated.
        * '{}' : in which of the 2 sample groups this given feature is increased.
    '''
    if field not in exp.sample_metadata.columns:
        raise ValueError('Field %s not in sample_metadata. Possible fields are: %s' % (field, exp.sample_metadata.columns))

    cexp = exp.filter_sum_abundance(0, strict=True)

    data = cexp.get_data(copy=True, sparse=False).transpose()

    labels = np.zeros([len(exp.sample_metadata)])
    for idx, clabel in enumerate(exp.sample_metadata[field].unique()):
        labels[exp.sample_metadata[field] == clabel] = idx

    # remove the nans
    nanpos = np.where(np.isnan(labels))[0]
    if len(nanpos) > 0:
        logger.warning('NaN values encountered in labels for correlation. Ignoring these samples')
        labels = np.delete(labels, nanpos)
        data = np.delete(data, nanpos, axis=1)
    if method == 'variance':
        method = variance_stat
    else:
        raise ValueError('method %s not supported' % method)

    # find the significant features
    keep, odif, pvals, qvals = ca.dsfdr.dsfdr(data, labels, method=method, transform_type=transform, alpha=alpha, numperm=numperm, fdr_method=fdr_method, random_seed=random_seed)
    logger.info('Positive dependent features : %d. Negative dependent features : %d. total %d'
                % (np.sum(odif[keep] > 0), np.sum(odif[keep] < 0), np.sum(keep)))
    newexp = ca.analysis._new_experiment_from_pvals(cexp, exp, keep, odif, pvals, qvals)
    return newexp


def plot_violin_category(exp, group_field, value_field, xlabel_params={'rotation': 90}, colors=None, show_stats=False):
    '''Draw a violin plot for metadata distribution (numeric) between different metadata categories (categorical)
    The plot shows a violin plot and the points (with random x jitter)

    Parameters
    ----------
    exp: calour.Experiment
    group_field: str
        the categorical field to use for the x axis groups
    value_field: str
        the numeric field for the values within each group
    xlabel_params: dict, optional
        the additional parameters for the xtick lables

    Returns
    -------
    labels: list of str - the ordered categoru names (i.e. group_filed)
    vals: list of list of float
        the values for each category
    f: matplotlib.figure
        the figure
    '''
    vals = []
    labels = []
    for clab, cexp in exp.iterate(group_field):
        labels.append(clab)
        vals.append(np.array(cexp.sample_metadata[value_field].values))

    if colors is not None:
        if isinstance(colors, str):
            colors = [colors] * len(labels)

    si = np.argsort(labels)
    vals = [vals[x] for x in si]
    labels = np.array(labels)[si]
    f = plt.figure()
    plt.violinplot(vals, showmedians=True)
    for idx, cvals in enumerate(list(vals)):
        offset = np.random.randn(len(cvals)) * 0.05
        if colors is None:
            plt.plot(offset + idx + 1, cvals, '.')
        else:
            plt.plot(offset + idx + 1, cvals, '.', c=colors[idx])
    ax = plt.gca()
    ax.set_xticks(np.arange(len(labels)) + 1)
    ax.set_xticklabels(labels, **xlabel_params)
    plt.xlabel(group_field)
    plt.ylabel(value_field)

    if show_stats:
        if len(labels) == 2:
            plt.title('Mann-Whitney: %s' % scipy.stats.mannwhitneyu(vals[0], vals[1])[1])
    return labels, vals, f


def exp_from_fasta(file1, file2, group_names=['s1', 's2']):
    '''Prepare a calour.AmpliconExperiment from two fastafiles and run diff_abundance on it.

    Parameters
    ----------
    files,file2: str
        Name of the fasta files to load for group1 and group2
    group_names: list of (str, str), optional
        name of the two fasta file groups

    Returns
    -------
    ca.AmpliconExperiment
    '''
    g1 = []
    for chead, cseq in ca.io._iter_fasta(file1):
        g1.append(cseq)
    logger.info('Loaded %d sequences from file %s' % (len(g1), file1))
    g2 = []
    for chead, cseq in ca.io._iter_fasta(file2):
        g2.append(cseq)
    logger.info('Loaded %d sequences from file %s' % (len(g2), file2))

    smd = pd.DataFrame(group_names, columns=['_sample_id'])
    smd = smd.set_index('_sample_id', drop=False)

    st = np.ones(len(g1) + len(g2))
    st[:len(g1)] = 0
    fmd = pd.DataFrame(st, columns=['type'])
    fmd['_feature_id'] = g1 + g2
    fmd = fmd.set_index('_feature_id', drop=False)

    data = np.zeros([len(smd), len(fmd)])
    data[0, :len(g1)] = 1
    data[1, len(g1):] = 1

    exp = ca.AmpliconExperiment(data=data, feature_metadata=fmd, sample_metadata=smd, sparse=False)
    dd = exp.diff_abundance('_sample_id', group_names[0], group_names[1], alpha=1)
    return dd

from collections import defaultdict

import calour as ca


def hash_sequences(exp, short_len=100):
	'''hash all the sequences in a fasta file

	Parameters
	----------
	short_len: int, optional
		the minimal sequence length in the database

	Returns
	-------
	seq_hash: dict of {seq: seqid}
	seq_lens : list of int
		all the sequence lengths in the fasta file (so we can hash all the lengths in the queries)
	short_hash: dict of {short_seq: seq_hash dict}
	'''
	total_seqs = 0
	num_too_short = 0
	seq_hash = {}
	seq_lens = set()
	all_ids = set()
	short_hash = defaultdict(dict)

	# get the sequence database index from name
	debug(2, 'Scanning dbbact sequences')
	for cseq, cid in iter_new_seqs(con):
		total_seqs += 1
		clen = len(cseq)
		if clen < short_len:
			num_too_short += 1
			continue
		if check_exists:
			err, existsFlag = db_translate.test_whole_seq_id_exists(con, cur, dbidVal=dbidVal, dbbactidVal=cid)
			if existsFlag:
				continue
		all_ids.add(cid)
		short_seq = cseq[:short_len]
		short_hash[short_seq][cseq] = cid
		if clen not in seq_lens:
			seq_lens.add(clen)
		seq_hash[cseq] = cid

	debug(2, 'processed %d dbbact sequences. found %d new sequences' % (total_seqs, len(seq_hash)))
	debug(2, 'lens: %s' % seq_lens)
	debug(2, 'num too short: %d' % num_too_short)
	return all_ids, seq_hash, seq_lens, short_hash


def seqs_to_silva(exp, silva):
	'''Add silva IDs to each feature in the experiment

	Paramaters
	----------
	exp: calour.Experiment
		the experiment to add the silva ids to (must have sequences as feature_ids)
	silva: str
		name of the silva fasta file

	Returns
	-------
	ca.Experiment
		with a list of comma separated silva ids for as feature ids
	'''

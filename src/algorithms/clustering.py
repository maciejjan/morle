from datastruct.graph import *
from utils.files import *
from utils.printer import *
import random

# converts a list of clusters to a dictionary "element => cluster label"
def clusters_list_to_dict(cl_list):
	cl_dict = {}
	for i, cluster in enumerate(cl_list, 1):
		for elem in cluster:
			if not cl_dict.has_key(elem):
				cl_dict[elem] = []
			cl_dict[elem].append(i)
	return cl_dict

# converts a dictionary "element => cluster label" to a list of clusters
def clusters_dict_to_list(cl_dict):
	inverted_dict = {}
	for elem, labels in cl_dict.iteritems():
		if isinstance(labels, list):
			for label in labels:
				if not inverted_dict.has_key(label):
					inverted_dict[label] = []
				inverted_dict[label].append(elem)
		else:
			if not inverted_dict.has_key(labels):
				inverted_dict[labels] = []
			inverted_dict[labels].append(elem)
	return inverted_dict.values()

### CLUSTERING FUNCTIONS ###

def clink(graph):
	# get all edges from the graph and sort them by weight
	edges = []
	for v in graph.vertices():
		for e in graph.edges(v):
			edges.append((v, e.vertex, e.weight))
	edges.sort(reverse = True, key = lambda x: x[2])
	# initially: each vertex is a separate cluster 
	v_clust, clusters = {}, {}
	for i, v in enumerate(graph.vertices()):
		v_clust[v] = i
		clusters[i] = [v]
	# merge the most similar clusters
#	pp = progress_printer(len(edges))
	for v1, v2, e in edges:
		merge = True
		for u1 in clusters[v_clust[v1]]:
			for u2 in clusters[v_clust[v2]]:
				if not graph.has_edge(u1, u2):
					merge = False
					break
		if merge:
			old_label, new_label = v_clust[v2], v_clust[v1]
			clusters[new_label].extend(clusters[old_label])
			for u2 in clusters[old_label]:
				v_clust[u2] = new_label
			del clusters[old_label]
#		pp.next()
	# return the clusters
	return clusters.values()

def incl_excl(lst):
	def eval_term(term):
		result = 1
		for i in term:
			result *= lst[i]
		return result

	result = 0.0
	terms = [(i, ) for i in range(len(lst))]
	sign = 1
	for i in range(min(len(lst), 5)):
		result += sign * sum([eval_term(t) for t in terms])
		new_terms = []
		for t in terms:
			for j in range(len(lst)):
				if j > t[-1]:
					new_terms.append(t + (j, ))
		terms = new_terms
		sign *= -1
	return result

#CW_COMBINING_FUN = lambda l: sum(l)*len(l)
#CW_COMBINING_FUN = incl_excl
CW_COMBINING_FUN = sum

def chinwhisp(graph, threshold=0, max_iterations=None):
	v_clust = {}		# cluster ID for each vertex
	# initially: each vertex in a separate cluster
	vertices = [v for v in graph.vertices()]
	for i, v in enumerate(vertices):
		v_clust[v] = i
	# clustering
	changes = True
	num_iterations = 0
	while changes:
		if max_iterations is not None and num_iterations > max_iterations:
			break
		chages = False
		random.shuffle(vertices)
		pp = progress_printer(len(vertices))
		for v in vertices:
			labels = {}
			for edge in graph.edges(v):
				label = v_clust[edge.vertex]
				if not labels.has_key(label):
					labels[label] = []
#				labels[label] += edge.weight
#				labels[label] = (labels[label][0]+edge.weight, labels[label][1]+1)
				labels[label].append(edge.weight)
			pp.next()
			if not labels:
				continue
			winning_label, winning_score = max([(label, CW_COMBINING_FUN(scores))\
				for label, scores in labels.iteritems()], \
				key = lambda x: x[1])
#			winning_label, winning_score = max([(label, score*num)\
#				for label, (score, num) in labels.iteritems()], \
#				key = lambda x: x[1])
			if winning_score < threshold:
				continue
			if v_clust[v] != winning_label:
				changes = True
			v_clust[v] = winning_label
		num_iterations += 1
	# retrieve clusters
	return clusters_dict_to_list(v_clust)

# probabilistic chinese whispers
def chinwhisp_prob(graph, threshold=0, max_iterations=None):
	v_clust = {}		# cluster ID for each vertex
	# initially: each vertex in a separate cluster
	vertices = [v for v in graph.vertices()]
	for i, v in enumerate(vertices):
		v_clust[v] = i
	# clustering
	changes = True
	num_iterations = 0
	while changes:
		if max_iterations is not None and num_iterations > max_iterations:
			break
		chages = False
		random.shuffle(vertices)
		pp = progress_printer(len(vertices))
		for v in vertices:
			labels = {}
			for edge in graph.edges(v):
				label = v_clust[edge.vertex]
				if not labels.has_key(label):
					labels[label] = []
				labels[label].append(edge.weight)
			pp.next()
			if not labels.has_key(v_clust[v]):
				labels[v_clust[v]] = [max(0.0, 1.0 - sum([CW_COMBINING_FUN(scores) \
					for scores in labels.values()]))]
			winning_label, winning_score = max([(label, CW_COMBINING_FUN(scores))\
				for label, scores in labels.iteritems()], \
				key = lambda x: x[1])
			if winning_score < threshold:
				continue
			if v_clust[v] != winning_label:
				changes = True
			v_clust[v] = winning_label
		num_iterations += 1
	# retrieve clusters
	return clusters_dict_to_list(v_clust)

### CLUSTERING EVALUATION MEASURES ###

def bcubed(words, clusters, classes, log_filename):
	w_cluster = clusters_list_to_dict(clusters)
	w_class = clusters_list_to_dict(classes)
	log_fp = open_to_write(log_filename)

	def get_words_in_same_cluster(word, cl, w_cl):
		result = []
		for num in w_cl[word]:
			result += cl[num-1]
		return list(set(result))
	
	def count_common_clusters(word_1, word_2, cl, w_cl):
		if not w_cl.has_key(word_1) or not w_cl.has_key(word_2):
			return 0
		return len(set(w_cl[word_1]) & set(w_cl[word_2]))
	
	def evaluate_word(word):
		precision, recall = 0.0, 0.0
		count = 0
		same_cluster = get_words_in_same_cluster(word, clusters, w_cluster)
		for word_2 in same_cluster:
			if w_class.has_key(word_2):
				common_clusters = count_common_clusters(word, word_2, clusters, w_cluster)
				common_classes = count_common_clusters(word, word_2, classes, w_class)
				precision += float(min(common_clusters, common_classes)) / common_clusters
#				log_fp.write('%s %s %d %d' % (word, word_2, common_clusters, common_classes))
				count += 1
		precision /= count
		count = 0
		same_class = get_words_in_same_cluster(word, classes, w_class)
		for word_2 in same_class:
			if w_cluster.has_key(word_2):
				common_clusters = count_common_clusters(word, word_2, clusters, w_cluster)
				common_classes = count_common_clusters(word, word_2, classes, w_class)
				recall += float(min(common_clusters, common_classes)) / common_classes
				count += 1
		recall /= count
		# write a record in the log file
#		log_fp.write('%s %0.2f %0.2f\n' % (word, 100*precision, 100*recall))
		same_cluster = set(same_cluster)
		same_class = set(same_class)
		tp = same_cluster & same_class
		fp = [w for w in (same_cluster - same_class) if w_class.has_key(w)]
		fn = [w for w in (same_class - same_cluster) if w_cluster.has_key(w)]
		log_fp.write('%s %d %d %d\n' % (word, len(tp), len(fp), len(fn)))
		log_fp.write('TP: %s\n' % ', '.join(sorted(list(tp))))
		log_fp.write('FP: %s\n' % ', '.join(sorted(list(fp))))
		log_fp.write('FN: %s\n' % ', '.join(sorted(list(fn))))
#		log_fp.write('FP: %s\n' % ', '.join(sorted([\
#			w for w in (same_cluster - same_class) if w_class.has_key(w)])))
#		log_fp.write('FN: %s\n' % ', '.join(sorted([\
#			w for w in (same_class - same_cluster) if w_cluster.has_key(w)])))
#		log_fp.write('TP: %s\n' % ', '.join(sorted(list(same_cluster & same_class))))
#		log_fp.write('FP: %s\n' % ', '.join(sorted([\
#			w for w in (same_cluster - same_class) if w_class.has_key(w)])))
#		log_fp.write('FN: %s\n' % ', '.join(sorted([\
#			w for w in (same_class - same_cluster) if w_cluster.has_key(w)])))
		log_fp.write('\n')
#		return len(tp), len(fp), len(fn)
#		return precision, recall
		n_tp, n_fp, n_fn = len(tp), len(fp), len(fn)
		return float(n_tp) / (n_tp + n_fp), float(n_tp) / (n_tp + n_fn)
	
	precision, recall, num_words_evaluated = 0.0, 0.0, 0
	tp, fp, fn = 0, 0, 0
	pp = progress_printer(len(words))
	for word in words:
		if w_class.has_key(word):
			w_precision, w_recall = evaluate_word(word)
			precision += w_precision
			recall += w_recall
#			w_tp, w_fp, w_fn = evaluate_word(word)
#			tp += w_tp
#			fp += w_fp
#			fn += w_fn
			num_words_evaluated += 1
		pp.next()
#	precision = float(tp) / (tp + fp)
#	recall = float(tp) / (tp + fn)
	precision /= num_words_evaluated
	recall /= num_words_evaluated
	f_score = 2*precision*recall / (precision + recall)
	log_fp.close()
	return precision, recall, f_score


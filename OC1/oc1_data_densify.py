#! usr/bin/python

# this reads in a sparse dataset and produces a dense dataset format file
# This is to assist with testing and running sparse format data
# [index value, index value, ...]
def main():
	# this is the main method
	bigname = "main"
	# read in a sparse format file
	infile = "absolute path to file"
	outfile = "absolute path to outpout file"

	# list of form [[(i:v),...],...]
	data_tups_lis = [] # each entry is a list of tuples of feature_index : value 
	lis_unique_feats = []
	feat_dict = {} # of form {feat_index:frequency}
	num_line = 0
	with open(infile, "r") as file:
		for line in file:
			num_line+=1
			temp_lis = [] # list of tuples
			l2 = line.split() # split the list into individual values
			for x in range(0, len(l2), 2): # for every second value starting at first (goes to every index)
				temp_lis.append((l2[x], l2[x+1])) # append the tuple of the index and its value to the list
				# if this is a never before seen feature, note it
				#if l2[x] not in lis_unique_feats:
				#	lis_unique_feats.append(l2[x])
				if l2[x] not in feat_dict:
					feat_dict[l2[x]] = 1
				else:
					feat_dict[l2[x]] = feat_dict[l2[x]] + 1
			data_tups_lis.append(temp_lis)
	num_less_than_freq_thresh = 0
	freq_thresh = 5
	num_feats = 0

	## Handling constraint to enable faster testing
	## THIS IS THE VALUE TO TWEAK FOR DF
	df_percentage = 0.0 # change to some decimal to exclude a percentage of values
	dfd_out = 0
	lis_feats_to_keep = [] # a list of feature values to keep based on the thresholds above
	for k,v in feat_dict.items():
		num_feats+=1 # update number of features
		if (float(v)/num_line) > df_percentage:
			dfd_out+=1
			lis_feats_to_keep.append(k)


	# map/keep the values kept to integer values.
	final_lis = []
	new_index_upper_bound = len(lis_feats_to_keep) # this is the max index value of the new list
	lis_feats_to_keep.sort()
	# this maps the remaining feature_indices to new indices
	dict_of_keeps = {} # of form feat:new_index
	ind = 0
	for feat in lis_feats_to_keep:
		dict_of_keeps[feat] = ind
		ind+=1

	# create the new mappings
	new_mapped_vals_lis = []
	for line_lis in data_tups_lis:
		temp_v_lis = [0] * new_index_upper_bound # create an empty list of appropriate size
		for tup in line_lis:
			if tup[0] in dict_of_keeps:
				temp_v_lis[int(dict_of_keeps[tup[0]])] = tup[1]
		new_mapped_vals_lis.append(temp_v_lis)


	## output the file
	with open(outfile, "w") as ofile:
		for line in new_mapped_vals_lis:
			for val in line:
				ofile.write("%s " % val)
			ofile.write("\n")

if __name__ == "__main__":
	main()

"""
Notes:
1) For very large (sparse) data feature constraint will need to be done for OC1
anything with more than about 1000 instances and/or 60,000 features will
cause OC1 to buffer overflow on a strcpy_chk (after around 50 splits)
2) The buffer overflow is handled by the OS, not OC1. Thus obtaining core dumps
will depend upon the OS. In Linux the dumps are sometimes swallowed for some reason.
On Windows they are not.
3) Printing OC1 with verbose (or extra verbose) output, sometimes causes a hang for
about 5 seconds before printing final results, but it will print.
"""

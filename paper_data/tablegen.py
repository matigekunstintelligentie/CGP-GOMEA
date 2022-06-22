import pandas as pd
import numpy as np
import scipy.stats
import warnings
#warnings.filterwarnings("ignore")


def signed_rank(x, y):
    return scipy.stats.wilcoxon(x, y, zero_method="pratt")[1]

def get_significance_by_idx(mat, comparitive_index, experiment_names=None, alpha=0.05):
	number_experiments = mat.shape[0]

	corrected_alpha = alpha/(number_experiments-1)

	l = [x for x in range(number_experiments)]
	l.remove(comparitive_index)

	significant = 1
	for i in l:
		significant = int(signed_rank(mat[comparitive_index], mat[i])<corrected_alpha)
		if significant==0:
			return 0

	return significant
	
def get_significance_between_idxs(mat, comparitive_index0, comparitive_index1, experiment_names=None, alpha=0.05):



	significant = 1

	significant = int(signed_rank(mat[comparitive_index0], mat[comparitive_index1])<alpha)
	if significant==0:
		return 0

	return significant

def get_significance(mat, experiment_names=None, alpha=0.05):
	number_experiments = mat.shape[0]

	corrected_alpha = alpha/(number_experiments-1)

	stri = [ "distributions are not significantly different", "distributions are significantly different"]

	stri_win = lambda x, y, z :"{}>{}".format(y,z) if x==0 else "{}>{}".format(z,y)


	for i in range(number_experiments):
		for j in range(i+1, number_experiments):
			medi = np.median(mat[i])
			medj = np.median(mat[j])
			if experiment_names is not None:
				# ve care node re-use want to be high
				print(medi, medj)
				print(experiment_names[i], experiment_names[j], signed_rank(mat[i], mat[j]), stri[int(signed_rank(mat[i], mat[j])<corrected_alpha)], stri_win(medi>medj, experiment_names[i], experiment_names[j]))
			else:

				print(i, j, signed_rank(mat[i], mat[j]), stri[int(signed_rank(mat[i], mat[j])<corrected_alpha)], stri_win(medi>medj, i, j))


header = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N",
"O","P","Q","R","S","T","U","V","W","X","Y","Z","AA","AB","AC","AD",
"AE","AF","AG","AH","AI","AJ","AK","AL","AM","AN",
"AO","AP"]

#header = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N",
#"O","P","Q","R","S","T","U","V","W","X","Y","Z","AA","AB","AC","AD",
#"AE","AF","AG","AH","AI","AJ","AK","AL","AM","AN",
#"AO"]

names = {"111116550.5-10":"GP-GOMEA", "011116441.0-10":"CGP-GOMEA", "011116441.0610":"CGP-GOMEA trunc", 
"01118881.0-10":"CGP-GOMEA 8x8", "0111110101.0-10":"CGP-GOMEA 10x1", "011116411.0-10":"CGP-GOMEA LB1","0111164641.0-10":"CGP-GOMEA 64x1", "000016441.0-10":"CGP-classic"}

#names = {"111116550.5-10":"GP-GOMEA", "011116441.0-11":"CGP-GOMEA 16x4", "011116441.0-10": "CGP-GOMEA 16x4 nomix", "01118881.0-11":"CGP-GOMEA 8x8", "011116441.0611": "trunc", "0111110101.0-11":"CGP-GOMEA 10x1", "011116411.0-11":"CGP-GOMEA LB1", "0111164641.0-11":"CGP-#GOMEA 64x1", "000016441.0-11":"CGP-CLassic"}

#names = {"000016441.0-100,1,2,8,":"CGP-classic","011116441.0-100,1,2,8,":"CGP-GOMEA 16x4", "111116550.5-100,1,2,8,": "GP-GOMEA", "011116411.0-100,1,2,8,":"CGP-GOMEA 16x4 LB1", "01118881.0-100,1,2,8,":"CGP-GOMEA 8x8"}

df = pd.read_csv("noerc_0_10.csv",sep="\t",header=None, names=header)

experiments = {}

for experiment in ["./data/datasets/boston.csv","./data/datasets/yacht.csv","./data/datasets/tower.csv"]:
#for experiment in ["./data/datasets/fake_data.csv"]:
	experiments[experiment] = {}
	for index, row in df.iterrows():
		if row["N"]==experiment:
		#if row["M"]==experiment:
			experiment_key = ""
			for letter in ["C","D","E","F","G","H","I","J","K","L"]:
			#for letter in ["B","C","D","E","F","G","H","I","J","K","L"]:
				experiment_key += str(row[letter])
			if experiment_key not in experiments[experiment].keys():
				experiments[experiment][experiment_key] = [[],[],[],[]]
				
			
			experiments[experiment][experiment_key][0].append(row["R"])
			experiments[experiment][experiment_key][1].append(row["V"])
			experiments[experiment][experiment_key][2].append(row["AG"])
			experiments[experiment][experiment_key][3].append(row["AI"])
			
			#experiments[experiment][experiment_key][0].append(row["Q"])
			#experiments[experiment][experiment_key][1].append(row["U"])
			#experiments[experiment][experiment_key][2].append(row["AF"])
			#experiments[experiment][experiment_key][3].append(row["AH"])

for experiment in ["./data/datasets/boston.csv","./data/datasets/yacht.csv","./data/datasets/tower.csv"]:
#for experiment in ["./data/datasets/fake_data.csv"]:
    print(experiment)
    print("#"*50)
    max_train = []
    train_mat = []
    max_test = []
    test_mat = []
    max_length = []
    length_mat = []
    max_reuse = []
    reuse_mat = []
    exp_names = [] 
	
    for key in experiments[experiment].keys():

        max_train.append(np.median(experiments[experiment][key][1]))
        train_mat.append(experiments[experiment][key][1])
        max_test.append(np.median(experiments[experiment][key][0]))
        test_mat.append(experiments[experiment][key][0])
        max_length.append(np.mean(experiments[experiment][key][2]))
        length_mat.append(experiments[experiment][key][2])
        
        max_reuse.append(np.mean(experiments[experiment][key][3]))
        reuse_mat.append(experiments[experiment][key][3])
        
        exp_names.append(names[key])
        
        print(exp_names[-1])

        
        stri = "{:.3}".format(np.median(experiments[experiment][key][1])) + "$\pm$" + "{:.2e}".format(np.std(experiments[experiment][key][1])) + "&" + "{:.3}".format(np.median(experiments[experiment][key][0])) + "$\pm$" + "{:.2e}".format(np.std(experiments[experiment][key][0])) + "&" + "{:.3}".format(np.mean(experiments[experiment][key][2])) + "$\pm$" + "{:.2e}".format(np.std(experiments[experiment][key][2])) + "&" + "{:.3}".format(np.mean(experiments[experiment][key][3])) + "$\pm$" + "{:.2e}".format(np.std(experiments[experiment][key][3]))
        
        stri = "{:.3}".format(np.median(experiments[experiment][key][1])) + "$\pm$" + "{:.2e}".format(np.std(experiments[experiment][key][1])) + "&" + "{:.3}".format(np.median(experiments[experiment][key][0])) + "$\pm$" + "{:.2e}".format(np.std(experiments[experiment][key][0])) + "{:.3}".format(np.mean(experiments[experiment][key][3])) + "$\pm$" + "{:.2e}".format(np.std(experiments[experiment][key][3]))
        
        stri = stri.replace("e+01","$\mathrm{e}^{1}$")
        stri = stri.replace("e+02","$\mathrm{e}^{2}$")
        stri = stri.replace("e+03","$\mathrm{e}^{3}$")
        stri = stri.replace("e+04","$\mathrm{e}^{4}$")
        stri = stri.replace("e-00","")
        stri = stri.replace("e+00","")
        stri = stri.replace("e-01","$\mathrm{e}^{-1}$")
        stri = stri.replace("e-02","$\mathrm{e}^{-2}$")
        stri = stri.replace("e-03","$\mathrm{e}^{-3}$")
        stri = stri.replace("e-04","$\mathrm{e}^{-4}$")
        print(stri)

    print("Max train: ", exp_names[np.argmax(max_train)], get_significance_by_idx(np.array(train_mat), np.argmax(max_train), exp_names))
    print("Max test: ", exp_names[np.argmax(max_test)], get_significance_by_idx(np.array(test_mat), np.argmax(max_test), exp_names))
    print("Min Length: ", exp_names[np.argmin(max_length)], get_significance_by_idx(np.array(length_mat), np.argmin(max_length), exp_names))
    print("Max Re-use: ", max_reuse[np.argmax(max_reuse)], exp_names[np.argmax(max_reuse)], get_significance_by_idx(np.array(reuse_mat), np.argmax(max_reuse), exp_names))
    print("GP-GOMEA vs 8x8 train", "GP-GOMEA", "8x8", get_significance_between_idxs(np.array(train_mat),0,3),np.median(np.array(train_mat[0])), np.median(np.array(train_mat[3])))
    print("GP-GOMEA vs 8x8 re-use", "GP-GOMEA", "8x8", get_significance_between_idxs(np.array(reuse_mat),0,3),np.average(np.array(reuse_mat[0])), np.average(np.array(reuse_mat[3])))
    print(reuse_mat[0], reuse_mat[3])
    
	

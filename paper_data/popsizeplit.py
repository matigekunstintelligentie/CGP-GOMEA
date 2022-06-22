import matplotlib.pyplot as plt
import matplotlib
#plt.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 16

import numpy as np

train_nmse = [[0.777784,0.7816115,0.7794105,0.7800035,0.791273,0.7934855,0.7895935,0.799716,0.7946155,0.803439, 0.7861115],[0.7662455,0.784572,0.781359,0.784691,0.79859,0.7850495,0.7713645,0.7706355,0.770127,0.762664,0.7606375],[0.7622265,0.762164,0.7627015,0.776461,0.777622,0.7871495,0.780529,0.7876165,0.7840665,0.787789,0.7689725]]
percentile_10 = [[0.7534237,0.7506365,0.7591077,0.7573498,0.7659483,0.7775275,0.7806897,0.7806613,0.7822701,0.7881023,0.7732277],[0.7366305,0.7474207,0.746976,0.766014,0.7641768,0.7679535,0.7529772,0.7464316,0.7465723,0.7458835,0.7401123],[0.7402957,0.7444234,0.750393,0.7484162,0.7541434,0.7563537,0.7665762,0.7795416,0.7659347,0.7630992,0.7491573]]
percentile_90 = [[0.8180026,0.8045971,0.8024332,0.8003685,0.8222448,0.8282396,0.816146,0.8216223,0.8123232,0.8194092,0.8081293],[0.7931643,0.8039825,0.7954029,0.8091306,0.830417,0.8163895,0.8044259,0.7911535,0.7919031,0.7943127,0.7851585],[0.7968378,0.7918222,0.7843284,0.7993767,0.7933585,0.8028918,0.7985683,0.8112757,0.8019792,0.8034674,0.7978806]]









plt.figure()
plt.title(r"Population size vs Train $R^2$ - Boston housing (5000 seconds, tree height=3)")
plt.ylabel(r"Train $R^2$")
plt.xlabel("Population size")
plt.xticks([500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000])
plt.xscale('log')
legends = ["GP-GOMEA", "CGP-GOMEA 16x4", "CGP-Classic"]
offset = -50
cols = ['r', 'g', 'b']
for idx, el in enumerate(train_nmse):
    plt.fill_between([500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000], percentile_10[idx], percentile_90[idx],color=cols[idx], alpha=0.1)
    plt.plot([500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000], el, "x--",c=cols[idx],label=legends[idx])

    
    
ax = plt.gca()
ax.set_xticks((500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000))
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
plt.legend()
plt.savefig("plots/{}.png".format("trainnmse_popsize_extended"), dpi=300)
plt.show()






# plt.figure()
# plt.title("Population size vs TrainNMSE - Boston housing")
# plt.ylabel("Train NMSE")
# plt.xlabel("Population size")
# plt.xticks([1000, 2000, 4000, 8000, 16000])
# plt.xscale('log')
# legends = ["GP-GOMEA", "CGP-GOMEA 16x4", "CGP-GOMEA 16x4 LB1"]
# offset = -50
# cols = ['r', 'g', 'b']
# for idx, el in enumerate(train_nmse):
#     popsizes = [1000, 2000,4000,8000, 16000]
#     for popsi, y in enumerate(el):
#         x = np.array([popsizes[popsi]+offset*((popsi+1)**2)]*10)
#         y = np.array(y)
#         plt.scatter(x, y, c=cols[idx], s=1)
#     el = [np.median(x) for x in el]
#     plt.plot([1000, 2000,4000,8000, 16000], el, "x--",c=cols[idx],label=legends[idx])
#     offset += 50
    
    
# ax = plt.gca()
# ax.set_xticks((1000, 2000,4000,8000, 16000))
# ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# plt.legend()
# plt.savefig("plots/{}.png".format("trainnmse_popsize"))
# plt.show()

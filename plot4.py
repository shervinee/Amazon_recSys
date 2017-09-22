import numpy as np
import matplotlib.pyplot as plt

#used for plotting
plt.style.use('ggplot')

models = ('SVD++','SVD','KNN','KNN+Baseline')
Y = np.arange(len(models))
#The gathered data from the results on results_thirdPlot.txt
RMSE = [1.1029,1.1584,1.1710,1.0902]

#for building the horizontol bar
plt.barh(Y, RMSE, align='center', color ='#543f32')
plt.yticks(Y, models)
plt.ylabel('Approaches')
plt.xlabel('RMSE for sampled Amazon Instant Video Data')

#used to give RMSE values for each bar
# for i, v in enumerate(RMSE):
#     plt.text(v + 3, i + .25, str(v), color='black', fontweight='bold')
axes = plt.gca()
axes.set_xlim([0.5,1.5])

plt.savefig('Video_sampled_data.png')

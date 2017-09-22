import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
models = ('SVD++','SVD','KNN','KNN+Baseline')
Y = np.arange(len(models))
RMSE = [1.0686,1.1112,1.1742,1.0535]

plt.barh(Y, RMSE, align='center', color ='#543f32')
plt.yticks(Y, models)
plt.ylabel('Approaches')
plt.xlabel('RMSE for Musical Instrument Data')
axes = plt.gca()
axes.set_xlim([0.5,1.5])

plt.savefig('Music_full_data.png')

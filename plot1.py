import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
models = ('SVD++','SVD','KNN','KNN+Baseline')
Y = np.arange(len(models))
RMSE = [1.0683,1.1117,1.1741,1.0600]

plt.barh(Y, RMSE, align='center', color ='#543f32')
plt.yticks(Y, models)
plt.ylabel('Approaches')
plt.xlabel('RMSE for Amazon Instant Video Data')
axes = plt.gca()
axes.set_xlim([0.5,1.5])

plt.savefig('Video_full_data.png')

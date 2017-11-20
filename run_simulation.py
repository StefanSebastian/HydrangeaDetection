import image_preprocess
import autoencoder_convolutional
import predict
import compare

print('Starting image processing')
image_preprocess.process()
print('Image processing finished')

print('Started training model')
autoencoder_convolutional.train_autoencoder()
print('Model training finished')

print('Started unsupervised learning')
predict.clusterization()
print('Finished unsupervised learning')

print('Comparing results')
compare.compare_results()
print('Simulation done')
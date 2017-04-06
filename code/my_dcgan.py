import scipy
from scipy import misc
import numpy as np
import math
import argparse
import glob
import os 

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Convolution1D, MaxPooling1D
from keras.optimizers import SGD

import matplotlib.pyplot as plt

def load_data(in_file):
    input_f = open(in_file, "r")
    matrix = []
    for line in input_f:
        channels = [] #get channels
        for l in line.split("|"):
            samples = l.split(";")
            channels.append([float(i) for i in samples])
        matrix.append(channels)
        del channels
    input_f.close()
    return matrix

def load_labels(in_file):
    input_f = open(in_file, "r")
    labels = []
    for line in input_f:
        if ";" in line:
            labels.append(line.replace("\n","").split(";"))
        else:
            labels.append(line.replace("\n",""))
    input_f.close()
    return labels 

def load_data_train():    
    #training - compounds
    X_train = load_data("data/data_train_2.txt")
    y_train = load_labels("data/labels_train_2.txt")
    #X_train = load_data("/home/ashadrin/Videos/HSE/data/data_train.txt")
    #y_train = load_labels("/home/ashadrin/Videos/HSE/data/labels_train.txt")
    print ("initial train data: ", np.array(X_train).shape)
        
    #testing - mixtures
    X_test = load_data("data/data_test.txt")
    y_test = load_labels("data/labels_test.txt")
    #X_test = load_data("/home/ashadrin/Videos/HSE/data/data_test.txt")
    #y_test = load_labels("/home/ashadrin/Videos/HSE/data/labels_test.txt")
    print ("initial test data: ", np.array(X_test).shape)

    return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test))

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=968, output_dim=2000, activation='tanh'))
    model.add(Dense(121*8, activation='tanh'))
    #model.add(BatchNormalization())
    model.add(Reshape((121, 8), input_shape=(121*8,)))
    model.add(UpSampling1D(length=2))
    model.add(Convolution1D(64, 5, border_mode='same', activation='tanh'))
    model.add(UpSampling1D(length=2))
    model.add(Convolution1D(64, 5, border_mode='same', activation='tanh'))
    model.add(UpSampling1D(length=2))
    model.add(Convolution1D(1, 5, border_mode='same', activation='tanh'))
    return model

def discriminator_model():
    model = Sequential()
    #model.add(Dense(input_dim=968, output_dim=968, activation='tanh'))
    model.add(Reshape((968, 1), input_shape=(968*1,)))
    model.add(Convolution1D(121, 5, border_mode='same',
                        input_shape=(121,8), activation='tanh'))
    model.add(MaxPooling1D(pool_length=11))
    model.add(Convolution1D(121, 3, activation='tanh'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(968, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    return model
    
def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def combine_timeseries(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    print (width, height)
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image

def plot_sample(generated_signal, epoch, index, BATCH_SIZE):
    generated_signal_save = generated_signal.reshape((BATCH_SIZE, 8, 121))
    generated_signal_list = generated_signal_save.tolist()
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    for signal,num in zip(generated_signal_list, range(len(generated_signal_list))):
        m = list(map(list, zip(*signal)))
        plt.clf()
        plt.cla()
        plt.figure(figsize=(12,9))
        plt.title("Temporary image on "+str(epoch)+" epoch, "+str(index)+", VOC #"+str(num))
        plt.plot(m)
        #plt.legend(sensors, loc='best') 
        plt.xlabel('Time (sec)')
        plt.ylabel('dF')
        try:
            os.stat("temp_images")
        except:
            os.mkdir("temp_images") 
        try:
            os.stat("temp_images/"+str(epoch))
        except:
            os.mkdir("temp_images/"+str(epoch)) 
        plt.savefig("temp_images/"+str(epoch)+"/"+str(index)+"_"+str(num)+".png")
        plt.close('all')

def plot_final(generated_signal,BATCH_SIZE):
    generated_signal_save = generated_signal.reshape((BATCH_SIZE, 8, 121))
    generated_signal_list = generated_signal_save.tolist()
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    for signal,num in zip(generated_signal_list, range(len(generated_signal_list))):
        m = list(map(list, zip(*signal)))
        plt.clf()
        plt.cla()
        plt.figure(figsize=(12,9))
        plt.title("Generated VOC #"+str(num))
        plt.plot(m)
        plt.legend(sensors, loc='best') 
        plt.xlabel('Time (sec)')
        plt.ylabel('dF')
        try:
            os.stat("res_images")
        except:
            os.mkdir("res_images") 
        plt.savefig("res_images/generated_signal_"+str(num)+".png")
        plt.close('all')

def fit_polynom(X_train, N):
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    X_train_new = []
    for matr in X_train:
        matr_new = []
        for i in range(len(matr)):
            vec = matr[i]
            L = len(vec)            
            T = 1/250.0       
            t = np.linspace(1,L,L)*T   
            xx = np.asarray(t)
            yy = np.asarray(vec)
            z = np.asarray(np.polyfit(xx, yy, N))
            ff = np.poly1d(z)
            x_new = np.linspace(xx[0], xx[-1], len(xx))
            y_new = ff(x_new)
            matr_new.append(y_new)
        X_train_new.append(matr_new)
    return X_train_new
    
def normalize_data(data):
    norm_matrix = []
    for block in data:
        current_max = np.amax(block)
        norm_col = []
        for col in block:
            norm_col.append([float(i)//current_max for i in col])
        norm_matrix.append(norm_col)
    return norm_matrix

def train(BATCH_SIZE):
    (X_train_0, y_train), (X_test, y_test) = load_data_train()
   # print (X_train.shape)
   # print (X_train[0].shape)
   
    X_train_poly = fit_polynom(X_train_0, 5)
    X_train = normalize_data(X_train_poly)
    X_train = np.array(X_train)
    #X_train = np.array(X_train_poly)
    
    nsamples00, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples00,nx*ny))  

    generator = generator_model()    
    discriminator = discriminator_model() 
    #generator.summary()      
    #discriminator.summary()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    #discriminator_on_generator.summary() 
    
    d_optim = SGD(lr=0.01, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.01, momentum=0.9, nesterov=True)
    #d optim = g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    noise = np.zeros((BATCH_SIZE, 968))
    for epoch in range(500):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                #noise[i, :] = np.random.uniform(-1, 1, 968)
                gen_noise = np.random.multinomial(155, [1/150.]*121, size=1).tolist()*8
                flattened  = [val for sublist in gen_noise for val in sublist]
                noise[i, :] =  np.array(flattened)
                
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_signal = generator.predict(noise, verbose=0)
            generated_signal = generated_signal.reshape((BATCH_SIZE, 968))
            
            #print ("Should we plot?"+str(index % 20))
            #if index % 20 == 0:
            #if epoch % 10 == 0:
            #    plot_sample(generated_signal, epoch, index, BATCH_SIZE)
                        
            X = np.concatenate((image_batch, generated_signal))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            for i in range(BATCH_SIZE):
                #noise[i, :] = np.random.uniform(-1, 968, 1)
                gen_noise = np.random.multinomial(155, [1/150.]*121, size=1).tolist()*8
                flattened  = [val for sublist in gen_noise for val in sublist]
                noise[i, :] =  np.array(flattened)            
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            #print ("Should we save?"+str(index % 10))
            if index % 10 == BATCH_SIZE-1:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)

def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE*20, 968))
        for i in range(BATCH_SIZE*20):
           # noise[i, :] = np.random.uniform(-1, 1, 968)
            gen_noise = np.random.multinomial(155, [1/150.]*121, size=1).tolist()*8
            flattened  = [val for sublist in gen_noise for val in sublist]
            noise[i, :] =  np.array(flattened)
        generated_signals = generator.predict(noise, verbose=1)
        generated_signals = generated_signals.reshape((BATCH_SIZE, 968))
        d_pret = discriminator.predict(generated_signals, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_signals = np.zeros((BATCH_SIZE, 1) +
                               (generated_signals.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_signals[i, 0, :, :] = generated_signals[idx, 0, :, :]
        signals_to_show = nice_signals
    else:
        noise = np.zeros((BATCH_SIZE, 968))
        for i in range(BATCH_SIZE):
          #  noise[i, :] = np.random.uniform(-1, 1, 968)
            gen_noise = np.random.multinomial(155, [1/150.]*121, size=1).tolist()*8
            flattened  = [val for sublist in gen_noise for val in sublist]
            noise[i, :] =  np.array(flattened)
        generated_signals = generator.predict(noise, verbose=1)
        signals_to_show = generated_signals
    plot_final(signals_to_show,BATCH_SIZE)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)

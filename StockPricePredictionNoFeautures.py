# IMPORTING IMPORTANT LIBRARIES
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
import csv
data_dir = 'kospi-data-12.24' #directory with training data
#data_dir = 'kospi-data-day'
res_dir = 'result-data-12.24-adam/'#directory to store the results or restore model
files = os.listdir(data_dir)
print(files)

########CAN BE ADJUSTED#########
TRAINING = True #train the model
optimizer = 'adam' #'sgd' #'adagrad'
loss = 'mean_squared_error'
epochs = 5
REGULARIZATION = False #if need to regilarize 
reg = L1L2(l1=0.01, l2=0.01) #parameters for regularization
RESTORE = False #restore trained model from disk
################################

all_log = []
if TRAINING:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential, model_from_json
    from keras.layers import Dense, Activation
    from keras.layers import LSTM
    from keras.regularizers import L1L2
    import preprocessing 
    print("##### TRAINING #####")
    ##LOAD FILES FROM kospi-data dir
    for file_idx in range(len(files)):
    #for file_idx in range(2):
        plt.cla()
        plt.clf()

        filename = files[file_idx]
        if filename[0] == '.':
            continue
        name = filename.split('.')[0]
        print(file_idx, name)
        # FOR REPRODUCIBILITY
        np.random.seed(7)


        path = data_dir + '/' + name + '.csv'
        # IMPORTING DATASET 
        #dataset = pd.read_csv('apple_share_price.csv', usecols=[1,2,3,4])
        dataset = pd.read_csv(path, usecols=[0,1,2,3,4])
        dataset = dataset.reindex(index = dataset.index[::-1])
        
        # CREATING OWN INDEX FOR FLEXIBILITY
        obs = np.arange(1, len(dataset) + 1, 1)
        # TAKING DIFFERENT INDICATORS FOR PREDICTION
        OHLC_avg = dataset[['close', 'high', 'low', 'open']].mean(axis = 1)
        datetime = dataset[['day']]
        datetime = np.reshape(datetime.values, (len(datetime),1)) 
        # PLOTTING ALL INDICATORS IN ONE PLOT
        plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
        plt.legend(loc = 'upper right')
        img_name = res_dir+ name  + '_ohlc_avg.png'
        plt.savefig(img_name)

        # PREPARATION OF TIME SERIES DATASE
        # Calculate average of Low and High values
        OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) 

        #Scale the values
        scaler = MinMaxScaler(feature_range=(0, 1))
        OHLC_avg = scaler.fit_transform(OHLC_avg)
        # TRAIN-TEST SPLIT
        train_OHLC = int(len(OHLC_avg) * 0.75)
        test_OHLC = len(OHLC_avg) - train_OHLC
        train_datetime = int(len(datetime) * 0.75)
        test_datetime = len(datetime) - train_datetime
        train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]
        train_datetime, test_datetime = datetime[0:train_datetime,:], datetime[train_datetime:len(datetime),:]
        #print(train_OHLC.shape, train_datetime.shape)
        # TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
        trainX, trainY = preprocessing.new_dataset(train_OHLC, 1)
        testX, testY = preprocessing.new_dataset(test_OHLC, 1)
        train_datetimeX, train_datetimeY = preprocessing.new_dataset(train_datetime, 1)
        test_datetimeX, test_datetimeY = preprocessing.new_dataset(test_datetime, 1)
        # RESHAPING TRAIN AND TEST DATA
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        train_datetime = np.reshape(train_datetime, (train_datetime.shape[0], 1, train_datetime.shape[1]))
        test_datetime = np.reshape(test_datetime, (test_datetime.shape[0], 1, test_datetime.shape[1]))

        step_size = 1
        # LSTM MODEL
        model = Sequential()
        if REGULARIZATION:
            model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True, bias_regularizer=reg))
        else:
            model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
        model.add(LSTM(16))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model_name = res_dir + name + '_' + optimizer + '.json'
        model_weights = res_dir + name + '_' + optimizer + '.h5'
        # MODEL COMPILING AND TRAINING
        #SAVE MODEL AFTER TRAINING IF IT'S NOT RESTORED FROM DISK
        if not RESTORE:
            model.compile(loss=loss, optimizer=optimizer) # Try SGD, adam, adagrad and compare!!!
            model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)
            # evaluate the model
            # serialize model to JSON
            model_json = model.to_json()
            with open(model_name, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(model_weights)
            print("Saved model to disk")
        
        #LOAD MODEL FROM DISK
        else:
            json_file = open(model_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(model_weights)
            print("Loaded model from disk")
            
            # evaluate loaded model on test data
            model.compile(loss=loss, optimizer=optimizer) # Try SGD, adam, adagrad and compare!!!
            score = model.evaluate(trainX, trainY, verbose=0)

        # PREDICTION
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # DE-NORMALIZING FOR PLOTTING
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])


        # TRAINING RMSE
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        maxTrain = np.amax(trainY[0])
        minTrain = np.amin(trainY[:])
        rangeTrain = maxTrain - minTrain
        # NORMALIZED TRAIN RMSE
        trainNRMSE = trainScore/rangeTrain
        print('Train RMSE: %.2f' % (trainScore))
        print('Train NRMSE: %.2f' % (trainNRMSE))

        # TEST RMSE
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:]))
        maxTest = np.amax(testY[0])
        minTest = np.amin(testY[:])
        rangeTest = maxTest - minTest
        # NORMALIZED TEST RMSE
        testNRMSE = testScore/rangeTest
        print('Test RMSE: %.2f' % (testScore))
        print('Test NRMSE: %.2f' % (testNRMSE))
        # CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
        trainPredictPlot = np.empty_like(OHLC_avg)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

        # CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
        testPredictPlot = np.empty_like(OHLC_avg)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict

        # DE-NORMALIZING MAIN DATASET 
        OHLC_avg = scaler.inverse_transform(OHLC_avg)

        # PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
        plt.plot(OHLC_avg, 'g', label = 'original dataset')
        plt.plot(trainPredictPlot, 'r', label = 'result set')
        plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
        plt.legend(loc = 'upper right')
        plt.xlabel('Time in Minutes')
        plt.ylabel('OHLC Value of Stocks')
        img_name = res_dir +  name + '_OHLC_test.png'
        print(img_name)
        plt.savefig(img_name)
        #SAVE ORIGINAL VALUES AND PREDICTIONS TO CSV
        df = pd.DataFrame({'datetime': train_datetimeY[:], 'original': trainY[0], 'predicted': trainPredict[:,0]}, columns=['datetime','original', 'predicted'])
        filename = res_dir + name + '_train-res.csv'
        df.to_csv(filename,index=False)
        df = pd.DataFrame({'datetime': test_datetimeY[:],'original': testY[0], 'predicted': testPredict[:,0]}, columns=['datetime','original', 'predicted'])
        filename = res_dir + name + '_test-res.csv'
        df.to_csv(filename,index=False)

        filename = res_dir + name + '_log.csv'
        all_log.append([name, maxTrain, minTrain, rangeTrain, trainScore, trainNRMSE, maxTest, minTest, rangeTest, testScore, testNRMSE])
        #WRITE LOG TO CSV
        with open(filename, mode='w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Max train value', maxTrain])
            csv_writer.writerow(['Min train value', minTrain])
            csv_writer.writerow(['Train values range', rangeTrain])
            csv_writer.writerow(['Train RMSE', trainScore])
            csv_writer.writerow(['Train NRMSE', trainNRMSE])
            csv_writer.writerow(['Max test value', maxTest])
            csv_writer.writerow(['Min test value', minTest])
            csv_writer.writerow(['Test values range', rangeTest])
            csv_writer.writerow(['Test RMSE', testScore])
            csv_writer.writerow(['Test NRMSE', testNRMSE])

    filename = res_dir + 'training-log.csv'
    #WRITE LOG TO CSV IN THE END
    total_train_nrmse = 0
    total_test_nrmse = 0
    with open(filename, mode='w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Stock name','Max train val', 'Min train val', 'Train vals range', 'Train RMSE', 'Train NRMSE', 'Max test val', 'Min test val', 'Test vals range', 'TestRMSE', 'TestNRMSE'])
        for row in all_log:
            total_train_nrmse += row[5]
            total_test_nrmse += row[10]
            csv_writer.writerow(row)
        avr_train_nrmse = total_train_nrmse/50.0
        avr_test_nrmse = total_test_nrmse/50.0
        csv_writer.writerow(['AVERAGE','','','','',avr_train_nrmse,'','','','',avr_test_nrmse])
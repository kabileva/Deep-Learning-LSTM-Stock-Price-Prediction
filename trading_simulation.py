import csv
import os
########CAN BE ADJUSTED#########

FEE_TAX = 0.0035 #transaction fee percentage
data_dir = 'kospi-data-12.24' #the directory with the data obtained by Kiwoom API
#Directories containing result data obtained by training with different parameters:
data_dirs = ['result-data-12.24-volume-adagrad/','result-data-12.24-adagrad/','result-data-12.24-sgd/','result-data-12.24-volume-sgd/', 'result-data-12.24-adam/','result-data-12.24-volume-adam/']
combined_filename = 'combined_log.csv' #name of the file where combined data will be saved
SELL_WHEN_PROFIT = 0.15 #Sell when transaction profit is bigger than..
BUY_WHEN_PREDICTED = 1.0000 #buy when the predicted price to curent price ratio is...
TRADING = True #simulate the trading with a given strategy
STRATEGY = 'buy_when_rise' #buy when is predicted to rise, sell when predicted to fall and profit is bigger than SELL_WHEN_PROFIT
#STRATEGY = 'buy_when_0.7pct'
profit_filename = STRATEGY + '_daily_profit.csv' #name of the file where profit percentage will be saved
################################

##################################################

################ COMBINING DATA  #################

# The following code combines result data obtained
# by training the model with different parameters
# based on the TEST NRMSE (it chooses the parameters
# which give the best normalized root mean squared
# error for a particular stock)

##################################################


#{
# 'stock_name': [name, maxTrain, minTrain, rangeTrain, trainScore, trainNRMSE, maxTest, minTest, rangeTest, testScore, testNRMSE]
# }
combined_log = {} 

for dir_name in data_dirs:
    file_path = dir_name + 'training-log.csv'
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            name = row[0]
            if name[0]=='A' or name[0]=='a' or name[0]=='S' or name[0]=='s':
                continue
            if not name in combined_log:
                combined_log[name] = row + [dir_name]
            else:
                testNRMSE = float(row[10])
                trainNRMSE = float(row[5])
                currTestNRMSE = float(combined_log[name][10])
                currTrainNRMSE = float(combined_log[name][5])
                if testNRMSE < currTestNRMSE:
                    combined_log[name] = row + [dir_name]

total_train_nrmse = 0
total_test_nrmse = 0
with open(combined_filename, mode='w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Stock name','Max train val', 'Min train val', 'Train vals range', 'Train RMSE', 'Train NRMSE', 'Max test val', 'Min test val', 'Test vals range', 'TestRMSE', 'TestNRMSE'])
    for name in combined_log:
        row = combined_log[name]
        total_train_nrmse += float(row[5])
        total_test_nrmse += float(row[10])
        csv_writer.writerow(row)
    avr_train_nrmse = total_train_nrmse/50.0
    avr_test_nrmse = total_test_nrmse/50.0
    print('Train NRMSE: ', avr_train_nrmse)
    print('Test NRMSE: ', avr_test_nrmse)
    csv_writer.writerow(['AVERAGE','','','','',avr_train_nrmse,'','','','',avr_test_nrmse])

##################################################

################ TRADING SIMULATION ##############

##################################################

if TRADING:
    files = os.listdir(data_dir)
    print("##### TRADING SIMULATION #####")
    daily_profit = dict() #dictionary for storing daily transactions profit
    #{
    #   day: {'spent': num1, 'earned': num2}
    # }
    print("Reading data...")
    for file_idx in range(len(files)):
    #for file_idx in range(2):
        filename = files[file_idx]
        if filename[0] == '.':
            continue
        name = filename.split('.')[0]
        res_dir = combined_log[name][-1]
        path = res_dir + name + '_test-res.csv'
        recs = []
        #insert days as keys to daily_profit dictonary
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                day = str(row['datetime'])[:8]
                if not day in daily_profit:
                    daily_profit[day] = {"spent": 0, "earned": 0}
                recs.append(row)
        if STRATEGY == 'buy_when_rise':
            volHave = False #False - don't have this stock, True - have this stock
            price_bought = 0
            total_recs = len(recs) 
            #Read prediction based on test data
            for i in range(1,len(recs)-1):
                day = str(recs[i]['datetime'])[:8] #date (YYYYMMDD)
                currTrue = float(recs[i]['original']) #real data of current minute
                nextTrue = float(recs[i+1]['original']) #real data of next minute  
                nextTrain = float(recs[i+1]['predicted']) #predicted data of next minute
                #predicted to rise and don't have this stock
                if nextTrain/currTrue > BUY_WHEN_PREDICTED and not volHave: 
                    #buy stock
                    volHave = True
                    #record the price for which stock was bought
                    price_bought = currTrue
                #predicted to fall and have this stock
                elif nextTrain <= currTrue and volHave:
                    #calculate expected profit
                    transaction_profit = (currTrue - price_bought - FEE_TAX*2*currTrue)/price_bought
                    #sell stock if the profit is higher than transaction fee
                    if transaction_profit > SELL_WHEN_PROFIT:
                        #sell
                        volHave = False
                        #calculate profit
                        spent_money = price_bought
                        earned_money = currTrue - price_bought - FEE_TAX*2*currTrue
                        #add profit to the list of a day's profits
                        #daily_profit[day]
                        daily_profit[day]['spent'] += spent_money
                        daily_profit[day]['earned'] += earned_money
                        price_bought = 0
        elif STRATEGY == 'buy_when_0.7pct':
            volHave = False #False - don't have this stock, True - have this stock
            profit = 0
            chosen_correctly = 0
            price_bought = 0
            fall_flag = False #True - predictied to fall, False - predicted to increase
            total_recs = 0
            #Read prediction based on test data
            for i in range(1,len(recs)-1):
                day = str(recs[i]['datetime'])[:8] #date (YYYYMMDD)
                currTrue = float(recs[i]['original']) #real data of current minute
                nextTrue = float(recs[i+1]['original']) #real data of next minute  
                nextTrain = float(recs[i+1]['predicted']) #predicted data of next minute
                predicted_price_increase = nextTrain/currTrue
                actual_price_increase = nextTrue/currTrue
                #expected to rise by 0.7% and don't have this stock
                if not volHave and predicted_price_increase >= 1.007:
                    #buy stock
                    volHave = True
                    #record the current price
                    price_bought = currTrue
                    #isn't predicted to fall
                    fall_flag = False
                #bough stock just rised by 0.7%
                if volHave and currTrue/price_bought >= 1.007:
                    #sell the stock
                    volHave = False
                    #calculate profit
                    spent_money = price_bought
                    earned_money = currTrue - price_bought - FEE_TAX*2*currTrue
                    #add profit to the list of a day's profits
                    daily_profit[day]['spent'] += spent_money
                    daily_profit[day]['earned'] += earned_money
                    price_bought = 0
                    fall_flag = False
                #if falls
                if volHave and currTrue/price_bought < 1:
                    #set fall flag to True
                    fall_flag = True
                #if falls under -1.5%
                if volHave and fall_flag and currTrue/price_bought <= 0.985:
                    #remove fall flag
                    fall_flag = False
                    #calculate profit
                    transaction_profit = (currTrue - price_bought - FEE_TAX*2*currTrue)/price_bought
                    #add to daily profit
                    spent_money = price_bought
                    earned_money = currTrue - price_bought - FEE_TAX*2*currTrue
                    #add profit to the list of a day's profits
                    #daily_profit[day]
                    daily_profit[day]['spent'] += spent_money
                    daily_profit[day]['earned'] += earned_money
                    #sell stock
                    volHave = False
                #if was falling but recovers by 0.34%
                if volHave and fall_flag and currTrue/price_bought >= 1.0034:
                    #remove fall flag
                    fall_flag = False
                    #calculate profit
                    transaction_profit =(currTrue - price_bought - FEE_TAX*2*currTrue)/price_bought
                    spent_money = price_bought
                    earned_money = currTrue - price_bought - FEE_TAX*2*currTrue
                    #add profit to the list of a day's profits
                    #daily_profit[day]
                    daily_profit[day]['spent'] += spent_money
                    daily_profit[day]['earned'] += earned_money
                    volHave = False
    total = 0
    print("DAILY PROFIT")
    #sort dictionary by day
    keylist = list(daily_profit.keys())
    keylist.sort()
    #WRITE DAILY PROFIT TO CSV
    with open(profit_filename, mode='w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for key in keylist:
            val = daily_profit[key]
            if not val['spent'] == 0:
                #Divide earned during the day money by spent during the day money to find day profit
                avr_day = val['earned']/val['spent']
            else:
                avr_day = 0
            print(key, avr_day)
            csv_writer.writerow([key, avr_day])
            total += avr_day
        #calculate average daily profit
        avr_total = total/len(daily_profit)
        print("Average daily profit: ", avr_total)
        csv_writer.writerow(["AVERAGE DAILY PROFIT: ", avr_total])

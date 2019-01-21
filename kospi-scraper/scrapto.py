from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
from pandas_datareader import data
import sqlite3
import sys
import pickle
import pandas as pd
import re
import glob
import time
import csv
class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()
        self._create_kiwoom_instance()
        self._set_signal_slots()

    # COM을 사용하기 위한 메서드
    def _create_kiwoom_instance(self):
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

    def _set_signal_slots(self):
        # 로그인할 시 OnEventConnect 이벤트 발생
        self.OnEventConnect.connect(self._event_connect)
        # tr후 이벤트 발생
        self.OnReceiveTrData.connect(self._receive_tr_data)

    # 로그인 메서드, 로그인 과정에서 프로그램이 진행되면 안 되기 때문에
    # 이벤트 루프 생성
    def comm_connect(self):
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    # 로그인 성공 여부 메서드
    def _event_connect(self, err_code):
        if err_code == 0:
            print("connected")
        else:
            print("disconnected")
        self.login_event_loop.exit()

    # tr 입력값을 서버 통신 전에 입력
    # ex. SetInputValue("종목코드","000660")
    def set_input_value(self,id,value):
        self.dynamicCall("SetInputValue(QString,QString)", id, value)

    # tr을 서버에 전송한다
    def comm_rq_data(self, rqname, trcode, next, screen_no):
        self.dynamicCall("CommRqData(QString, QString, int, QString)", rqname, trcode, next, screen_no)
        self.tr_event_loop = QEventLoop()
        self.tr_event_loop.exec_()

    # 서버한테 받은 데이터를 반환한다.
    def _comm_get_data(self, code, real_type, field_name, index, item_name):
        ret = self.dynamicCall("CommGetData(QString, QString, QString, int, QString)", code,
                               real_type, field_name, index, item_name)
        return ret.strip()

    # 서버한테 받은 데이터의 갯수를 반환한다.
    def _get_repeat_cnt(self, trcode, rqname):
        ret = self.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        return ret

    def _receive_tr_data(self, screen_no, rqname, trcode, record_name, next, unused1, unused2, unused3, unused4):
        #print("receive_tr_data call")
        if next == '2':
            self.remained_data = True
        else:
            self.remained_data = False

        if rqname == "opt10080_req":
            self._opt10080(rqname, trcode)
        elif rqname == "opt10081_req":
            self._opt10081(rqname, trcode)

        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass

    # KODEX 코스닥 150 레버리지 종목의 분봉 데이터를 요청한다.
    # 약 100일치 분봉 데이터를 요청한다.
    def req_minute_data(self):
        tick_val = 15
        self.set_input_value("종목코드",ticker)
        self.set_input_value("틱범위", str(tick_val))
        self.set_input_value("수정주가구분", str(0))
        self.comm_rq_data("opt10080_req", "opt10080", 0, "1999")
        # for interval in range(1,8):
        #     print("interval: ", interval)
        for i in range(99):
            time.sleep(0.2)
            self.set_input_value("종목코드", ticker)
            self.set_input_value("틱범위", str(tick_val))
            self.set_input_value("수정주가구분", str(0))
            self.comm_rq_data("opt10080_req", "opt10080", 2, "1999")
        # for i in range(48):
        #     print("i: ", i)
        #     time.sleep(0.2)
        #     self.set_input_value("종목코드", ticker)
        #     self.set_input_value("틱범위", 1)
        #     self.set_input_value("수정주가구분", 0)
        #     self.comm_rq_data("opt10080_req", "opt10080", 2, "1998")
            

    # KODEX 코스닥 150 레버리지 종목의 일봉 데이터를 요청한다.
    def req_day_data(self):
        self.set_input_value("종목코드",ticker)
        self.set_input_value("기준일자", "20181204")
        self.set_input_value("수정주가구분",0)
        self.comm_rq_data("opt10081_req", "opt10081",0,"2000")


    # 서버에게 받은 분봉 데이터를 kosdaq_leve 테이블에 저장한다.
    # 또한 kosdaq_start 테이블에 매일 시가 정보를 저장한다.
    def _opt10080(self,rqname,trcode):
        data_cnt = self._get_repeat_cnt(trcode,rqname)
        global list_start
        global list_leve
        global prev_day
        #df_leve = pd.DataFrame(columns=['day', 'high', 'low'])
        #df_start = pd.DataFrame(columns=['day', 'start'])   

        for i in range(data_cnt):
            # print(ticker +" "+ kor_name+ " 분봉 데이터 저장중")
            day = self._comm_get_data(trcode, "",rqname, i, "체결시간")
            if (day[:8])=="20180504":
                print("threshold")
                #break
                #print(prev_day, day)
            open = self._comm_get_data(trcode, "", rqname, i, "시가")
            high = self._comm_get_data(trcode, "", rqname, i, "고가")
            low = self._comm_get_data(trcode, "", rqname, i, "저가")
            close = self._comm_get_data(trcode, "", rqname, i, "현재가")
            volume = self._comm_get_data(trcode, "", rqname, i, "거래량")
            if(high[0] == '-'):
                high = high[1:]
            if(low[0] == '-'):
                low = low[1:]
            ohlcv_leve = {'day': day, 'low': int(high), 'high': int(low)}
            #ohlcv_leve = {'day': day}
            list_leve.append(ohlcv_leve)
            #df_leve.append(ohlcv_leve, ignore_index=True)
            #self.db.insert_Leve(day, abs(int(high)), abs(int(low)))
            #print(list_leve)
            if day[8:] == "090000":

                start = self._comm_get_data(trcode, "",rqname, i, "시가")
                if(start[0] == '-'):
                    start = start[1:]
                ohlcv_start = {'day': day, 'start': int(start)}
                list_start.append(ohlcv_start)
                #df_start.append(ohlcv_start, ignore_index=True)
                #self.db.insert_Start(day,abs(int(start)))
            prev_day = day


    # 서버에게 받은 일봉 데이터를 DB에 저장한다.
    def _opt10081(self,rqname, trcode):
        for i in range(150):
            print("일봉 데이터 저장중")
            day = self._comm_get_data(trcode, "",rqname, i, "일자")
            end = self._comm_get_data(trcode, "",rqname, i,"현재가")
            start = self._comm_get_data(trcode, "",rqname, i,"시가")
    def get_code_list_by_market(self, market):
        code_list = self.dynamicCall("GetCodeListByMarket(QString)", market)
        code_list = code_list.split(';')
        return code_list[:-1]
#def save_kospi(ticker):
#    for stock in kospi.values:
#        kor_name = stock[0]
#        ticker = stock[1]
#        print(stock[0])
#        print(stock[1])
#       con = sqlite3.connect("./kospi/"+kor_name+ " - "+ticker+ ".csv")
#       df = data.get_data_yahoo(ticker + '.KS', '2018-11-20', thread=20)
#       df.to_csv('./kospi/'+ kor_name +'- {}.csv'.format(ticker))
#       print('{}.csv is saved'.format(ticker))



if __name__ == "__main__":
    
    #kospi = pd.read_pickle('./kospi.pickle')
    #kospi.to_csv('./{}.csv')
#    checker=0

    MARKET_KOSPI   = 0

    prev_day = 0
    code_idx = 0
#    for stock in kospi.values:
#        kor_name = stock[0]
    #IDX = 0
    IDX = int(sys.argv[1])
    rng = 1
    stock_list = []
    with open('kospi_top50.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            stock_list.append(row)
        
    for i in range(IDX,IDX+rng):
        stock_name = stock_list[i][0]
        ticker = stock_list[i][1]
        print(stock_name,ticker)
        print("종목 코드:")
        #ticker = '001810'

        print("00")
    #      if checker==0:

        # for i in range(0,1522,2):
            
        #     app = QApplication(sys.argv)
        #     kiwoom = Kiwoom()
        #     kiwoom.comm_connect()
        #     kospi_codes = kiwoom.get_code_list_by_market(MARKET_KOSPI)
        #     for j in range(2):
        list_leve = []
        list_start = []
        app = QApplication(sys.argv)
        kiwoom = Kiwoom()
        kiwoom.comm_connect()
        #print("iteration:", i+j)
        #ticker = kospi_codes[i+j]

        print("11")
        print("22")
        kiwoom.req_minute_data()
        print("33")
        df_leve = pd.DataFrame(list_leve)
        leve_csv = 'kospi-data/'+ str(IDX+1)+ '_'+ stock_name + '_' + ticker + '.csv'
        df_leve.to_csv(leve_csv)
        app.exit()
        print("44")
        time.sleep(10)
        
    #       checker=checker+1

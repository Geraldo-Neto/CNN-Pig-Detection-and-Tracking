# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
# Added a signal



class Ui_MainWindow(QtWidgets.QWidget):

    changedValue = QtCore.pyqtSignal(int)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1040, 518)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(254, 255, 239))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(254, 255, 239))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(254, 255, 239))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(254, 255, 239))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(254, 255, 239))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(254, 255, 239))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(254, 255, 239))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(254, 255, 239))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(254, 255, 239))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        MainWindow.setPalette(palette)
        MainWindow.setStyleSheet("background-color:  #feffef")
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(0, 0))
        self.centralwidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(22)
        self.label_13.setFont(font)
        self.label_13.setAutoFillBackground(False)
        self.label_13.setStyleSheet("font-color: #0499D0")
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.verticalLayout.addWidget(self.label_13)
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(10)
        font.setItalic(True)
        self.label_14.setFont(font)
        self.label_14.setStyleSheet("font-color: #0499D0")
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.verticalLayout.addWidget(self.label_14)
        spacerItem = QtWidgets.QSpacerItem(17, 37, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem)
        self.btSelectVideo = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btSelectVideo.sizePolicy().hasHeightForWidth())
        self.btSelectVideo.setSizePolicy(sizePolicy)
        self.btSelectVideo.setMinimumSize(QtCore.QSize(0, 100))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(10)
        self.btSelectVideo.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("outline-perm_media-24px.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btSelectVideo.setIcon(icon)
        self.btSelectVideo.setFlat(False)
        self.btSelectVideo.setObjectName("btSelectVideo")
        self.verticalLayout.addWidget(self.btSelectVideo)
        spacerItem1 = QtWidgets.QSpacerItem(17, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem1)
        self.setupWidget = QtWidgets.QWidget(self.centralwidget)
        self.setupWidget.setEnabled(False)
        self.setupWidget.setObjectName("setupWidget")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.setupWidget)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_19 = QtWidgets.QLabel(self.setupWidget)
        self.label_19.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.label_19.setFont(font)
        self.label_19.setStyleSheet("font-color: #0499D0")
        self.label_19.setObjectName("label_19")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.label_20 = QtWidgets.QLabel(self.setupWidget)
        self.label_20.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.label_20.setFont(font)
        self.label_20.setStyleSheet("font-color: #0499D0")
        self.label_20.setObjectName("label_20")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.label_21 = QtWidgets.QLabel(self.setupWidget)
        self.label_21.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.label_21.setFont(font)
        self.label_21.setStyleSheet("font-color: #0499D0")
        self.label_21.setObjectName("label_21")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_21)
        self.sbNFeeders = QtWidgets.QSpinBox(self.setupWidget)
        self.sbNFeeders.setEnabled(True)
        self.sbNFeeders.setObjectName("sbNFeeders")
        self.sbNFeeders.setValue(1)
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.sbNFeeders)
        self.sbStartTime = QtWidgets.QSpinBox(self.setupWidget)
        self.sbStartTime.setEnabled(True)
        self.sbStartTime.setMaximum(99999999)
        self.sbStartTime.setValue(270)
        self.sbStartTime.setObjectName("sbStartTime")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.sbStartTime)
        self.sbEndTime = QtWidgets.QSpinBox(self.setupWidget)
        self.sbEndTime.setEnabled(True)
        self.sbEndTime.setMaximum(99999999)
        self.sbEndTime.setObjectName("sbEndTime")
        self.sbEndTime.setValue(285)
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.sbEndTime)
        self.verticalLayout_13.addLayout(self.formLayout_4)
        self.horizontalLayout_7.addLayout(self.verticalLayout_13)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem2)
        self.lbVideoFile = QtWidgets.QLabel(self.setupWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbVideoFile.setFont(font)
        self.lbVideoFile.setObjectName("lbVideoFile")
        self.horizontalLayout_7.addWidget(self.lbVideoFile)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem3)
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.cbGenerateVideo = QtWidgets.QCheckBox(self.setupWidget)
        self.cbGenerateVideo.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.cbGenerateVideo.setFont(font)
        self.cbGenerateVideo.setStyleSheet("font-color: #0499D0")
        self.cbGenerateVideo.setObjectName("cbGenerateVideo")
        self.cbGenerateVideo.setChecked(True)
        self.verticalLayout_14.addWidget(self.cbGenerateVideo)
        self.cbGenerateTable = QtWidgets.QCheckBox(self.setupWidget)
        self.cbGenerateTable.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.cbGenerateTable.setFont(font)
        self.cbGenerateTable.setStyleSheet("font-color: #0499D0")
        self.cbGenerateTable.setObjectName("cbGenerateTable")
        self.cbGenerateTable.setChecked(True)
        self.verticalLayout_14.addWidget(self.cbGenerateTable)
        self.horizontalLayout_7.addLayout(self.verticalLayout_14)
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.horizontalLayout_7.addLayout(self.verticalLayout_15)
        self.verticalLayout.addWidget(self.setupWidget)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.processingWidget = QtWidgets.QWidget(self.centralwidget)
        self.processingWidget.setEnabled(False)
        self.processingWidget.setObjectName("processingWidget")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.processingWidget)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_22 = QtWidgets.QLabel(self.processingWidget)
        self.label_22.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.label_22.setFont(font)
        self.label_22.setStyleSheet("font-color: #0499D0")
        self.label_22.setObjectName("label_22")
        self.horizontalLayout_8.addWidget(self.label_22)
        spacerItem4 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem4)
        self.line_5 = QtWidgets.QFrame(self.processingWidget)
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.horizontalLayout_8.addWidget(self.line_5)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.btStart = QtWidgets.QPushButton(self.processingWidget)
        self.btStart.setEnabled(True)
        self.btStart.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.btStart.setFont(font)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("outline-play_arrow-24px.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btStart.setIcon(icon1)
        self.btStart.setFlat(False)
        self.btStart.setObjectName("btStart")
        self.verticalLayout_4.addWidget(self.btStart)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.processingWidget)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lcdElapsedTime = QtWidgets.QLCDNumber(self.processingWidget)
        self.lcdElapsedTime.setSmallDecimalPoint(True)
        self.lcdElapsedTime.setProperty("value", 0.0)
        self.lcdElapsedTime.setProperty("intValue", 0)
        self.lcdElapsedTime.setObjectName("lcdElapsedTime")
        self.lcdElapsedTime.setEnabled(True)
        self.horizontalLayout_2.addWidget(self.lcdElapsedTime)
        self.btStop = QtWidgets.QPushButton(self.processingWidget)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("outline-stop-24px.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btStop.setIcon(icon2)
        self.btStop.setObjectName("btStop")
        self.btStop.setFlat(False)
        self.btStop.setEnabled(False)
        self.horizontalLayout_2.addWidget(self.btStop)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.pbProgress = QtWidgets.QProgressBar(self.processingWidget)
        self.pbProgress.setProperty("value", 0)
        self.pbProgress.setObjectName("pbProgress")
        self.verticalLayout_4.addWidget(self.pbProgress)
        self.horizontalLayout_8.addLayout(self.verticalLayout_4)
        self.verticalLayout.addWidget(self.processingWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_13.setText(_translate("MainWindow", "DETECTOR DE COMPORTAMENTO ANIMAL"))
        self.label_14.setText(_translate("MainWindow", "Para iniciar, selecione um arquivo de vídeo"))
        self.btSelectVideo.setText(_translate("MainWindow", "SELECIONAR VIDEO"))
        self.label_19.setText(_translate("MainWindow", "Número de comedouros:"))
        self.label_20.setText(_translate("MainWindow", "Início (segundos):"))
        self.label_21.setText(_translate("MainWindow", "Fim (segundos):"))
        self.lbVideoFile.setText(_translate("MainWindow", "Video_selecionado.avi"))
        self.cbGenerateVideo.setText(_translate("MainWindow", "Gerar vídeo de saída"))
        self.cbGenerateTable.setText(_translate("MainWindow", "Gerar tabela Excel"))
        self.label_22.setText(_translate("MainWindow", "Desenvolvido por:    Geraldo Neto  -        geraldolcneto123@gmail.com \n"
"Zootecnista:              Rodrigo de Lima -   rodrigolimadomingos95@hotmail.com "))
        self.btStart.setText(_translate("MainWindow", "INICIAR"))
        self.label_2.setText(_translate("MainWindow", "Tempo processado (segundos):"))
        self.btStop.setText(_translate("MainWindow", "PARAR"))
        
        ###################################################################
        self.btSelectVideo.clicked.connect(self.openFileNameDialog)
        self.btStart.clicked.connect(self._onbtStartClicked)
        self.btStop.clicked.connect(self._onbtStopClicked)
        self.detector = None
        
        self.changedValue.connect(self.set_pbProgressValue)


    def closeEvent(self, event):
        
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()      
        
    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileNames, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"Abrir arquivo de vídeo", "","AVI (*.avi);; MP4 (*.mp4)", options=options)
        #print(fileNames)
        if fileNames:
            self.fileNames = fileNames
            self._onFileSelected(self.fileNames[0])
            

    def _onFileSelected(self,fileName):
        self.updateProcessingInterface(fileName)
        

        
    def updateProcessingInterface(self,fileName):
        self.setupWidget.setEnabled(True)
        self.processingWidget.setEnabled(True)
        self.lbVideoFile.setText(fileName)
        

    def getProcessingData(self,fileName):
        self.endTime = self.sbEndTime.value()
        self.startTime = self.sbStartTime.value()
        self.nFeeders = self.sbNFeeders.value()
        self.generateVideo = self.cbGenerateVideo.isChecked()
        self.generateTable = self.cbGenerateTable.isChecked()
        

    def _onbtStartClicked(self):
        self.endTime = self.sbEndTime.value()
        self.startTime = self.sbStartTime.value()
        self.nFeeders = self.sbNFeeders.value()
        self.generateVideo = self.cbGenerateVideo.isChecked()
        self.generateTable = self.cbGenerateTable.isChecked()
        self.startProcessing()

    def _update_processing_interface_processing(self,state):
        
        self.btStart.setEnabled(not state)
        self.btStop.setEnabled(state)
        self.pbProgress.setEnabled(state)
        self.lcdElapsedTime.setEnabled(state)
        self.btSelectVideo.setEnabled(not state)
        self.setupWidget.setEnabled(not state)

    def _onbtStopClicked(self):
        #self.videoProcessingProcess.terminate()
        
        self.detector.stop()
        #self.pbProgress.setProperty("value", 0)
        self._update_processing_interface_processing(False)
       

    @QtCore.pyqtSlot(int)
    def set_pbProgressValue(self, val):
        self.pbProgress.setProperty("value", val)


    def start_progress_watching(self):
        i = 0
        self.lcdElapsedTime.setEnabled(True)
        while(True):
            lock = threading.Lock()
            lock.acquire()
            
            if (self.detector.finished):
                
                #self.pbProgress.setProperty("value", 100)
                self.changedValue.emit(100)
                self._update_processing_interface_processing(False)
                lock.release()
                break
            else:
                
                progress = self.detector.processing_progress
                self.changedValue.emit(progress)
                #self.pbProgress.setProperty("value", progress)
                self.lcdElapsedTime.setProperty("intValue", int(i))
            i+=0.5
            
                
            lock.release()
            time.sleep(0.5)
        
    def startProcessing(self):

        self._update_processing_interface_processing(True)
        feeders = []
        self.scale_factor = 1.0
        self.frame_step = 5
        
        for i in range(self.nFeeders):
            #print("Desenhe o comedouro {}".format(i+1))
            feeders.append(feeder_delimiter.get_feeder_points(self.fileNames[0],i+1,self.scale_factor))
        
        self.dispatcherThread = threading.Thread(target = self.process_queue_dispatcher, args = ([feeders]))
        self.dispatcherThread.start()
        

    def process_queue_dispatcher(self,feeders):
        
        for fn in self.fileNames:
            self.detector= eating_pig_detector_class.PigDetector()
            
            self.videoProcessingThread = threading.Thread(target=self.detector.start,
                                                     args=(feeders,fn, False,
                                                           self.startTime,self.endTime,
                                                           self.generateVideo,self.generateTable,self.frame_step,
                                                           self.scale_factor))
            
            self.videoProcessingThread.start()

            self.videoProcessingWatchingThread =  threading.Thread(target=self.start_progress_watching,
                                                                   args = ())        
            self.videoProcessingWatchingThread.start()

            self.videoProcessingWatchingThread.join()
            
        
        
        
        #self.detector_qthread = swineDetectorThread(self.fileName, False,self.startTime,self.endTime,self.generateVideo,self.generateTable)
        #self.detector_qthread.run()
        
        
        
        
if __name__ == "__main__":
    
    import sys, eating_pig_detector_class, feeder_delimiter, threading,time
    #from swine_detector_thread import swineDetectorThread
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
    
    








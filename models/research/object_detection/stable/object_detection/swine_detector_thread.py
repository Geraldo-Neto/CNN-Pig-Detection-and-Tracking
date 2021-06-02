from PyQt5.QtCore import QThread

class swineDetectorThread(QThread):
    def __init__(self,fileName, debugging,startTime,endTime,generateVideo,generateTable):
        QThread.__init__(self)
        self.fileName =  fileName
        self.debugging = debugging
        self.startTime =  startTime
        self.endTime = endTime
        self.generateVideo = generateVideo
        self.generateTable = generateTable

    def __del__(self):
        self.wait()


    def run(self):
        import eating_pig_detector, feeder_delimiter
        feeders = []
        for i in range(self.nFeeders):
            #print("Desenhe o comedouro {}".format(i+1))
            feeders.append(feeder_delimiter.get_feeder_points(self.fileName,i+1))
        start(feeders, self.fileName, False,self.startTime,self.endTime,self.generateVideo,self.generateTable)

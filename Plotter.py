import numpy        as np
import threading    as Thr
import collections  as C
import PyQt4.QtCore as qtc
import PyQt4.QtGui  as qtg
import pyqtgraph    as pg


################################################################################
## The Plotter Class provides an easy way to plot lines and distributions
################################################################################
class Plotter(qtc.QObject):

    ## Usual plot type
    PLOT = 0
    ## Percentile plot type
    PERC = 1
    
    ## Font used to display the groups in the list
    GRP_FONTS     = qtg.QFont("", -1, qtg.QFont.Black, False)
    ## Font used to display an error
    ERROR_FONTS   = qtg.QFont("", -1, qtg.QFont.Bold , False)
    ## Font used to display a plot in the list
    PLT_FONTS     = qtg.QFont()
    ## Font used to display a subplot in the subplot list
    SUB_PLT_FONTS = qtg.QFont()
    ## Basic color used in the percentile charts
    PER_COLOR     = pg.mkColor("#ff7500")
    ## Color used to display an error
    FG_ERR_COLOR  = pg.mkColor("#ff3e04")
    
    ## 'Role' id for the group name
    GRP_ROLE      = qtc.Qt.UserRole
    ## 'Role' id for the plot name
    PLT_ROLE      = qtc.Qt.UserRole + 1
    ## 'Role' id that indicates if an error already arised or not
    ERROR_ROLE    = qtc.Qt.UserRole + 2
    ## 'Role' id for the subplot id
    SUB_ID        = qtc.Qt.UserRole + 3
    
    ## Signal used to ask the plotting thread to update the list of plots
    update_list_sig       = qtc.pyqtSignal()
    ## Signal used to ask the plotting thread to update a plot
    update_plot_sig       = qtc.pyqtSignal(str, str)
    ## Signal used to ask the plotting thread to update a percetile plot
    update_percentile_sig = qtc.pyqtSignal(str, str)
    
    ## The randColor static method computes and returns a random color
    #
    #   @return A QColor with random channels R,G,B
    def randColor():
        return pg.mkColor(np.random.randint(0, 256),
                          np.random.randint(0, 256),
                          np.random.randint(0, 256))
   
    ## The Plotter constructor
    def __init__(self):
        qtc.QObject.__init__(self)
        self._win     = None
        self._list    = None
        self._plt     = None
        self._xm      = None
        self._xM      = None
        self._datas   = C.OrderedDict()
        self._listLck = Thr.Lock()
        self._dataLck = Thr.Lock()
        self.initialize()
        self.connectSignals()
        
    ## The initialize method initializes the Plotter class
    def initialize(self):
        self._win   = qtg.QWidget()
        self._list  = qtg.QListWidget()
        self._subl  = qtg.QListWidget()
        self._plt   = pg.PlotWidget()
        self._xmw   = qtg.QLineEdit()
        self._xMw   = qtg.QLineEdit()
        fLayout     = qtg.QBoxLayout(qtg.QBoxLayout.LeftToRight)
        lLayout     = qtg.QBoxLayout(qtg.QBoxLayout.LeftToRight)
        leftLayout  = qtg.QBoxLayout(qtg.QBoxLayout.TopToBottom)
        winLayout   = qtg.QBoxLayout(qtg.QBoxLayout.LeftToRight)
        
        self._list.setFixedWidth(125)
        self._subl.setFixedWidth(125)
        self._xmw .setFixedWidth(125)
        self._xMw .setFixedWidth(125)

        fLayout.addWidget(self._xmw)
        fLayout.addWidget(self._xMw)
        lLayout.addWidget(self._list)
        lLayout.addWidget(self._subl)
        leftLayout.addLayout(fLayout)
        leftLayout.addLayout(lLayout)
        
        self._win.setLayout(winLayout)
        self._win.layout().addLayout(leftLayout)
        self._win.layout().addWidget(self._plt)
        self._win.show()
        
    ## The connectSignals connects the signals to the rights methods
    def connectSignals(self):
        self.update_list_sig      .connect(self._updateList)
        self.update_plot_sig      .connect(self._updatePlot)
        self.update_percentile_sig.connect(self._updatePercentile)
        self._list.itemClicked    .connect(self._itemClicked)
        self._subl.itemChanged    .connect(self._sublChanged)
        self._xmw.textChanged     .connect(self._validateBoundaries)
        self._xMw.textChanged     .connect(self._validateBoundaries)
        self._updateList()
        self._itemClicked(self._list.currentItem())
        
    ## The _validateBoundaries method validates the x min and x max inputs
    def _validateBoundaries(self):
        try:
            new_xm = float(self._xmw.text())
            self._xmw.setStyleSheet("color : #000000")
        except:
            new_xm = None
            self._xmw.setStyleSheet("color : #b11b00")
            
        try:
            new_xM = float(self._xMw.text())
            self._xMw.setStyleSheet("color : #000000")
        except:
            new_xM = None
            self._xMw.setStyleSheet("color : #b11b00")
            
        if not (new_xm is None) and \
           not (new_xM is None) :
            if new_xm >= new_xM :
                self._xmw.setStyleSheet("color : #b11b00")
                self._xMw.setStyleSheet("color : #b11b00")
                new_xm = None
                new_xM = None
            else:
                self._xmw.setStyleSheet("color : #000000")
                self._xMw.setStyleSheet("color : #000000")
            
        if (new_xM != self._xM) or (new_xm != self._xm):
            self._dataLck.acquire()
            self._xM = new_xM
            self._xm = new_xm
            self._dataLck.release()
            self._itemClicked(self._list.currentItem())
            
        
    ## The addGroup method add the given group to the list
    #
    #   @param name : The name of the group to add
    def addGroup(self, name):
        self._listLck.acquire()
        self._datas[name] = C.OrderedDict()
        self._listLck.release()
        self.update_list_sig.emit()
        
    ## The delGroup method delete the given group
    #
    #   @param name : The name of the group to delete
    def delGroup(self, name):
        try:
            self._listLck.acquire()
            del self._datas[name]
            self._listLck.release()
            self.update_list_sig.emit()
        except:
            pass

    ## The addPlot method adds a plot to the given group
    #
    #   @param group  : The name of the group the plot belongs to
    #   @param name   : The name of the new plot to add. This name should be
    #                   unique inside the given group
    #   @param cnt    : How many lines will be plotted at the same time
    #   @param styles : The style of the lines to plot. Styles must have 'cnt'
    #                   elements. The elements in 'styles' will be directly 
    #                   passed to 'pyqtgraph.mkPen'. See <a href="http://www.pyqtgraph.org/documentation/functions.html?highlight=mkpen#pyqtgraph.mkPen">pyqtgraph.mkPen</a>
    #                   documentaiton for more information
    def addPlot(self, group, name, cnt, styles):
        self._listLck.acquire()
        self._datas[group][name]            = {}
        self._datas[group][name]["x"]       = []
        self._datas[group][name]["y"]       = []
        self._datas[group][name]["type"]    = Plotter.PLOT
        self._datas[group][name]["datas"]   = []
        self._datas[group][name]["styles"]  = []
        self._datas[group][name]["enable"]  = C.OrderedDict()
        self._datas[group][name]["error"]   = None
        for i in range(cnt):
            self._datas[group][name]["y"]     .append([])
            self._datas[group][name]["styles"].append(pg.mkPen(styles[i]))
            self._datas[group][name]["enable"][i] = True
        self._listLck.release()
        self.update_list_sig.emit()
        
    ## The addPercentile methods add a percentile chart to the list.
    #
    #   @param group            : The name of the group the percentile chart
    #                             belongs to
    #   @param name             : The name of the chart
    #   @param percentagesPairs : A list of tuple of values between and
    #                             including 0 and 100. If percentagesPairs is
    #                             made of pairs (p_i, p_j)
    #                             the chart will be made of :
    #                               - A line at (x, percentile(A(x), p_i))
    #                               - A line at (x, percentile(A(x), p_j))
    #                               - An area filled between the two previous
    #                                 lines with color where the alpha value
    #                                 is proportional to p_i - p_j
    def addPercentile(self, group, name, percentagesPairs):
        self._listLck.acquire()
        self._datas[group][name]            = {}
        self._datas[group][name]["x"]       = []
        self._datas[group][name]["y"]       = {}
        self._datas[group][name]["type"]    = Plotter.PERC
        self._datas[group][name]["datas"]   = []
        self._datas[group][name]["pairs"]   = percentagesPairs
        self._datas[group][name]["styles"]  = []
        self._datas[group][name]["enable"]  = C.OrderedDict()
        self._datas[group][name]["error"]   = None
        for p in percentagesPairs:
            self._datas[group][name]["enable"][p] = True
            self._datas[group][name]["y"][p[0]] = []
            self._datas[group][name]["y"][p[1]] = []
            c = pg.mkColor(Plotter.PER_COLOR)
            c.setAlpha(max(10, int(255 * (1 - ((max(p) - min(p))/100)))))
            self._datas[group][name]["styles"].append(
                                        (pg.mkPen(Plotter.PER_COLOR, width = 1),
                                         pg.mkBrush(c)))
        self._listLck.release()
        self.update_list_sig.emit()
    
    ## The updatePlot method updates the given plot with the given coordinates
    #
    #   @param group   : The group the plot belongs to
    #   @param name    : The name of the plot to update
    #   @param x       : The x coordinate for the given y values
    #   @param yValues : The y values to plot
    def updatePlot(self, group, name, x, yValues):
        self._dataLck.acquire()
        if not np.isfinite(yValues).all():
            if self._datas[group][name]["error"] is None:
                self._datas[group][name]["error"] = "Infinite/NaN value error"
        else:
            self._datas[group][name]["x"].append(x)
            for i,y in enumerate(yValues):
                self._datas[group][name]["y"][i].append(y)
        self._dataLck.release()
        self.update_plot_sig.emit(group, name)
        
    ## The updatePercentile method updates the given percentile plot
    #
    #   @param group  : The group the plot belongs to
    #   @param name   : The name of the plot to update
    #   @param x      : The x coordinate associated to the distribution to plot
    #   @param values : The distribution to plot
    def updatePercentile(self, group, name, x, values):
        self._dataLck.acquire()
        if not np.isfinite(values).all():
            if self._datas[group][name]["error"] is None:
                self._datas[group][name]["error"] = "Infinite/NaN value error"
        else:
            self._datas[group][name]["x"].append(x)
            for p in self._datas[group][name]["pairs"]:
                self._datas[group][name]["y"][p[0]].append(
                                                    np.percentile(values, p[0]))
                if p[1] != p[0]:
                   self._datas[group][name]["y"][p[1]].append(
                                                    np.percentile(values, p[1]))
        self._dataLck.release()
        self.update_percentile_sig.emit(group, name)
        
    ## The _updateList cleans and update the list of plots
    def _updateList(self):
        self._listLck.acquire()
        self._list.clear()
        for k,v in self._datas.items():
            i = qtg.QListWidgetItem(k)
            i.setFont(Plotter.GRP_FONTS)
            i.setData(Plotter.GRP_ROLE, k)
            i.setData(Plotter.PLT_ROLE, None)
            self._list.addItem(i)
            for kp in v:
                i = qtg.QListWidgetItem("    " + kp)
                i.setFont(Plotter.PLT_FONTS)
                i.setData(Plotter.GRP_ROLE, k)
                i.setData(Plotter.PLT_ROLE, kp)
                i.setData(Plotter.ERROR_ROLE, False)
                self._list.addItem(i)
        self._listLck.release()
        
    ## The _updatePlot method recomputes the given plot according to its
    #  parameters
    #
    #   @param grp : The group the plot belongs to
    #   @param plt : The name of the plot to refresh
    def _updatePlot(self, grp, plt):
        self._dataLck.acquire()
        self._datas[grp][plt]["datas"] = []
        
        if self._datas[grp][plt]["error"] is None:
            x = self._datas[grp][plt]["x"]
            for y,s in zip(self._datas[grp][plt]["y"],
                           self._datas[grp][plt]["styles"]):
                self._datas[grp][plt]["datas"].append(
                                               pg.PlotDataItem(x, y, pen = s))
        self._dataLck.release()
        
        if not (self._list.currentItem() is None)                 and \
           self._list.currentItem().data(Plotter.GRP_ROLE) == grp and \
           self._list.currentItem().data(Plotter.PLT_ROLE) == plt :
            self._itemClicked(self._list.currentItem())
            
    ## The _updatePercentile method recomputes the given percentile plot
    #  according to its parameters
    #
    #   @param grp : The name of the group the plot belongs to
    #   @param plt : The name of the plot to refresh
    def _updatePercentile(self, grp, plt):
        self._dataLck.acquire()
        self._datas[grp][plt]["datas"] = {}
        
        if self._datas[grp][plt]["error"] is None:
            x = self._datas[grp][plt]["x"]
            for p,s in zip(self._datas[grp][plt]["pairs"],
                           self._datas[grp][plt]["styles"]):
                c1 = pg.PlotDataItem(x, self._datas[grp][plt]["y"][p[0]],
                                        pen = s[0])
                c2 = pg.PlotDataItem(x, self._datas[grp][plt]["y"][p[1]],
                                        pen = s[0])
                f  = pg.FillBetweenItem(c1, c2, s[1])
                self._datas[grp][plt]["datas"][p] = []
                self._datas[grp][plt]["datas"][p].append(c1)
                self._datas[grp][plt]["datas"][p].append(c2)
                self._datas[grp][plt]["datas"][p].append(f)
        self._dataLck.release()
            
        if not (self._list.currentItem() is None)                 and \
           self._list.currentItem().data(Plotter.GRP_ROLE) == grp and \
           self._list.currentItem().data(Plotter.PLT_ROLE) == plt :
            self._itemClicked(self._list.currentItem())  
        
    ## The _itemClicked method clear the deawing area and redraw the plots
    #  associated to the given iten
    #
    #   @param iten : The item associated to the plot we want to display
    def _itemClicked(self, item):
        self._plt .clear()
        self._subl.clear()
        
        if  (item is None) or (item.data(Plotter.PLT_ROLE) is None):
            return
        
        self._listLck.acquire()
        self._dataLck.acquire()
        grp = item.data(Plotter.GRP_ROLE)
        plt = item.data(Plotter.PLT_ROLE)
        if self._datas[grp][plt]["error"] is None:
            
            dataList = []
            yList    = []
            
            if self._datas[grp][plt]["type"] == Plotter.PLOT:
                for (k,v) in self._datas[grp][plt]["enable"].items():
                    if v:
                        dataList.append(self._datas[grp][plt]["datas"][k])
                        yList   .append(self._datas[grp][plt]["y"]    [k])
            elif self._datas[grp][plt]["type"] == Plotter.PERC:
                for (k,v) in self._datas[grp][plt]["enable"].items():
                    if v:
                        dataList.append(self._datas[grp][plt]["datas"][k][0])
                        dataList.append(self._datas[grp][plt]["datas"][k][1])
                        dataList.append(self._datas[grp][plt]["datas"][k][2])
                        yList   .append(self._datas[grp][plt]["y"][k[0]])
                        yList   .append(self._datas[grp][plt]["y"][k[1]])
            
            if len(dataList) > 0:
                for d in dataList:
                    self._plt.addItem(d)
                
                if (self._xm is None) or \
                   (self._xm > np.max(self._datas[grp][plt]["x"])) :
                    mx    = np.min(self._datas[grp][plt]["x"])
                    mx_id = 0
                else:
                    mx    = max(self._xm, np.min(self._datas[grp][plt]["x"]))
                    mx_id = np.argmax(np.array(self._datas[grp][plt]["x"]) 
                                      >= mx)
                    
                if (self._xM is None) or \
                   (self._xM < np.min(self._datas[grp][plt]["x"])) :
                    Mx    = np.max(self._datas[grp][plt]["x"])
                    Mx_id = len(self._datas[grp][plt]["x"])
                else:
                    Mx    = min(self._xM, np.max(self._datas[grp][plt]["x"]))
                    Mx_id = np.argmin(np.array(self._datas[grp][plt]["x"])
                                      < Mx)
                    Mx_id = Mx_id + 1
                
                my = None
                My = None
                
                for y in yList:
                    m = np.min(y[mx_id:Mx_id])
                    M = np.max(y[mx_id:Mx_id])
                    my = m if (my is None) else min(my, m)
                    My = M if (My is None) else max(My, M)
                
                # my = my - 0.01 * (My - my)
                # My = My + 0.01 * (My - my)
                # Mx = Mx + 0.01 * (Mx - mx)
                self._plt.setXRange(mx, Mx)
                self._plt.setYRange(my, My)
            
        elif not item.data(Plotter.ERROR_ROLE):
            item.setText(item.text() + " - " + self._datas[grp][plt]["error"])
            item.setForeground(pg.mkBrush(Plotter.FG_ERR_COLOR))
            item.setFont(Plotter.ERROR_FONTS)
            item.setData(Plotter.ERROR_ROLE, True)
        self._dataLck.release()
        self._listLck.release()
        self._updateSubList(grp, plt)

    def _updateSubList(self, grp, plt):
        self._listLck.acquire()
        self._dataLck.acquire()
        for (k,v) in self._datas[grp][plt]["enable"].items():
            s = qtc.Qt.Checked if v else qtc.Qt.Unchecked
            i = qtg.QListWidgetItem(str(k))
            i.setFont(Plotter.SUB_PLT_FONTS)
            i.setCheckState(s)
            i.setData(Plotter.GRP_ROLE, grp)
            i.setData(Plotter.PLT_ROLE, plt)
            i.setData(Plotter.SUB_ID  , k)
            self._subl.addItem(i)
        self._dataLck.release()
        self._listLck.release()
        
    def _sublChanged(self, item):
        if item.checkState() == qtc.Qt.PartiallyChecked:
            item.setCheckState(qtc.Qt.Unchecked)
            return
        
        self._listLck.acquire()
        self._dataLck.acquire()
        grp   = item.data(Plotter.GRP_ROLE)
        plt   = item.data(Plotter.PLT_ROLE)
        subId = item.data(Plotter.SUB_ID)
        s     = (item.checkState() == qtc.Qt.Checked)
        self._datas[grp][plt]["enable"][subId] = s
        self._dataLck.release()
        self._listLck.release()
        self._itemClicked(self._list.currentItem())

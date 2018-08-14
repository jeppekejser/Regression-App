import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
import pandas as pd
import platform

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
import seaborn as sns

#Tilte
title = 'Regressions Appen'

#Cross Platform Code
os = platform.system()

style = tk

"""
if os == 'Darwin':
    style = tk
elif os == 'Windows':
    style = ttk
else:
    style = tk
"""
#Fonts
LARGE_FONT = ('Verdana', 10, 'bold')
EXTRA_LARGE_FONT = ('Verdana', 12, 'bold')
MAIN_FONT = ('Verdana', 10)

class RegressionApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        if os == 'Windows':
            tk.Tk.iconbitmap(self, default='favicon.ico')
        
        tk.Tk.wm_title(self, title)
        tk.Tk.minsize(self, 800, 600)
        #tk.Tk.geometry(self,'1000x750')

        container = tk.Frame(self)

        container.grid(row=0, column=0, sticky='nsew')

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        frame = StartPage(container, self)

        self.frames[StartPage] = frame

        frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()
    


class StartPage(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        
        
        self.regression_types = ['Lineær Regression']
        
        self.seperation_types = [';', ',']
        
        regression_type_label = style.Label(self, text='Vælg regressions form:')
        regression_type_label.grid(row=0, column=0, sticky='nw')
        
        self.regression_options = ttk.Combobox(self, values=self.regression_types)
        self.regression_options.grid(row=0, column=1, sticky='nw')
        
        regression_process_btn = style.Button(self, text='Lav Regression', command=lambda: self.RegressionOption())
        regression_process_btn.grid(row=2, column=0, sticky='nw')
        
        
        select_file = style.Button(self, text='Vælg fil', command=lambda : self.OpenFile())
        select_file.grid(row=8, column=0, sticky='sw', padx=2.5, pady=15)
        
        regression_type_label = style.Label(self, text='Vælg typen af seperation din data bruger:')
        regression_type_label.grid(row=8, column=1, sticky='sw', padx=0, pady=15)
        
        self.seperation_options = ttk.Combobox(self, values=self.seperation_types)
        self.seperation_options.grid(row=8, column=2, sticky='sw', padx=0, pady=12.5)

        quit_btn = style.Button(self, text='Luk Programmet', command=lambda: quit(self))
        quit_btn.grid(row=12, column=0, sticky='sw', padx=2.5, pady=10)

        self.instruction_label = style.Label(self, text='Instruktioner', font=EXTRA_LARGE_FONT, justify=tk.LEFT)
        self.instruction_label.grid(row=3, column=1, sticky='w', padx=2, pady=0)

        self.instruction_txt = style.Label(self, text='Brug . til decimaler i din fil\nSørg for at vælge seperations type før du vælger fil\nVælg x og y-akse samt regressions type, før du trykker "Lav regression"', font=MAIN_FONT, justify=tk.LEFT)
        self.instruction_txt.grid(row=4, column=1, sticky='w', padx=2, pady=0)

        self.update()
        
        
    def OpenFile(self):
        self.filename = tk.filedialog.askopenfilename(filetypes=[('CSV Fil', '.csv')], title='Vælg din .csv data fil')
        
        """
        if self.filename == '':
            self.file_status = 'Ingen fil blev valgt'

        else:
            self.file_status = 'Filen \"{0}\" blev åbnet'.format(self.filename)
        
        self.file_found = style.Label(self, text=self.file_status, font=LARGE_FONT)
        self.file_found.grid(row=6, column=0, sticky='se', padx=0, pady=2.5)
        """
        
        self.sep_opts = str(self.seperation_options.get())
        

        self.data = pd.read_csv(self.filename, sep=self.sep_opts)
        
        x_data_label = style.Label(self, text='Vælg data til x-akse:')
        y_data_label = style.Label(self, text='Vælg data til y-akse:')
        x_data_label.grid(row=1, column=0, sticky='sw', pady=5)
        y_data_label.grid(row=1, column=2, sticky='sw', pady=5)
        
        self.x_value_options = ttk.Combobox(self, values=list(self.data.columns))
        self.y_value_options = ttk.Combobox(self, values=list(self.data.columns))
        self.x_value_options.grid(row=1, column=1, sticky='sw', padx=2.5, pady=2.5)
        self.y_value_options.grid(row=1, column=3, sticky='sw', padx=2.5, pady=2.5)
        
        return self.filename
    

    def RegressionOption(self):

        identifier = str(self.regression_options.get())
        
        self.data = self.data.astype(float)
        
        self.x_axis = str(self.x_value_options.get())
        self.y_axis = str(self.y_value_options.get())
        
        if identifier == 'Lineær Regression':
            
            if self.filename == '':
                pass #throw an error
            
            X_train, X_test, y_train, y_test = train_test_split(self.data[self.x_axis], self.data[self.y_axis], test_size=0.2, random_state=0)
            
            X_train = X_train.values
            X_test = X_test.values
            
            self.regressor = LinearRegression(fit_intercept=True)

            X_train = np.reshape(X_train, (-1, 1))
            X_test = np.reshape(X_test, (-1, 1))

            self.regressor.fit(X_train, y_train)

            pred_for_score = self.regressor.predict(X_test)

            self.a = self.regressor.coef_[0]
            self.b = self.regressor.intercept_

            self.ln_func = 'Funktionsforskriften: f(x) = {0:.2f}x + {1:.2f}'.format(self.a, self.b)
            self.r2_score_text = 'R^2 score: %.4f' % r2_score(y_test, pred_for_score)

            ln_func_label = style.Label(self, text=self.ln_func, font=MAIN_FONT)
            ln_func_label.grid(row=4, column=1, sticky='nw', pady=3)
            r2_score_label = style.Label(self, text=self.r2_score_text, font=MAIN_FONT)
            r2_score_label.grid(row=4, column=1, sticky='ne', pady=3)

            self.plot_fig, (ax1) = plt.subplots(1, figsize=(7,5))
            self.graph2 = sns.regplot(x=self.x_axis, y=self.y_axis, data=self.data, ci=None, ax=ax1).set_title('Lineær Reggression')
            #self.graph1 = plt.scatter(self.data[self.x_axis], self.data[self.y_axis],  color='black', ax=ax2)
            #self.graph1.plot(X_test, pred_for_score, color='red', linewidth=1.5)

            self.canvas = FigureCanvasTkAgg(self.plot_fig, self)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=3, column=1, sticky='nw')

            new_y_exp_label = style.Label(self, text='Tast x-værdi ind som du ønsker at vide en y-værdi af', font=MAIN_FONT)
            new_y_exp_label.grid(row=5, column=0, sticky='sw')

            self.get_y_value = style.Button(self, text='Få y-værdi', command=lambda: self.PredNewyFunc())
            self.get_y_value.grid(row=6, column=0, sticky='sw')

            self.x_value = style.Entry(self)
            self.x_value.grid(row=6, column=0, sticky='se')

            save_graph_btn = style.Button(self, text='Gem graf', command=lambda: self.SavePlot())
            save_graph_btn.grid(row=6, column=1, sticky='sw', padx=2.5)

            self.instruction_label.destroy()

            self.instruction_txt.destroy()

            self.update()

    def PredNewyFunc(self):

        self.pred_value = self.x_value.get()

        self.pred_value = float(self.pred_value)

        self.pred_new_y = self.regressor.predict(self.pred_value)

        df = pd.DataFrame({self.x_axis: self.pred_value, self.y_axis: self.pred_new_y})

        self.data = self.data.append(df)

        new_vals_txt = 'Den nye y-værdi er: {0:.2f}, det vil sige det nye punkt er ({1}, {0:.2f})'.format(float(self.pred_new_y), self.pred_value)

        new_vals_label = style.Label(self, text=new_vals_txt, font=MAIN_FONT)
        new_vals_label.grid(row=7, column=0, sticky='se')

        self.RefreshFigure(self.data[self.x_axis], self.data[self.y_axis])

    def RefreshFigure(self, x, y):
        #self.graph1.set_data(x,y)
        self.plot_fig, (ax1) = plt.subplots(1)
        self.graph2 = sns.regplot(x=x, y=y, data=self.data, ci=None, ax=ax1).set_title('Lineær Reggression')
        #ax = self.canvas.figure.axes[0]
        #ax.set_xlim(x.min(), x.max())
        #ax.set_ylim(y.min(), y.max())
        self.canvas = FigureCanvasTkAgg(self.plot_fig, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=3, column=1, sticky='nw')
        self.update()

    def SavePlot(self):
        foldername = filedialog.askdirectory(title='Vælg hvilken mappe din graf skal gemmes i')
        self.plot_fig.savefig(foldername + '/graf.png')

    def quit(self):
        quit()

app = RegressionApp()
app.mainloop()

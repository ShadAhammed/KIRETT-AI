import tkinter as tk
from tkinter.filedialog import askopenfilename
import pandas as pd

class SelectData:
    def __init__(self, complication):
        self.complication= complication
        print(f'\nWorking with {complication} complication\n')

    def SelectFile(self):
        root= tk.Tk()
        filename = askopenfilename()
        root.destroy()
        Data= pd.read_excel(filename, index_col= 0 )

        return Data



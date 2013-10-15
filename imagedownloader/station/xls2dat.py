
import csv
import sys
from xlrd import open_workbook, empty_cell, XL_CELL_NUMBER

wb =  open_workbook(sys.argv[1])
'''  sys.argv[1] ->  nombre de la planilla excel a procesar  '''

sheet= wb.sheet_by_index(1)
#le = len(sheet.col(0))

col_i = 0
fila_i = 8

matriz = []

def dato(line):
    try:
        return sheet.cell(line,9).value
    except  ValueError:
        return none

for line in xrange(8, 5000):
   #if sheet.cell_type(line, 0) is XL_CELL_NUMBER: 
   #if sheet.cell_type(line, 0) is not empty_cell:
   #while sheet.cell(line, 0).value:
    try: 
        print line
        if sheet.cell(line,0).value == 111 :
            year = int(sheet.cell(line,1).value)
            dia =  int(sheet.cell(line,2).value)
            hora = int(sheet.cell(line,3).value)
            #dato = sheet.cell(line,9).value
            matriz.append((year, dia, hora, dato(line)))
    except IndexError:
        pass

#with open('test.csv', 'a', newline='') as fp:
with open('Mayo_2011.csv', 'a') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(matriz)


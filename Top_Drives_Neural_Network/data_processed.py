import numpy as np

data = np.zeros((435, 11))
results = np.zeros((435, 1))

f = open(".venv/Top_Drives_Neural_Network/data.txt", "r")
f.readline()
for i in range(435):
    dataStr = f.readline()
    split = dataStr.split()

    if(split[6]=='No'):
        split[6] = '0'
    elif(split[6]=='Yes'):
        split[6] = '1'
    else:
        raise Exception("Traction Control not Yes or No")

    if(split[7]=='No'):
        split[7] = '0'
    elif(split[7]=='Yes'):
        split[7] = '1'
    else:
        raise Exception("ABS not Yes or No")

    if(split[8]=='Low'):
        split[8] = '1'
    elif(split[8]=='Medium'):
        split[8] = '2'
    elif(split[8]=='High'):
        split[8] = '3'
    else:
        raise Exception("Clearance not an option")
    
    if(split[9]=='FWD'):
        split[9] = '1'
    elif(split[9]=='RWD'):
        split[9] = '2'
    elif(split[9]=='4WD'):
        split[9] = '3'
    else:
        raise Exception("Drive Style not an option")

    if(split[10]=='Slick'):
        split[10] = '1'
    elif(split[10]=='Performance'):
        split[10] = '2'
    elif(split[10]=='Standard'):
        split[10] = '3'
    elif(split[10]=='All-Surface'):
        split[10] = '4'
    elif(split[10]=='Off-Road'):
        split[10] = '5'
    else:
        raise Exception("Drive Style not an option")

    as_num = [float(numeric_string) for numeric_string in split]
    results[i][0] = as_num[11]
    for j in range(11):
        data[i][j] = as_num[j]

data_formatted = data.T
results_formatted = results.T
f.close()
import os
import pandas as pd
import numpy as np
import helper_functions as hf


def rotation_angles(rotation_matrix):
    """
    Calculate rotation angles (in degrees) from a given rotation matrix.

    Args:
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        tuple: Tuple containing rotation angles around the X, Y, and Z axes (in degrees).
    """
    # Extract rotation angles
    angle_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    angle_y = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
    angle_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert angles from radians to degrees
    angle_x_deg = np.degrees(angle_x)
    angle_y_deg = np.degrees(angle_y)
    angle_z_deg = np.degrees(angle_z)

    # Return the rotation angles
    return angle_x_deg, angle_y_deg, angle_z_deg





def calc_features(coordinates,fps):
    #1) Summe aller Bewegungen -Richtung pro Sekunde
    #2) Summe aller Bewegungen in +Richtung pro Sekunde
    #3) Maximale Bewegung am Stück in - Richtung
    #4) Maximale Bewegung am Stück in + Richtung
    #5) Maximale Bewegung am Stück in - Richtung geteilt durch Dauer der Bewegung(Anzahl der Bewegungen * 1/Fps)
    #6) Maximale Bewegung am Stück in + Richtung geteilt durch Dauer der Bewegung(Anzahl der Bewegungen * 1/Fps)
    #7) Mean Derivative - = Summe aller Bewegungen in - Richtung  durhc BEwegungen in - Richtung
    #8) Mean Derivative + = Summe aller Bewegungen in + Richtung  durhc BEwegungen in + Richtung
    #9) Max Derivative - Maximale Bewegung von einem Frame zum nächsten in - Richtung (*1/fps)
    #10) Max Derivative + Maximale Bewegung von einem Frame zum nächsten in + Richtung (*1/fps)
    #11) Max Distance: Größter - kleinster Value pro Second
    #12) Variance of values pro second
    negsum=pd.DataFrame() #1
    possum = pd.DataFrame() #2
    maxnegmov = pd.DataFrame()#3
    maxposmov = pd.DataFrame()#4
    maxnegmovpertime = pd.DataFrame()#5
    maxposmovpertime = pd.DataFrame() #6
    meanderneg = pd.DataFrame() #7
    meanderpos = pd.DataFrame() #8
    maxderneg = pd.DataFrame() #9
    maxderpos = pd.DataFrame() #10
    max_dist = pd.DataFrame() #11
    var = pd.DataFrame() #12
    #for each collumn 
    for col in coordinates.columns:
        if col!="frame":
            Cnegsum= [] #1
            Cpossum = [] #2
            Cmaxnegmov = [] #3
            Cmaxposmov = [] #4
            Cmaxnegmovpertime = [] #5
            Cmaxposmovpertime = [] #6
            Cmeanderneg = [] #7
            Cmeanderpos = [] #8
            Cmaxderneg = [] #9
            Cmaxderpos = [] #10
            Cmax_dist =  [] #11
            Cvar = [] #12

            #for each second  = fps frames
            for j in range(0,int(len(coordinates)/fps)):
                # get next fps values
                values = np.asarray(coordinates[col].iloc[j*fps:j*fps+fps])  
                #print(values)
                if values.size > 0:
                    dists = [values[i] -values[i+1] for i in range(len(values)-1)]
                    #print(dists)
                    neg_dists = [d for d in dists if d <0]
                    pos_dists = [d for d in dists if d >=0]
                    #1
                    sum_neg_dists = sum(neg_dists)
                    #2
                    sum_pos_dists = sum(pos_dists)
                    #7
                    if len(neg_dists)>0:
                        mean_derivitive_neg = sum_neg_dists/(len(neg_dists)*1/fps)#achtung hiere 1/ neu
                    else:
                        mean_derivitive_neg = 0
                    #8
                    if len(pos_dists)>0:
                        mean_derivitive_pos = sum_pos_dists/(len(pos_dists)*1/fps)#achtun hier 1/neu
                    else:
                        mean_derivitive_pos = 0
                    #9
                    if len(neg_dists)>0:
                        maxdernegv = min(neg_dists)/fps
                    else:
                        maxdernegv = 0
                    if len(pos_dists)>0: 
                    #10
                        maxderposv = max(pos_dists)/fps
                    else: 
                        maxderposv = 0

                    #11
                    maxdistv = abs(max(values)-min(values)) #here abs neu

                    #12
                    varv = np.var(values)
                    if len(pos_dists)>0 and (sum(pos_dists)>0):
                        # maximale bewegung am stück in + richtung
                        max_sum = 0.0
                        max_length = 0
                        current_sum = 0
                        current_length = 0

                        for num in dists:
                            if num >0:
                                current_sum += num
                                current_length +=1
                            else:
                                #negative zahl gefunden, überprüfe auf neue maximale Summe 
                                if current_sum > max_sum or (current_sum == max_sum and current_length < max_length):
                                    max_sum  = current_sum
                                    max_length = current_length
                                #setze summe und länge zurück
                                current_sum = 0.0
                                current_lengh = 0
                        #überprüfe am Ende noch mal auf maximale SUmme:
                        if current_sum > max_sum or (current_sum == max_sum and current_length < max_length):
                            #4
                            max_sum = current_sum
                            max_length = current_length
                        #6
                        max_sum_per_time = max_sum/(max_length/fps)
                    else:
                        max_sum = 0
                        max_length = 0
                        max_sum_per_time = 0

                    if len(neg_dists)>0:
                        # maximale bewegung am stück in - richtung
                        min_sum = 0.0
                        min_length = 0
                        current_sum = 0
                        current_length = 0

                        for num in dists:
                            if num <0:
                                current_sum += num
                                current_length +=1
                            else:
                                #positive zahl gefunden, überprüfe auf neue maximale Summe 
                                if current_sum < max_sum or (current_sum == min_sum and current_length < min_length):
                                    min_sum  = current_sum
                                    min_length = current_length
                                #setze summe und länge zurück
                                current_sum = 0.0
                                current_lengh = 0
                        #überprüfe am Ende noch mal auf min SUmme:
                        if current_sum < min_sum or (current_sum == min_sum and current_length < min_length):
                            #3
                            min_sum = current_sum
                            min_length = current_length
                        #5
                        min_sum_per_time = min_sum/(min_length/fps)
                    else:
                        min_sum = 0
                        min_length = 0
                        min_sum_per_time = 0

                    Cnegsum.append(sum_neg_dists)
                    Cpossum.append(sum_pos_dists)
                    Cmaxnegmov.append(min_sum)
                    Cmaxposmov.append(max_sum)
                    Cmaxnegmovpertime.append(min_sum_per_time)
                    Cmaxposmovpertime.append(max_sum_per_time)
                    Cmeanderneg.append(mean_derivitive_neg)
                    Cmeanderpos.append(mean_derivitive_pos)
                    Cmaxderneg.append(maxdernegv)#9
                    Cmaxderpos.append(maxderposv)#10
                    Cmax_dist.append(maxdistv) #11
                    Cvar.append(varv) #12
                else:
                    Cnegsum.append(0)
                    Cpossum.append(0)
                    Cmaxnegmov.append(0)
                    Cmaxposmov.append(0)
                    Cmaxnegmovpertime.append(0)
                    Cmaxposmovpertime.append(0)
                    Cmeanderneg.append(0)
                    Cmeanderpos.append(0)
                    Cmaxderneg.append(0)#9
                    Cmaxderpos.append(0)#10
                    Cmax_dist.append(0) #11
                    Cvar.append(0) #12



            #coords_visible[col[:-1]]=coordinates[col].copy()
            negsum[col] = Cnegsum
            possum[col] = Cpossum
            maxnegmov[col] = Cmaxnegmov
            maxposmov[col] = Cmaxposmov
            maxnegmovpertime[col] = Cmaxnegmovpertime
            maxposmovpertime[col] = Cmaxposmovpertime
            meanderneg[col] = Cmeanderneg#7
            meanderpos[col] = Cmeanderpos#8
            maxderneg[col] = Cmaxderneg #9
            maxderpos[col] = Cmaxderpos #10
            max_dist[col] = Cmax_dist #11
            var[col] = Cvar#12
    return negsum, possum, maxnegmov, maxposmov, maxnegmovpertime, maxposmovpertime, meanderneg, meanderpos, maxderneg, maxderpos, max_dist, var


def calc_all_features(folder,folder_extension, certain_subjects, override, fps):
    window_size = 3

    checkpath = os.path.join(folder, folder_extension + "_Sum_neg_movements_X" )
    for data_name in os.listdir(os.path.join(folder, "Coordinates_X_" + folder_extension)):
         if certain_subjects == []:
            certain_subjects = ["_"]
         if any(string in data_name for string in certain_subjects):
                # if override is false, check if coordinates already exist and skip this video
                if override == False:
                    if os.path.exists(os.path.join(checkpath, data_name)):
                        print("file " + data_name  + " exists, skipping")
                        continue
                print("data name", data_name)


                #load data                    
                X = pd.read_csv(os.path.join(folder,"Coordinates_X_" + folder_extension, data_name))
                Y = pd.read_csv(os.path.join(folder,"Coordinates_Y_" + folder_extension, data_name))
                Z = pd.read_csv(os.path.join(folder,"Coordinates_Z_" + folder_extension, data_name))
                T = np.load(folder + "Transformation_matrix_arrays/" + data_name + ".npy", allow_pickle = True)


                # TRANSLATION
                #if "center" in folder_extension:             
                Xold = pd.read_csv(os.path.join(folder, "Coordinates_X_prepared", data_name))
                Yold = pd.read_csv(os.path.join(folder, "Coordinates_Y_prepared", data_name))
                Zold = pd.read_csv(os.path.join(folder, "Coordinates_Z_prepared", data_name))

                #add translation (coordinate of middle point before transformations
                X["translation"] = Xold["0"].rolling(window_size, center=True, min_periods=1).median()
                Y["translation"] = Yold["0"].rolling(window_size, center=True, min_periods=1).median()
                Z["translation"] = Zold["0"].rolling(window_size, center=True, min_periods=1).median()

                X = X.drop(columns = "Unnamed: 0")
                Y = Y.drop(columns = "Unnamed: 0")
                Z = Z.drop(columns = "Unnamed: 0")

                # ROTATION
                #calculate and add rotation columns
                rotx = []
                roty = []
                rotz = []

                for i in range(0,len(T)):

                    if type(T[i]) == float:
                        rotx.append(0)
                        roty.append(0)
                        rotz.append(0)
                    else:
                        x,y,z = rotation_angles(T[i])
                        rotx.append(x)
                        roty.append(y)
                        rotz.append(z)

                X["Rot"] = pd.Series(rotx).rolling(window_size, center=True, min_periods=1).median()
                Y["Rot"] = pd.Series(roty).rolling(window_size, center=True, min_periods=1).median()
                Z["Rot"] = pd.Series(rotz).rolling(window_size, center=True, min_periods=1).median()
                
                if len(X) < 5000:
                    print("cutting to 3750")
                    X = X.iloc[:3750]
                    Y = Y.iloc[:3750]
                    Z = Z.iloc[:3750]

                for feat in ["Sum_neg_movements", "Sum_pos_movements", "Max_neg_movements", "Max_pos_movements", "Max_neg_mov_per_time","Max_pos_mov_per_time",\
                       "Mean_derivative_neg","Mean_derivative_pos","Max_derivative_neg","Max_derivative_pos","Max_dist","Var"]:
                    for ax in ["_X", "_Y", "_Z"]:
                        hf.create_path_if_not_existent(os.path.join(folder, feat + ax))
                for ax in ["X","Y","Z"]:
                    if ax =="X":
                        data = X
                    elif ax =="Y":
                        data = Y
                    else:
                        data = Z

                    negsum, possum, maxnegmov, maxposmov, maxnegmovpertime, maxposmovpertime, meanderneg, meanderpos, maxderneg, maxderpos, max_dist, var = calc_features(data,fps)

                    savepath = os.path.join(folder,"Sum_neg_movements_" + ax,data_name)
                    print(savepath)
                    negsum.to_csv(savepath )

                    savepath = os.path.join(folder,"Sum_pos_movements_" + ax,data_name)
                    possum.to_csv(savepath )

                    savepath = os.path.join(folder, "Max_neg_movements_" + ax,data_name)
                    maxnegmov.to_csv(savepath )

                    savepath = os.path.join(folder,"Max_pos_movements_" + ax,data_name)
                    maxposmov.to_csv(savepath) 

                    savepath = os.path.join(folder, "Max_neg_mov_per_time_" + ax,data_name)
                    maxnegmovpertime.to_csv(savepath )

                    savepath = os.path.join(folder,"Max_pos_mov_per_time_" + ax,data_name)
                    maxposmovpertime.to_csv(savepath)

                    savepath = os.path.join(folder,"Mean_derivative_neg_" + ax,data_name)
                    meanderneg.to_csv(savepath )

                    savepath = os.path.join(folder, "Mean_derivative_pos_" + ax,data_name)
                    meanderpos.to_csv(savepath )

                    savepath = os.path.join(folder,"Max_derivative_neg_" + ax,data_name)
                    maxderneg.to_csv(savepath )

                    savepath = os.path.join(folder,"Max_derivative_pos_" + ax,data_name)
                    maxderpos.to_csv(savepath )

                    savepath = os.path.join(folder,f"Var_" + ax,data_name)
                    var.to_csv(savepath )

                    savepath = os.path.join(folder,"Max_dist_" + ax,data_name)
                    max_dist.to_csv(savepath )
                    
                    
                    
                    
def create_all_data_from_folder(folder_name, save_folder,calc_mean = True):
    data=[]


    print("feature folder", folder_name)
    for data_name in os.listdir(folder_name):

            coordinates = pd.read_csv(os.path.join(folder_name, data_name))
            coordinates["ID"] = data_name + "_"

            coordinates["ID_idx"] = [str(i).zfill(3) for i in range(0,len(coordinates))]
            coordinates["ID"] = coordinates["ID"]+coordinates["ID_idx"].astype(str)
            coordinates = coordinates.drop(columns = "ID_idx")
            data.append(coordinates)

    d = pd.concat(data, ignore_index = True) 
    d = d.loc[:,~d.columns.str.contains('Unnamed', case=False)] 



    #calculate mean:
    if calc_mean:

        d["ID1"] = d["ID"].copy().str.split('.csv').str[0]
        for col in d.columns:
            if col not in ["ID", "ID1"]:
                #
                d[col + "_mean"] = d.groupby("ID1")[col].transform(lambda x: x.abs().mean())
        d = d.drop(columns = "ID1")
        d.to_csv(os.path.join(save_folder, os.path.basename(folder_name)+"_all.csv"), index = False)

    else:
        d.to_csv(os.path.join(save_folder, os.path.basename(folder_name)+"_all_without_mean.csv"), index = False)
    print("len final save data", len(d))
    print("len list of data?", len(data))

def create_all_data_for_all_features(folder_name, folder_extension, calc_mean = True, override= False):

    for func in np.flip(["Max_derivative_neg", "Max_derivative_pos","Max_dist","Max_neg_movements", "Max_pos_movements", "Mean_derivative_neg","Max_pos_mov_per_time", "Max_neg_mov_per_time","Mean_derivative_pos","Sum_neg_movements","Sum_pos_movements","Var"]):
        for ax in ["X","Y","Z"]:
            
            savepath = os.path.join(folder_name, "All_Features")
            hf.create_path_if_not_existent(savepath)

            # If override is False, check if prepared coordinates already exist and skip this video
            if calc_mean:
                print("checkpath",os.path.join(savepath,folder_extension + "_" + func + "_" + ax + "_all.csv"))

                if not override and os.path.exists(os.path.join(savepath,func+ "_" + ax + "_all.csv")):
                    print("File " +  func + "_" + ax + "_all.csv" + " exists, skipping")
                    continue
            else:
                if not override and os.path.exists(os.path.join(savepath,func+ "_" + ax + "_all_without_mean.csv")):
                    print("File " + func+ "_" + ax + "_all_without_mean.csv" + " exists, skipping")
                    continue
                                                   
                            
            featfolder = os.path.join(folder_name, func + "_" + ax)

            
            create_all_data_from_folder(featfolder,savepath,calc_mean)



